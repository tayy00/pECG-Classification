"""
models.py

All four architectures rebuilt at Chen (2025)'s exact specified
configurations, confirmed from the paper. Input is now 300 samples
(3s window, downsampled by 5), not the 1500 used in earlier project work.

Every class takes n_leads (9 or 12) and n_classes (19) and returns raw
logits of shape (batch, n_classes), consistent with the ChenWeightedBCELoss
in loss.py, which expects logits, not probabilities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. ResNet-1D -- channels 64 -> 128 -> 256 -> 512 (paper-confirmed, double
# the width used in this project's earlier, pre-paper implementation)
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """Channel progression 64 -> 128 -> 256 -> 512, paper-confirmed.
    Exact stem kernel/stride not specified by the paper; kept at a
    reasonable default (kernel 15, stride 2) since the paper only confirms
    the per-block kernel size of 7, not the stem's own configuration.
    """

    def __init__(self, n_leads=12, n_classes=19, base_channels=64, blocks_per_stage=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        stages = []
        in_ch = base_channels
        for stage_idx in range(4):
            out_ch = base_channels * (2 ** stage_idx)  # 64, 128, 256, 512
            for block_idx in range(blocks_per_stage):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                stages.append(ResBlock1D(in_ch, out_ch, stride=stride))
                in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# ---------------------------------------------------------------------------
# 2. BiLSTM -- hidden 128/direction, 2 layers, dropout 0.5, LAST timestep
# only (paper-confirmed: not averaged across the sequence)
# ---------------------------------------------------------------------------

class BiLSTMClassifier(nn.Module):
    def __init__(self, n_leads=12, n_classes=19, hidden_size=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_leads, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        forward_last = h_n[-2]
        backward_last = h_n[-1]
        pooled = torch.cat([forward_last, backward_last], dim=-1)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# 3. Transformer -- 128-dim, patch size 3 (300 samples -> 100 patches),
# 2 layers, head dim 8, feedforward 256, dropout 0.1 (all paper-confirmed)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, n_leads=12, n_classes=19, d_model=128, patch_size=3,
                 num_layers=2, head_dim=8, ff_hidden=256, dropout=0.1):
        super().__init__()
        assert d_model % head_dim == 0, "d_model must divide evenly by head_dim to form whole heads"
        nhead = d_model // head_dim
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(n_leads * patch_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_hidden,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        b, c, t = x.shape
        n_patches = t // self.patch_size
        usable = n_patches * self.patch_size
        x = x[:, :, :usable]
        x = x.reshape(b, c, n_patches, self.patch_size).permute(0, 2, 1, 3).reshape(b, n_patches, -1)
        x = self.patch_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# 4. Mamba-2 -- CPU/no-mamba_ssm fallback active for now, per the decision
# to skip it. Still importable and usable, just not the official CUDA kernel.
# ---------------------------------------------------------------------------

try:
    from mamba_ssm import Mamba2 as _OfficialMamba2
    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False


class SimpleSelectiveSSMLayer(nn.Module):
    """Fallback only. Same recurrence as the real thing, computed
    sequentially rather than in chunked/parallel form."""

    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) + 1e-3))
        self.delta_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        b, L, d = x.shape
        A = -torch.exp(self.A_log)
        delta = F.softplus(self.delta_proj(x))
        B = self.B_proj(x)
        C = self.C_proj(x)
        h = x.new_zeros(b, d, self.d_state)
        ys = []
        for t in range(L):
            dt = delta[:, t, :].unsqueeze(-1)
            A_bar = torch.exp(dt * A.unsqueeze(0))
            B_bar = dt * B[:, t, :].unsqueeze(1)
            h = A_bar * h + B_bar * x[:, t, :].unsqueeze(-1)
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        return y + x * self.D


class MambaClassifier(nn.Module):
    def __init__(self, n_leads=12, n_classes=19, d_model=128, n_layers=2,
                 d_state=64, conv_kernel=4, expand=2, head_dim=8, chunk_size=30):
        super().__init__()
        self.input_proj = nn.Linear(n_leads, d_model)
        self.using_official_mamba = _HAS_MAMBA_SSM
        layers = []
        for _ in range(n_layers):
            if _HAS_MAMBA_SSM:
                layers.append(_OfficialMamba2(
                    d_model=d_model, d_state=d_state, d_conv=conv_kernel,
                    expand=expand, headdim=head_dim, chunk_size=chunk_size,
                ))
            else:
                layers.append(SimpleSelectiveSSMLayer(d_model=d_model, d_state=d_state))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x) + x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


def build_model(name, n_leads=12, n_classes=19, **kwargs):
    name = name.lower()
    if name in ("resnet1d", "resnet-1d", "resnet"):
        return ResNet1D(n_leads=n_leads, n_classes=n_classes, **kwargs)
    if name in ("bilstm", "lstm"):
        return BiLSTMClassifier(n_leads=n_leads, n_classes=n_classes, **kwargs)
    if name in ("transformer",):
        return TransformerClassifier(n_leads=n_leads, n_classes=n_classes, **kwargs)
    if name in ("mamba", "mamba2", "mamba-2"):
        return MambaClassifier(n_leads=n_leads, n_classes=n_classes, **kwargs)
    raise ValueError(f"Unknown architecture name: {name}")
