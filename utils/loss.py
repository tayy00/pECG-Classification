"""
loss.py

Chen (2025)'s exact weighted BCE formula, confirmed directly from the paper:

    L^{i,c}_BCE = (1 - y_ic) * log(1 + e^{z_ic}) + y_ic * w_c * log(1 + e^{-z_ic})

Only the POSITIVE term is weighted; the negative term has implicit weight 1.
This is different from a standard symmetric class-weighted BCE (which
weights both terms equally), and different from what this project's
earlier notebook work used (inverse-frequency, normalized to mean 1).

Weight per class:
    w_c = min( (N - sum_i y_ic) / max(1, sum_i y_ic), tau )   with tau = 100

This is a negative-to-positive count ratio, clipped at 100, not
inverse-frequency normalized.

Mathematical note on why this formula is written the way it is: the paper's
own form, log(1+e^z) for the negative term and log(1+e^{-z}) for the
positive term, is algebraically the softplus formulation of binary cross
entropy from logits. It is equivalent to the more common
sigmoid-then-log form, just written to be numerically stable directly from
raw logits (avoids computing sigmoid separately, which can under/overflow
for large |z|). Implemented here via PyTorch's own numerically-stable
softplus rather than a naive log(1+exp(z)), for the same numerical-safety
reason the paper's own formulation exists.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_chen_class_weights(train_labels: np.ndarray, tau: float = 100.0) -> torch.Tensor:
    """train_labels: (n_samples, n_classes) binary. Returns w_c per class,
    exactly Chen's formula: negative-to-positive ratio, clipped at tau.
    """
    n = train_labels.shape[0]
    pos_counts = train_labels.sum(axis=0)
    neg_counts = n - pos_counts
    weights = neg_counts / np.maximum(1, pos_counts)
    weights = np.minimum(weights, tau)
    return torch.tensor(weights, dtype=torch.float32)


class ChenWeightedBCELoss(nn.Module):
    """Chen's exact formula: only the positive term is scaled by w_c."""

    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        neg_term = (1 - targets) * F.softplus(logits)
        pos_term = targets * self.class_weights * F.softplus(-logits)
        return (neg_term + pos_term).mean()
