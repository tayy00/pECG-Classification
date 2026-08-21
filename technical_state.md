# Technical State: Environment, Data Schema, Confirmed Parameters

## HPC environment (working, confirmed configuration)

Cluster: no job scheduler, jobs run directly. 2× Tesla V100S-PCIE-32GB. Working directory `~/ecg_work`, venv at `~/ecg_work/.ecg_venv`.

**Confirmed working:** Python 3.12.3, `torch==2.5.1+cu121`, `wfdb==4.3.1`, pandas/numpy/scikit-learn/scipy, `iterative-stratification`.

**Critical, easy to lose, must be set every fresh session before building any model:**
```python
torch.backends.cudnn.enabled = False
```
Without this, training crashes on the first `Conv1d` call: `RuntimeError: GET was unable to find an engine to execute this computation`, later confirmed with `benchmark=True` too (`FIND was unable to find an engine`, a broader search, same result). Root cause: cuDNN has no working convolution kernel for compute capability 7.0 (V100/Volta) in this install. Confirmed NOT a fixable setting, tested both the default heuristic and the benchmark-search mode, both fail identically. With cuDNN disabled, ResNet-1D (8.7M params) still trains at ~0.6 min/epoch, full 200-epoch run ≈ 2 hours, entirely practical.

**torch version history, don't let pip silently change this again:** started at 2.5.1+cu121 (confirmed working). Installing `mamba-ssm` without `--no-deps` silently upgraded it to 2.13.0+cu130, which turned out to have **dropped V100 kernel support entirely** (confirms as `CC 7.0 not in supported list: 7.5, 8.0, 8.6, 8.7, 9.0, 10.0, 12.0`, causes `torch.AcceleratorError: no kernel image available`). Reverted via `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`, confirmed working again with a real GPU matmul test. **Always use `--no-deps` when installing anything into this environment going forward**, pip's resolver has already broken the environment twice this way.

**mamba-ssm: currently skipped, not installed.** Build needs CUDA toolkit ≥11.6 (cluster's active `nvcc` was 11.5; newer toolkits exist at `/usr/local/cuda-12.4` etc., not on PATH by default, no `module` command on this cluster) and triton ≥3.5 (torch 2.5.1 pulls in triton 3.1.0, which conflicts). Got mamba-ssm itself to build once, but its own `__init__.py` unconditionally imports Mamba-3 support first, which needs a triton function (`set_allocator`) not present in 3.1.0. Decision: skip for now, ResNet-1D/BiLSTM/Transformer don't need it, `models.py` already auto-detects and falls back to a tested pure-PyTorch equivalent layer (`SimpleSelectiveSSMLayer`) if `mamba_ssm` isn't importable. Revisit only if time allows after the other three architectures are done.

**GitHub backup**: `github.com/tayy00/pECG-Classification`, mirrors `~/ecg_work`'s code (not the data or checkpoints, those need a `.gitignore`, see repo).

## Confirmed real data schema (AttributesDictionary.csv, 14,190 rows, 14 columns)

Exact confirmed column names: `Filename`, `ECG_ID`, `Patient_ID`, `Age`, `Gender`, `Acquisition_date`, `Sampling_point`, `Lead`, `AHA_code`, `CHN_code`, `ICD-10 code`, `pSQI`, `basSQI`, `bSQI`.

- `Filename`: already the working relative path, e.g. `P00/P00001/P00001_E01` (bucket = patient ID's first 3 characters, confirmed across 5+ independently-checked buckets, no exceptions). Use this directly, don't reconstruct paths from patient ID.
- `Age`: string, `"572d"` format, **days**, not years. Every one of 14,190 rows matches this pattern, no exceptions. Convert: `int(s.rstrip("d")) / 365.25`.
- `ICD-10 code`: semicolon-separated when a record has multiple (up to 11 observed), quote-wrapped per code (`'I34.0';'Q21.0'`), prefix-stripping needed for matching against DiseaseCode.csv (see PROJECT_CONTEXT.md for the full matching logic).
- `Lead`: 9 or 12. Full dataset: 12,334 twelve-lead / 1,856 nine-lead. Within Chen's 3,716-record subset specifically: 2,783 twelve-lead / 933 nine-lead (9-lead is a larger share of the disease-labeled subset than of the full dataset, ~25% vs ~13%, a real pattern, not noise).
- `pSQI`/`basSQI`/`bSQI`: pre-computed per-lead signal quality indices, already in the file, semicolon-separated per lead (e.g. `'I':0.288;'II':0.323;...`). Not yet used in the pipeline; available if signal-quality filtering is ever needed, wouldn't need to be computed from scratch.
- `Gender`: values carry literal embedded quotes (`'Female'`, 10 characters), strip with `.str.strip("'")`.

DiseaseCode.csv: 20 rows, columns `Disease Type`, `Disease Category`, `ICD-10 Code`, `ICD-10 Description`. Row 20 (`Other diseases (OD)`) has literal text `"See attribute dictionary file"` in the ICD-10 Code column, not a real code, must be excluded before matching.

## Chen (2025)'s confirmed exact parameters (from the real paper, not the earlier default-guess version)

**Preprocessing:** downsample by factor 5 (500Hz → effective 100Hz, simple decimation, no anti-aliasing filter specified), then window into 3-second segments → **300 samples per window** (not 1500). Per-lead normalization, computed independently within each split (train/val/test each get their own mean/std, not train's stats applied to val/test). No denoising applied (paper states signal quality was already judged sufficient).

**Split:** window-level (see PROJECT_CONTEXT.md), 7:1:2, stratified (multi-label).

**Loss:** weighted BCE, **only the positive term weighted**, not both:
`L = (1-y)·log(1+e^z) + y·w_c·log(1+e^{-z})`, `w_c = min(neg_count/pos_count, 100)` (clipped negative-to-positive ratio, not inverse-frequency-normalized-to-mean-1 as earlier versions of this project used).

**Training:** 200 epochs, batch size 64, AdamW, lr 7e-4 fixed (no weight decay, no schedule), decision threshold 0.5 fixed for all classes at inference.

**Architectures**, all confirmed exact:
- ResNet-1D: channels **64→128→256→512** (double the earlier pre-paper guess of 32→64→128→256), kernel size 7 per block, 4 stages.
- BiLSTM: 128 hidden/direction, 2 layers, dropout 0.5, uses **last timestep only** (not averaged across sequence).
- Transformer: 128-dim, patch size 3 (300 samples → 100 patches), 2 layers, head dim 8 (→16 heads), feedforward 256, dropout 0.1.
- Mamba-2: 128-dim, state dim 64, conv kernel 4, expansion 2, head dim 8, chunk size 30.

**Chen's reported results (both lead configs, all four models):**

| Model | 9-lead Hamming | 12-lead Hamming | 9-lead F1 | 12-lead F1 |
|---|---|---|---|---|
| ResNet-1D | 0.0076 | 0.0069 | 89.15% | 94.67% |
| BiLSTM | 0.0148 | 0.0092 | 79.75% | 90.96% |
| Mamba-2 | 0.0124 | 0.0188 | 85.63% | 84.30% |
| Transformer | 0.0178 | 0.0116 | 81.82% | 90.08% |

## Code architecture (all confirmed working, tested, and now on GitHub)

`data/preprocessing.py` (downsampling, windowing, normalization), `utils/loss.py` (Chen's exact weighted BCE), `models/models.py` (all four architectures at confirmed widths, `build_model(name, n_leads, n_classes)` factory), `data/dataset.py` (label matching, both split modes, `ChenECGDataset`). All four files delivered to and running on the HPC cluster, `~/ecg_work/{data,models,utils}/`.
