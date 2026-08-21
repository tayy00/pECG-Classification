# pECG-Classification

Reproduction and extension of Chen (2025), "A Benchmark Study of Deep Learning Methods for Multi-Label Pediatric Electrocardiogram-Based Cardiovascular Disease Classification" (arXiv:2510.03780), on the ZZU pECG dataset (Tan et al., 2025), as part of an MPhil thesis on age-stratified generalization in pediatric ECG classification.

## Status

Chen's exact setup, reproduced against his real, confirmed parameters (downsampling, 300-sample windows, per-lead normalization, his exact weighted-BCE formula, his exact architecture widths, 200 epochs, AdamW, lr 7e-4):

| Model | Test macro F1 | Test Hamming loss | Chen reported (F1 / Hamming) |
|---|---|---|---|
| ResNet-1D | 0.9499 | 0.0069 | 0.9467 / 0.0069 |
| BiLSTM | not yet run | | 0.9096 / 0.0092 |
| Transformer | not yet run | | 0.9008 / 0.0116 |
| Mamba-2 | not yet run (fallback layer, no CUDA kernel available on this cluster) | | 0.8430 / 0.0188 |

## Data

Not included in this repository, download separately:

- ZZU pECG: figshare DOI [10.6084/m9.figshare.27078763](https://doi.org/10.6084/m9.figshare.27078763)
- Expects `AttributesDictionary.csv`, `DiseaseCode.csv`, `ECGCode.csv`, and the extracted signal folder (`Child_ecg_extracted/Child_ecg/`, nested as `<3-char bucket>/<patient_id>/<patient_id>_E<episode>.hea/.dat`)

## Structure

```
data/
  preprocessing.py   # downsampling, windowing, per-lead normalization
  dataset.py         # disease-label matching, window-level and record-level splits, PyTorch Dataset
models/
  models.py          # ResNet-1D, BiLSTM, Transformer, Mamba-2 (with CPU/no-kernel fallback)
utils/
  loss.py            # Chen's exact weighted BCE (positive term only)
notebooks/           # interactive training notebooks
outputs/             # trained checkpoints (not tracked in git, see .gitignore)
```

## Setup

```bash
python -m venv .ecg_venv
source .ecg_venv/bin/activate
pip install -r requirements.txt
```

**Known environment issue on Tesla V100 (compute capability 7.0):** cuDNN has no working convolution kernel for this GPU with the installed PyTorch build. Every fresh session needs, before building any model:

```python
torch.backends.cudnn.enabled = False
```

Without this, training crashes on the first convolution with `RuntimeError: GET was unable to find an engine to execute this computation`. Confirmed real: with cuDNN disabled, ResNet-1D still trains at roughly 0.6 minutes/epoch on this hardware, well within a practical range for the full 200-epoch protocol.

**Mamba-2** currently runs on a pure-PyTorch fallback layer (same recurrence, not the fused CUDA kernel), since `mamba-ssm`'s official package requires CUDA toolkit 11.6+ and triton 3.5+, neither reliably available in this environment as of this writing. Not blocking the other three architectures.

## Reference

Chen, Y. (2025). *A Benchmark Study of Deep Learning Methods for Multi-Label Pediatric Electrocardiogram-Based Cardiovascular Disease Classification.* arXiv:2510.03780.

Tan, J., et al. (2025). *A pediatric ECG database with disease diagnosis covering 11,643 children.* Scientific Data, 12, 867.
