"""
preprocessing.py

Chen (2025)'s EXACT preprocessing, confirmed from the paper's Methods section:
  1. Downsample by a factor of 5 (500Hz -> effectively 100Hz)
  2. Segment into 3-second windows -> 300 samples per window (not 1500)
  3. Per-lead normalization, computed independently WITHIN each split
     (train stats from train data only, val stats from val data only, etc.)

One thing the paper does not specify and is flagged here rather than
assumed silently: the exact downsampling method. "Downsampled by a factor
of 5" is implemented here as simple decimation (every 5th sample), the
most literal reading, with no anti-aliasing filter, since the paper does
not mention one. If a later, more precise source specifies otherwise
(e.g. scipy.signal.decimate with its built-in filter), change ONLY
`downsample_signal` below, everything else is independent of this choice.
"""

import numpy as np


def downsample_signal(signal: np.ndarray, factor: int = 5) -> np.ndarray:
    """signal: (n_leads, n_samples). Simple decimation, no filtering."""
    return signal[:, ::factor]


def window_signal(signal: np.ndarray, window_samples: int = 300) -> np.ndarray:
    """signal: (n_leads, n_samples), already downsampled.
    Returns (n_windows, n_leads, window_samples). Matches Chen's "slice-then-split":
    windowing happens BEFORE any split assignment, on the full downsampled recording.
    """
    n_leads, n_samples = signal.shape
    n_windows = n_samples // window_samples
    if n_windows == 0:
        return np.zeros((0, n_leads, window_samples), dtype=signal.dtype)
    usable = n_windows * window_samples
    trimmed = signal[:, :usable]
    return trimmed.reshape(n_leads, n_windows, window_samples).transpose(1, 0, 2)


def normalize_per_lead(windows: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None):
    """windows: (n_windows, n_leads, window_samples).
    Per-lead z-score normalization. If mean/std are not provided, computes
    them from THIS data (use this for the training split). If provided,
    applies them without recomputing (use this for val/test, passing the
    TRAINING split's own mean/std) -- Chen's paper says normalization is
    computed independently WITHIN each split, meaning val and test get
    their own statistics too, not train's statistics applied to them. This
    function supports both usages; which one Section 3 actually needs is
    "independently within each split", so val/test should call this with
    mean=None, std=None too, computing their own stats, not reusing train's.
    """
    if mean is None:
        mean = windows.mean(axis=(0, 2), keepdims=True)  # per-lead, over windows and time
    if std is None:
        std = windows.std(axis=(0, 2), keepdims=True)
    std_safe = np.where(std < 1e-8, 1.0, std)  # avoid divide-by-zero on a flat/dead lead
    normalized = (windows - mean) / std_safe
    return normalized, mean, std
