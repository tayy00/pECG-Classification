"""
dataset.py

Ties together: real column names confirmed from AttributesDictionary.csv,
the validated disease-code matching logic (confirmed twice against the
real files to select exactly 3,716 records), the corrected preprocessing
(downsample-by-5, 300-sample windows, per-lead normalization), and Chen's
"slice-then-split" + multi-label stratified 7:1:2 split.

Two split modes, genuinely different, both actually implemented below:

  window_level_split()  -> Section 3 ONLY, the exact reproduction. Windows
  are the unit that gets split, not records. A given recording's windows
  CAN land in different splits. This matches Chen's own paper exactly,
  including the leakage he names himself as a limitation.

  record_level_split()  -> Section 4 onward. Records are the unit that
  gets split; every window from a given record travels together. This is
  the correction, not part of "reproducing Chen exactly."
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import wfdb
except ImportError:
    wfdb = None

from data.preprocessing import downsample_signal, window_signal, normalize_per_lead

COL_PATIENT_ID = "Patient_ID"
COL_RECORD_ID = "ECG_ID"
COL_FILENAME = "Filename"
COL_AGE_RAW = "Age"
COL_N_LEADS = "Lead"
COL_ICD10_CODE = "ICD-10 code"


def _strip_code_prefix(code: str) -> str:
    return re.sub(r"^\(\w+\)\s*", "", str(code)).strip()


def load_disease_code_table(disease_code_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(disease_code_csv_path)
    real = df[df["ICD-10 Code"] != "See attribute dictionary file"].copy()
    real["bare_code"] = real["ICD-10 Code"].apply(_strip_code_prefix)
    return real


def _record_icd10_codes(raw: str) -> set:
    codes = set()
    for c in str(raw).split(";"):
        c = c.strip().strip("'").strip()
        if c:
            codes.add(_strip_code_prefix(c))
    return codes


def build_chen_labels(metadata: pd.DataFrame, disease_code_table: pd.DataFrame):
    target_codes = sorted(set(disease_code_table["bare_code"]))
    record_code_sets = metadata[COL_ICD10_CODE].apply(_record_icd10_codes)
    out = metadata.copy()
    label_columns = [f"label_{code}" for code in target_codes]
    for code, col in zip(target_codes, label_columns):
        out[col] = record_code_sets.apply(lambda s, c=code: 1.0 if c in s else 0.0)
    out["in_chen_subset"] = out[label_columns].sum(axis=1) > 0
    return out, label_columns


def _materialize_all_windows(subset: pd.DataFrame, label_columns: list, data_dir: str,
                              downsample_factor: int, window_samples: int) -> pd.DataFrame:
    rows = []
    for record_idx in range(len(subset)):
        row = subset.iloc[record_idx]
        path = os.path.join(data_dir, str(row[COL_FILENAME]))
        header = wfdb.rdheader(path)
        down_len = header.sig_len // downsample_factor
        n_windows = down_len // window_samples
        for w in range(n_windows):
            rows.append({"record_idx": record_idx, "window_idx": w,
                         COL_RECORD_ID: row[COL_RECORD_ID], **{c: row[c] for c in label_columns}})
    return pd.DataFrame(rows)


def window_level_split(subset: pd.DataFrame, label_columns: list, data_dir: str,
                        downsample_factor: int = 5, window_samples: int = 300, seed: int = 42):
    """Section 3 ONLY. A record's windows CAN and often will end up split
    across more than one of train/val/test, deliberately, matching Chen's
    own described method, leakage included.
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    window_table = _materialize_all_windows(subset, label_columns, data_dir, downsample_factor, window_samples)
    X = window_table.index.values.reshape(-1, 1)
    Y = window_table[label_columns].values

    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(msss1.split(X, Y))
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=2 / 3, random_state=seed)
    val_idx_rel, test_idx_rel = next(msss2.split(temp_idx.reshape(-1, 1), Y[temp_idx]))
    val_idx, test_idx = temp_idx[val_idx_rel], temp_idx[test_idx_rel]

    def to_pairs(idx):
        sub = window_table.iloc[idx]
        return list(zip(sub["record_idx"].tolist(), sub["window_idx"].tolist()))

    return {"train": to_pairs(train_idx), "val": to_pairs(val_idx), "test": to_pairs(test_idx)}


def record_level_split(subset: pd.DataFrame, label_columns: list, seed: int = 42):
    """Section 4 onward. Every window belonging to a given record stays in
    that same split, always."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    X = subset.index.values.reshape(-1, 1)
    Y = subset[label_columns].values

    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(msss1.split(X, Y))
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=2 / 3, random_state=seed)
    val_idx_rel, test_idx_rel = next(msss2.split(temp_idx.reshape(-1, 1), Y[temp_idx]))
    val_idx, test_idx = temp_idx[val_idx_rel], temp_idx[test_idx_rel]

    return {"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()}


def expand_record_split_to_pairs(subset: pd.DataFrame, record_indices: list, data_dir: str,
                                  downsample_factor: int, window_samples: int) -> list:
    """For record_level_split's output: expands record indices into the
    full list of (record_idx, window_idx) pairs, every window of every
    record in this split, all staying together."""
    pairs = []
    for record_idx in record_indices:
        row = subset.iloc[record_idx]
        path = os.path.join(data_dir, str(row[COL_FILENAME]))
        header = wfdb.rdheader(path)
        down_len = header.sig_len // downsample_factor
        n_windows = down_len // window_samples
        for w in range(n_windows):
            pairs.append((record_idx, w))
    return pairs


class ChenECGDataset(Dataset):
    """Takes a pre-built list of (record_idx, window_idx) pairs directly,
    agnostic to whether they came from window_level_split or
    expand_record_split_to_pairs. Applies downsampling, windowing, and
    per-lead normalization, with this split's own statistics computed once
    and reused, not recomputed per __getitem__ call.
    """

    def __init__(self, subset: pd.DataFrame, label_columns: list, data_dir: str,
                 index_pairs: list, downsample_factor: int = 5, window_samples: int = 300):
        self.subset = subset.reset_index(drop=True)
        self.label_columns = label_columns
        self.data_dir = data_dir
        self.downsample_factor = downsample_factor
        self.window_samples = window_samples
        self._index = index_pairs
        self._window_cache = {}
        self._norm_mean = None
        self._norm_std = None

    def _load_and_process(self, record_idx):
        if record_idx not in self._window_cache:
            row = self.subset.iloc[record_idx]
            path = os.path.join(self.data_dir, str(row[COL_FILENAME]))
            record = wfdb.rdrecord(path)
            signal = record.p_signal.T.astype(np.float32)
            down = downsample_signal(signal, self.downsample_factor)
            windows = window_signal(down, self.window_samples)
            self._window_cache[record_idx] = windows
        return self._window_cache[record_idx]

    def fit_normalization(self):
        all_windows = []
        for record_idx, window_idx in self._index:
            windows = self._load_and_process(record_idx)
            all_windows.append(windows[window_idx])
        if all_windows:
            stacked = np.stack(all_windows, axis=0)
            _, mean, std = normalize_per_lead(stacked)
            self._norm_mean, self._norm_std = mean, std

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        record_idx, window_idx = self._index[i]
        windows = self._load_and_process(record_idx)
        window = windows[window_idx]
        normed, _, _ = normalize_per_lead(window[None, :, :], self._norm_mean, self._norm_std)
        x = torch.from_numpy(normed[0]).float()
        row = self.subset.iloc[record_idx]
        y = torch.from_numpy(row[self.label_columns].to_numpy(dtype=np.float32))
        return x, y
