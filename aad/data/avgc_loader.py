"""AV-GC loader for trial-independent BeaST experiments.

Raw AV-GC recordings are intentionally kept outside the repository. Point
``AAD_AVGC_PATH`` or ``data_document_path`` at the dataset parent directory.

Expected layout::

    <root>/<condition>/<variant>/S1/
        csv/<label-file>.csv
        S1Tra1.csv
        S1Tra2.csv
        ...

For the two-fold protocol, fold 0 holds out trial 1 and fold 1 holds out trial
2. Validation trials are selected from the remaining trials before windowing.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


AVGC_CONDITIONS = {
    "NV": "AVGCDataset_no_visuals",
    "SV": "AVGCDataset_fixed_video",
    "MV": "AVGCDataset_moving_video",
    "MTN": "AVGCDataset_moving_target_noise",
}
AVGC_DEFAULT_VARIANT = os.environ.get("AAD_AVGC_VARIANT", "No_vanilla_128")
AVGC_DEFAULT_ROOT = os.environ.get("AAD_AVGC_PATH", "./data/AVGC")

LEFT_ELECTRODES = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1",
    "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9",
    "PO7", "PO3", "O1",
]
RIGHT_ELECTRODES = [
    "Fp2", "AF8", "AF4", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4", "FC2", "C2",
    "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8", "P10",
    "PO8", "PO4", "O2",
]
BIOSEMI64 = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1",
    "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9",
    "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2", "AF8", "AF4",
    "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz",
    "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8",
    "P10", "PO8", "PO4", "O2",
]


class DualInputDataset(Dataset):
    """Dataset shape expected by ``aad.training.proposed``."""

    def __init__(self, data_left, data_right, labels):
        self.data_left = np.asarray(data_left, dtype=np.float32)
        self.data_right = np.asarray(data_right, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.uint8).reshape(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data_left[index], self.data_right[index], self.labels[index]


def get_AVGC_data(
    name="S1",
    timelen=0.25,
    data_document_path=None,
    fold_id=0,
    condition="NV",
    variant=None,
    label_mode="half",
    batch_size=32,
):
    """Return train/validation/test loaders for one AV-GC fold."""

    if fold_id not in (0, 1):
        raise ValueError("AV-GC fold_id must be 0 or 1")
    trials, labels = _load_subject(name, data_document_path, condition, variant)
    if len(trials) < 3:
        raise ValueError("AV-GC requires at least three trials for train/valid/test splitting")

    test_trial = int(fold_id)
    remaining = [index for index in range(len(trials)) if index != test_trial]
    valid_trials = _choose_validation_trials(labels, remaining, seed=1234 + int(fold_id))
    valid_set = set(valid_trials.tolist())
    train_trials = [index for index in remaining if index not in valid_set]
    _assert_disjoint(train_trials, valid_trials, [test_trial])

    window_length = math.ceil(128 * float(timelen))
    train_data, train_labels = _make_windows(trials, labels, train_trials, window_length, label_mode)
    valid_data, valid_labels = _make_windows(trials, labels, valid_trials, window_length, label_mode)
    test_data, test_labels = _make_windows(trials, labels, [test_trial], window_length, label_mode)

    left_indices = [BIOSEMI64.index(channel) for channel in LEFT_ELECTRODES]
    right_indices = [BIOSEMI64.index(channel) for channel in RIGHT_ELECTRODES]

    def make_dataset(data, targets):
        data = np.expand_dims(data, axis=1)  # N, C=1, T, V
        return DualInputDataset(data[:, :, :, left_indices], data[:, :, :, right_indices], targets)

    train_set = make_dataset(train_data, train_labels)
    valid_set = make_dataset(valid_data, valid_labels)
    test_set = make_dataset(test_data, test_labels)
    loader_args = dict(batch_size=batch_size, pin_memory=True, drop_last=True)
    return (
        DataLoader(train_set, shuffle=True, **loader_args),
        DataLoader(valid_set, shuffle=False, **loader_args),
        DataLoader(test_set, shuffle=False, **loader_args),
    )


def get_available_subjects(data_document_path=None, condition="NV", variant=None):
    base = _condition_dir(data_document_path, condition, variant)
    return sorted(
        [path.name for path in base.iterdir() if path.is_dir() and path.name.startswith("S")],
        key=lambda value: int(value[1:]) if value[1:].isdigit() else 0,
    )


def _condition_dir(root, condition, variant):
    root = Path(root or AVGC_DEFAULT_ROOT)
    condition = AVGC_CONDITIONS.get(str(condition).upper(), condition)
    path = root / condition / (variant or AVGC_DEFAULT_VARIANT)
    if not path.exists():
        raise FileNotFoundError(
            f"AV-GC directory does not exist: {path}. "
            "Pass --avgc-root or set AAD_AVGC_PATH."
        )
    return path


def _load_subject(subject, root, condition, variant):
    subject_dir = _condition_dir(root, condition, variant) / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"AV-GC subject directory does not exist: {subject_dir}")
    label_files = sorted((subject_dir / "csv").glob("*.csv"))
    if not label_files:
        raise FileNotFoundError(f"No AV-GC label CSV found under {subject_dir / 'csv'}")

    labels = pd.read_csv(label_files[0]).iloc[:, 0].to_numpy().reshape(-1).astype(np.int64)
    if labels.min(initial=0) >= 1:
        labels -= 1
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"AV-GC labels must be binary after normalization, got {np.unique(labels)}")

    trials = []
    for trial_number in range(1, len(labels) + 1):
        trial_path = subject_dir / f"{subject}Tra{trial_number}.csv"
        if not trial_path.exists():
            raise FileNotFoundError(f"Missing AV-GC trial file: {trial_path}")
        trial = pd.read_csv(trial_path, header=None).iloc[:, :64].to_numpy(dtype=np.float32)
        if trial.shape[1] != 64:
            raise ValueError(f"Expected 64 EEG channels in {trial_path}, got {trial.shape[1]}")
        trials.append(trial)
    return trials, labels.astype(np.uint8)


def _make_windows(trials, labels, trial_indices, window_length, label_mode):
    if label_mode not in ("half", "trial"):
        raise ValueError("label_mode must be 'half' or 'trial'")
    stride = max(1, int(window_length * 0.5))
    windows, targets = [], []
    for trial_index in trial_indices:
        trial = np.asarray(trials[trial_index])
        label = int(labels[trial_index])
        ranges = [(0, trial.shape[0], label)]
        if label_mode == "half":
            half = trial.shape[0] // 2
            if half < window_length:
                raise ValueError(
                    f"Trial {trial_index + 1} half ({half}) is shorter than window ({window_length})"
                )
            ranges = [(0, half, label), (half, trial.shape[0], 1 - label)]
        for start, end, current_label in ranges:
            for position in range(start, end - window_length + 1, stride):
                windows.append(trial[position:position + window_length])
                targets.append(current_label)
    if not windows:
        raise ValueError("No AV-GC windows were generated; check time-len and trial duration")
    return np.asarray(windows, dtype=np.float32), np.asarray(targets, dtype=np.uint8).reshape(-1, 1)


def _choose_validation_trials(labels, remaining, seed):
    labels = np.asarray(labels).reshape(-1)
    remaining = np.asarray(remaining, dtype=int)
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels[remaining]):
        candidates = remaining[labels[remaining] == label]
        selected.append(int(rng.choice(candidates)))
    if not selected:
        selected = [int(rng.choice(remaining))]
    return np.asarray(sorted(set(selected)), dtype=int)


def _assert_disjoint(train_trials, valid_trials, test_trials):
    groups = [set(np.asarray(group).tolist()) for group in (train_trials, valid_trials, test_trials)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise AssertionError("AV-GC trial split contains overlapping trials")
