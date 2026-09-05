import numpy as np


def split_trials_by_label(labels, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    trial_indices = np.arange(len(labels))

    if len(labels) < 3:
        raise ValueError("At least three trials are required for train/valid/test splitting.")

    if len(labels) < 10:
        test_indices = _choose_two_label_balanced_trials(labels, rng)
        remaining = np.setdiff1d(trial_indices, test_indices, assume_unique=False)
        rng.shuffle(remaining)
        n_valid = max(1, int(round(len(remaining) * val_ratio / (train_ratio + val_ratio))))
        valid_indices = remaining[:n_valid]
        train_indices = remaining[n_valid:]
    else:
        n_test = max(1, int(round(len(labels) * test_ratio)))
        n_valid = max(1, int(round(len(labels) * val_ratio)))
        test_indices, leftover = _stratified_take(labels, trial_indices, n_test, rng)
        valid_indices, train_indices = _stratified_take(labels, leftover, n_valid, rng)

    _assert_no_trial_overlap(train_indices, valid_indices, test_indices)
    return np.sort(train_indices), np.sort(valid_indices), np.sort(test_indices)


def build_trial_cv_splits(labels, seed=42, n_folds=10):
    labels = np.asarray(labels).reshape(-1)
    trial_indices = np.arange(len(labels))

    if len(labels) >= n_folds:
        folds = _make_stratified_folds(labels, n_folds, seed)
        splits = []
        for fold_idx in range(n_folds):
            test_indices = folds[fold_idx]
            valid_indices = folds[(fold_idx + 1) % n_folds]
            train_indices = np.concatenate(
                [folds[i] for i in range(n_folds) if i not in (fold_idx, (fold_idx + 1) % n_folds)]
            )
            splits.append(
                {
                    "fold": fold_idx,
                    "train": np.sort(train_indices),
                    "valid": np.sort(valid_indices),
                    "test": np.sort(test_indices),
                    "protocol": "10-fold-trial-cv",
                }
            )
    else:
        splits = []
        fold_idx = 0
        for test_indices in _label_balanced_test_pairs(labels):
            remaining = np.setdiff1d(trial_indices, test_indices, assume_unique=False)
            valid_indices = _choose_validation_from_remaining(labels, remaining, seed + fold_idx)
            train_indices = np.setdiff1d(remaining, valid_indices, assume_unique=False)
            splits.append(
                {
                    "fold": fold_idx,
                    "train": np.sort(train_indices),
                    "valid": np.sort(valid_indices),
                    "test": np.sort(test_indices),
                    "protocol": "label-balanced-leave-two-trials-out",
                }
            )
            fold_idx += 1

    for split in splits:
        _assert_no_trial_overlap(split["train"], split["valid"], split["test"])
    return splits


def make_windows_from_trials(eeg_trials, labels, trial_indices, window_length, overlap, eeg_channel):
    labels = np.asarray(labels).reshape(-1)
    stride = int(window_length * (1 - overlap))
    if stride <= 0:
        raise ValueError("Window stride must be positive. Check window_length and overlap.")

    all_windows = []
    all_labels = []
    for trial_idx in trial_indices:
        eeg = np.asarray(eeg_trials[trial_idx])
        label = labels[trial_idx]
        for start in range(0, eeg.shape[0] - window_length + 1, stride):
            all_windows.append(eeg[start:start + window_length, :])
            all_labels.append(label)

    if not all_windows:
        raise ValueError("No windows were generated. Check trial length and window_length.")

    data = np.asarray(all_windows).reshape(-1, window_length, eeg_channel)
    target = np.asarray(all_labels).reshape(-1, 1)
    return data, target


def print_split_summary(labels, train_indices, valid_indices, test_indices):
    labels = np.asarray(labels).reshape(-1)
    parts = {
        "train": train_indices,
        "valid": valid_indices,
        "test": test_indices,
    }
    summary = []
    for name, indices in parts.items():
        unique, counts = np.unique(labels[indices], return_counts=True)
        counts_by_label = {int(k): int(v) for k, v in zip(unique, counts)}
        summary.append(f"{name}: trials={indices.tolist()}, labels={counts_by_label}")
    print("Trial-independent split | " + " | ".join(summary))


def print_cv_split_summary(labels, split):
    print(
        f"Trial CV split | protocol={split['protocol']} | fold={split['fold']} | ",
        end="",
    )
    print_split_summary(labels, split["train"], split["valid"], split["test"])


def _choose_two_label_balanced_trials(labels, rng):
    unique_labels = np.unique(labels)
    if len(unique_labels) >= 2:
        selected = []
        for label in unique_labels[:2]:
            candidates = np.flatnonzero(labels == label)
            selected.append(rng.choice(candidates))
        return np.asarray(selected, dtype=int)

    return rng.choice(np.arange(len(labels)), size=1, replace=False)


def _make_stratified_folds(labels, n_folds, seed):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        rng.shuffle(label_indices)
        for offset, trial_idx in enumerate(label_indices):
            folds[offset % n_folds].append(trial_idx)
    return [np.asarray(sorted(fold), dtype=int) for fold in folds]


def _label_balanced_test_pairs(labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        indices = np.arange(len(labels))
        return [np.asarray(pair, dtype=int) for pair in zip(indices[:-1], indices[1:])]

    first_label = unique_labels[0]
    second_label = unique_labels[1]
    first_indices = np.flatnonzero(labels == first_label)
    second_indices = np.flatnonzero(labels == second_label)
    pairs = []
    for first_idx in first_indices:
        for second_idx in second_indices:
            pairs.append(np.asarray([first_idx, second_idx], dtype=int))
    return pairs


def _choose_validation_from_remaining(labels, remaining, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels[remaining]):
        candidates = remaining[labels[remaining] == label]
        if len(candidates) > 0:
            selected.append(rng.choice(candidates))

    if selected:
        return np.asarray(selected, dtype=int)
    return rng.choice(remaining, size=1, replace=False)


def _stratified_take(labels, candidate_indices, n_take, rng):
    candidate_indices = np.asarray(candidate_indices)
    n_take = min(n_take, len(candidate_indices) - 1)
    selected = []

    label_groups = []
    for label in np.unique(labels[candidate_indices]):
        group = candidate_indices[labels[candidate_indices] == label].copy()
        rng.shuffle(group)
        label_groups.append(group.tolist())

    while len(selected) < n_take and any(label_groups):
        for group in label_groups:
            if len(selected) >= n_take:
                break
            if group:
                selected.append(group.pop())

    selected = np.asarray(selected, dtype=int)
    remaining = np.setdiff1d(candidate_indices, selected, assume_unique=False)
    return selected, remaining


def _assert_no_trial_overlap(train_indices, valid_indices, test_indices):
    train = set(np.asarray(train_indices).tolist())
    valid = set(np.asarray(valid_indices).tolist())
    test = set(np.asarray(test_indices).tolist())
    if train & valid or train & test or valid & test:
        raise AssertionError("Trial-level split contains overlapping trials.")
