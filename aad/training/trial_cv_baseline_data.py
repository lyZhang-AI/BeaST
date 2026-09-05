import math
import os

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from aad.data.trial_split import build_trial_cv_splits, make_windows_from_trials, print_cv_split_summary


DTU_PATH = os.environ.get("AAD_DTU_PATH", "./data/DTU/DATA_preproc")
KUL_PATH = os.environ.get("AAD_KUL_PATH", "./data/KUL/KUL_single_single3")


class WindowDataset(Dataset):
    def __init__(self, data, labels, input_mode):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels.reshape(-1), dtype=torch.long)
        self.input_mode = input_mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data[index]
        if self.input_mode == "channels_time":
            x = x.transpose(0, 1)
        elif self.input_mode == "image_channels_time":
            x = x.transpose(0, 1).unsqueeze(0)
        elif self.input_mode == "image_time_channels":
            x = x.unsqueeze(0)
        elif self.input_mode == "graph":
            x = x.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported input mode: {self.input_mode}")
        return x, self.labels[index]


def get_trial_cv_loaders(
    dataset,
    subject,
    time_len,
    fold_id,
    input_mode,
    batch_size=32,
    seed=1234,
    num_workers=0,
):
    eeg_trials, labels = load_trials(dataset, subject)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    splits = build_trial_cv_splits(labels, seed=seed)
    if fold_id < 0 or fold_id >= len(splits):
        raise ValueError(f"{dataset} {subject} has {len(splits)} folds, but fold_id={fold_id}.")

    split = splits[fold_id]
    print_cv_split_summary(labels, split)
    window_length = int(math.ceil(128 * float(time_len)))
    overlap = 0.5
    eeg_channel = eeg_trials.shape[-1]

    train_data, train_label = make_windows_from_trials(
        eeg_trials, labels, split["train"], window_length, overlap, eeg_channel
    )
    valid_data, valid_label = make_windows_from_trials(
        eeg_trials, labels, split["valid"], window_length, overlap, eeg_channel
    )
    test_data, test_label = make_windows_from_trials(
        eeg_trials, labels, split["test"], window_length, overlap, eeg_channel
    )

    rng = np.random.default_rng(seed + fold_id)
    order = rng.permutation(len(train_label))
    train_data, train_label = train_data[order], train_label[order]

    train_loader = DataLoader(
        WindowDataset(train_data, train_label, input_mode),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        WindowDataset(valid_data, valid_label, input_mode),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        WindowDataset(test_data, test_label, input_mode),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )
    return train_loader, valid_loader, test_loader, split["protocol"], len(splits)


def load_trials(dataset, subject):
    dataset = dataset.upper()
    if dataset == "DTU":
        return _load_dtu(subject)
    if dataset == "KUL":
        return _load_kul(subject)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _load_dtu(subject):
    mat_path = os.path.join(DTU_PATH, f"{subject}_data_preproc.mat")
    matstruct_contents = loadmat(mat_path)["data"]
    mat_event = matstruct_contents[0, 0]["event"]["eeg"].item()
    mat_event_value = mat_event[0]["value"]
    mat_eeg = matstruct_contents[0, 0]["eeg"]

    eeg_trials = []
    labels = []
    for i in range(mat_eeg.shape[1]):
        eeg_trials.append(mat_eeg[0, i][:, :64])
        labels.append(mat_event_value[i][0][0] - 1)
    return np.asarray(eeg_trials, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _load_kul(subject):
    label_path = os.path.join(KUL_PATH, "csv", f"{subject}No.csv")
    labels_df = pd.read_csv(label_path)
    eeg_trials = []
    labels = []
    for trial_idx in range(8):
        trial_path = os.path.join(KUL_PATH, "No", f"{subject}Tra{trial_idx + 1}.csv")
        trial_df = pd.read_csv(trial_path, header=None)
        eeg_trials.append(trial_df.iloc[:, 2:66].to_numpy(dtype=np.float32))
        labels.append(labels_df.iloc[trial_idx, 0])
    labels = np.asarray(labels, dtype=np.int64)
    if labels.min() == 1:
        labels = labels - 1
    return np.asarray(eeg_trials, dtype=np.float32), labels
