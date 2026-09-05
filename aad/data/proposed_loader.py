import math
import os

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import mne
import numpy as np
import pandas as pd
import torch
from dotmap import DotMap
from mne.decoding import CSP
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

from aad.data.trial_split import (
    build_trial_cv_splits,
    make_windows_from_trials,
    print_cv_split_summary,
    print_split_summary,
    split_trials_by_label,
)



def get_DTU_data(name="S1", timelen=2, data_document_path=None, fold_id=None):
    data_document_path = data_document_path or os.environ.get("AAD_DTU_PATH", "./data/DTU/DATA_preproc")

    class CustomDatasets(torch.utils.data.Dataset):
        def __init__(self, data_left, data_right, labels):
            self.data_left = data_left
            self.data_right = data_right
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            left = self.data_left[idx]
            right = self.data_right[idx]
            label = self.labels[idx]
            return left, right, label  # 分别返回左脑数据、右脑数据和标签

    def get_data_from_mat(mat_path):
        '''
        discription:load data from mat path and reshape
        param{type}:mat_path: Str
        return{type}: onesub_data
        '''
        mat_eeg_data = []
        mat_wavA_data = []
        mat_wavB_data = []
        mat_event_data = []
        matstruct_contents = loadmat(mat_path)
        matstruct_contents = matstruct_contents['data']
        mat_event = matstruct_contents[0, 0]['event']['eeg'].item()
        mat_event_value = mat_event[0]['value']  # 1*60 1=male, 2=female
        mat_eeg = matstruct_contents[0, 0]['eeg']  # 60 trials 3200*66
        mat_wavA = matstruct_contents[0, 0]['wavA']
        mat_wavB = matstruct_contents[0, 0]['wavB']
        for i in range(mat_eeg.shape[1]):
            mat_eeg_data.append(mat_eeg[0, i])
            mat_wavA_data.append(mat_wavA[0, i])
            mat_wavB_data.append(mat_wavB[0, i])
            mat_event_data.append(mat_event_value[i][0][0])


        return mat_eeg_data, mat_event_data

    print("Num GPUs Available: ", torch.cuda.is_available())
    print(name)
    time_len = timelen
    random_seed = 42
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs*time_len)
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 200
    args.random_seed = random_seed
    args.people_number = 18
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 60
    args.cell_number = 3200
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.log_interval = 20
    args.csp_comp = 64
    args.label_col = 0
    args.log_path = "ConvTran-main-DTU/Results/1s"
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    subpath = args.data_document_path + '/' + str(args.name) + '_data_preproc.mat'
    eeg_data, event_data = get_data_from_mat(subpath)
    eeg_data = np.array(eeg_data)
    eeg_data = eeg_data[:, :, 0:64]
    event_data = np.array(event_data)
    print(eeg_data.shape)
    eeg_data = np.vstack(eeg_data)
    eeg_data = eeg_data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)
    eeg_data = np.array(eeg_data)
    print(eeg_data.shape)

    event_data = np.squeeze(event_data - 1)

    if fold_id is None:
        train_trials, valid_trials, test_trials = split_trials_by_label(event_data, seed=args.random_seed)
        print_split_summary(event_data, train_trials, valid_trials, test_trials)
    else:
        cv_splits = build_trial_cv_splits(event_data, seed=args.random_seed)
        split = cv_splits[fold_id]
        train_trials, valid_trials, test_trials = split["train"], split["valid"], split["test"]
        print_cv_split_summary(event_data, split)

    train_data, train_label = make_windows_from_trials(
        eeg_data, event_data, train_trials, args.window_length, args.overlap, args.csp_comp
    )
    valid_data, valid_label = make_windows_from_trials(
        eeg_data, event_data, valid_trials, args.window_length, args.overlap, args.csp_comp
    )
    test_data, test_label = make_windows_from_trials(
        eeg_data, event_data, test_trials, args.window_length, args.overlap, args.csp_comp
    )
    print(train_data.shape)
    print(valid_data.shape)
    print(test_data.shape)#(2584, 256, 64)(288, 256, 64)
    train_data = np.expand_dims(train_data, axis=1)
    valid_data = np.expand_dims(valid_data, axis=1)
    test_data = np.expand_dims(test_data, axis=1)
    del eeg_data
    del event_data

    montage = mne.channels.make_standard_montage('biosemi64')
    electrode_names = montage.ch_names
    print(electrode_names)


    left_electrodes = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1',
                        'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                        'PO7', 'PO3', 'O1']
    right_electrodes = ['Fp2', 'AF8', 'AF4', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'C2',
                        'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10',
                        'PO8', 'PO4', 'O2']



    left_indices = [electrode_names.index(elec) for elec in left_electrodes if elec in electrode_names]
    right_indices = [electrode_names.index(elec) for elec in right_electrodes if elec in electrode_names]

    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data, train_label = train_data[indices], train_label[indices]

    train_data_left = train_data[:, :, :, left_indices]
    train_data_right = train_data[:, :, :, right_indices]
    valid_data_left = valid_data[:, :, :, left_indices]
    valid_data_right = valid_data[:, :, :, right_indices]
    test_data_left = test_data[:, :, :, left_indices]
    test_data_right = test_data[:, :, :, right_indices]
    train_loader = DataLoader(dataset=CustomDatasets(train_data_left, train_data_right, train_label),
                              batch_size=args.batch_size, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(dataset=CustomDatasets(valid_data_left, valid_data_right, valid_label),
                              batch_size=args.batch_size, drop_last=True, pin_memory=True)
    test_loader = DataLoader(dataset=CustomDatasets(test_data_left, test_data_right, test_label),
                             batch_size=args.batch_size, drop_last=True, pin_memory=True)
    return train_loader, valid_loader, test_loader


def get_KUL_data(name="S1", time_len=1, data_document_path=None, fold_id=None):
    data_document_path = data_document_path or os.environ.get("AAD_KUL_PATH", "./data/KUL/KUL_single_single3")

    class CustomDatasets(torch.utils.data.Dataset):
        def __init__(self, data_left, data_right, labels):
            self.data_left = data_left
            self.data_right = data_right
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            left = self.data_left[idx]
            right = self.data_right[idx]
            label = self.labels[idx]
            return left, right, label

    def read_prepared_data(args):
        data = []
        target = []
        for l in range(len(args.ConType)):
            label = pd.read_csv(args.data_document_path + "/csv/" + args.name + args.ConType[l] + ".csv")

            for k in range(args.trail_number):
                filename = args.data_document_path + "/" + args.ConType[l] + "/" + args.name + "Tra" + str(
                    k + 1) + ".csv"
                data_pf = pd.read_csv(filename, header=None)
                eeg_data = data_pf.iloc[:, 2:]  # KUL,DTU


                data.append(eeg_data)
                target.append(label.iloc[k, args.label_col])

        return data, target

    print("Num GPUs Available: ", torch.cuda.is_available())
    print(name)
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 200
    args.random_seed = 1234
    args.image_size = 32
    args.people_number = 16
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 8
    args.cell_number = 46080
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.log_interval = 20
    args.label_col = 0
    args.alpha_low = 8
    args.alpha_high = 13
    args.log_path = "result"
    args.frequency_resolution = args.fs / args.window_length
    args.point_low = math.ceil(args.alpha_low / args.frequency_resolution)
    args.point_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    args.csp_comp = 64

    eeg_data, event_data = read_prepared_data(args)
    data = np.vstack(eeg_data)
    eeg_data = data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)

    event_data = np.squeeze(event_data - 1)
    print("eeg_data.shape", eeg_data.shape)

    print(event_data.shape)


    if fold_id is None:
        train_trials, valid_trials, test_trials = split_trials_by_label(event_data, seed=args.random_seed)
        print_split_summary(event_data, train_trials, valid_trials, test_trials)
    else:
        cv_splits = build_trial_cv_splits(event_data, seed=args.random_seed)
        split = cv_splits[fold_id]
        train_trials, valid_trials, test_trials = split["train"], split["valid"], split["test"]
        print_cv_split_summary(event_data, split)

    train_data, train_label = make_windows_from_trials(
        eeg_data, event_data, train_trials, args.window_length, args.overlap, args.csp_comp
    )
    valid_data, valid_label = make_windows_from_trials(
        eeg_data, event_data, valid_trials, args.window_length, args.overlap, args.csp_comp
    )
    test_data, test_label = make_windows_from_trials(
        eeg_data, event_data, test_trials, args.window_length, args.overlap, args.csp_comp
    )
    print(train_data.shape)
    print(test_data.shape)#(2584, 256, 64)(288, 256, 64)
    train_data = np.expand_dims(train_data, axis=1)
    valid_data = np.expand_dims(valid_data, axis=1)
    test_data = np.expand_dims(test_data, axis=1)

    montage = mne.channels.make_standard_montage('biosemi64')
    electrode_names = montage.ch_names
    print(electrode_names)

    left_electrodes = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1',
                        'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                        'PO7', 'PO3', 'O1']
    right_electrodes = ['Fp2', 'AF8', 'AF4', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'C2',
                        'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10',
                        'PO8', 'PO4', 'O2']


    left_indices = [electrode_names.index(elec) for elec in left_electrodes if elec in electrode_names]
    right_indices = [electrode_names.index(elec) for elec in right_electrodes if elec in electrode_names]

    print(1, data.shape)
    print("len of test_label", len(test_label), len(train_label))
    del data


    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data, train_label = train_data[indices], train_label[indices]

    train_data_left = train_data[:, :, :, left_indices]
    train_data_right = train_data[:, :, :, right_indices]
    valid_data_left = valid_data[:, :, :, left_indices]
    valid_data_right = valid_data[:, :, :, right_indices]
    test_data_left = test_data[:, :, :, left_indices]
    test_data_right = test_data[:, :, :, right_indices]



    train_loader = DataLoader(dataset=CustomDatasets(train_data_left, train_data_right, train_label),
                              batch_size=args.batch_size, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(dataset=CustomDatasets(valid_data_left, valid_data_right, valid_label),
                              batch_size=args.batch_size, drop_last=True, pin_memory=True)
    test_loader = DataLoader(dataset=CustomDatasets(test_data_left, test_data_right, test_label),
                             batch_size=args.batch_size, drop_last=True, pin_memory=True)
    return train_loader, valid_loader, test_loader
