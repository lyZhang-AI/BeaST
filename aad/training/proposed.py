import logging
import os
import random
import time
from pathlib import Path

from tqdm import tqdm


from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from dotmap import DotMap
from aad.data import proposed_loader as data_loader3
from aad.data import avgc_loader
from aad.models.proposed import BeaST
from aad.graphs.left_brain2 import AdjMatrixGraph
from aad.graphs.right_brain2 import AdjMatrixGraph_right
import torch
import numpy as np

import torch.nn as nn
from torch.optim import Adam


result_logger = logging.getLogger('result')
result_logger.setLevel(logging.INFO)

PROJECT_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = Path(os.environ.get("AAD_CHECKPOINT_DIR", PROJECT_DIR / "checkpoints"))
DTU_DATA_DIR = os.environ.get("AAD_DTU_PATH", "./data/DTU/DATA_preproc")
KUL_DATA_DIR = os.environ.get("AAD_KUL_PATH", "./data/KUL/KUL_single_single3")
AVGC_DATA_DIR = os.environ.get("AAD_AVGC_PATH", "./data/AVGC")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_load_name(args, name=''):
    name = name if len(name) > 0 else 'default_model'
    return name


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    save_dir = CHECKPOINT_DIR / "BeaST"
    save_dir.mkdir(parents=True, exist_ok=True)
    fold_suffix = f"_fold{args.fold_id}" if getattr(args, "fold_id", None) is not None else ""
    condition_suffix = f"_{args.condition}" if args.dataset == "AVGC" else ""
    torch.save(model, save_dir / f"{args.dataset}{condition_suffix}_{args.method}_{name}_{args.time_len}s{fold_suffix}.pt")


def load_model(args, name=''):
    name = save_load_name(args, name)
    save_dir = CHECKPOINT_DIR / "BeaST"
    fold_suffix = f"_fold{args.fold_id}" if getattr(args, "fold_id", None) is not None else ""
    condition_suffix = f"_{args.condition}" if args.dataset == "AVGC" else ""
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.load(
        save_dir / f"{args.dataset}{condition_suffix}_{args.method}_{name}_{args.time_len}s{fold_suffix}.pt",
        map_location=map_location,
        weights_only=False,
    )


def getData(name="S1", time_len=2, dataset="DTU", fold_id=None,
            condition="NV", variant=None, label_mode="half"):
    DTU_document_path = DTU_DATA_DIR
    KUL_document_path = KUL_DATA_DIR
    if dataset == "DTU":
        return data_loader3.get_DTU_data(name, time_len, DTU_document_path, fold_id=fold_id)
    if dataset == "KUL":
        return data_loader3.get_KUL_data(name, time_len, KUL_document_path, fold_id=fold_id)
    if dataset == "AVGC":
        return avgc_loader.get_AVGC_data(
            name=name,
            timelen=time_len,
            data_document_path=AVGC_DATA_DIR,
            fold_id=0 if fold_id is None else fold_id,
            condition=condition,
            variant=variant,
            label_mode=label_mode,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def initiate(args, train_loader, valid_loader, test_loader, subject):
    A_left = AdjMatrixGraph().A_binary
    A_right = AdjMatrixGraph_right().A_binary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BeaST(
        num_class=2,
        num_point_left=27,
        num_point_right=27,
        num_g3d_scales=2,
        num_gcn_scales=2,
        in_channels=1,
        out_channels=16,
        c1=16,
        graph_left=A_left,
        graph_right=A_right,
    ).to(device)

    print(model)
    print(f"The model has {count_parameters(model):,} trainable parameters.")

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(params=model.parameters(), lr=0.001, weight_decay=3e-4)


    criterion = criterion.to(device)

    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion}

    return train_model(settings, args, train_loader, valid_loader, test_loader, subject)


def train_model(settings, args, train_loader, valid_loader, test_loader, subject):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    device = next(model.parameters()).device

    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator != 0 else 0.0

    def get_logits(model_output):
        if isinstance(model_output, (tuple, list)):
            return torch.stack(list(model_output), dim=0).mean(dim=0)
        return model_output

    def train(model, optimizer, criterion):
        model.train()
        train_acc_sum = 0
        train_loss_sum = 0
        batch_size = train_loader.batch_size

        for batch_data in train_loader:
            train_data_left, train_data_right, train_label = batch_data
            train_label = train_label.squeeze(-1)
            train_data_left = train_data_left.to(device).float()
            train_data_right = train_data_right.to(device).float()
            train_label = train_label.to(device).long()
            preds = get_logits(model(train_data_left, train_data_right))

            loss = criterion(preds, train_label.long())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            with torch.no_grad():
                train_loss_sum += loss.item() * batch_size
                predicted = preds.data.max(1)[1]
                train_acc_sum += predicted.eq(train_label).cpu().sum()

        return train_loss_sum / len(train_loader.dataset), train_acc_sum / len(train_loader.dataset)

    def evaluate(model, criterion, test=False):
        model.eval()
        if test:
            loader = test_loader
            num_batches = len(test_loader)
        else:
            loader = valid_loader
            num_batches = len(valid_loader)
        total_loss = 0.0
        test_acc_sum = 0
        all_labels = []
        all_preds = []
        batch_size = loader.batch_size

        start_time = time.time()
        with torch.no_grad():
            for batch_data in loader:
                test_data_left, test_data_right, test_label = batch_data
                test_label = test_label.squeeze(-1)
                test_data_left = test_data_left.to(device).float()
                test_data_right = test_data_right.to(device).float()
                test_label = test_label.to(device).long()

                preds = get_logits(model(test_data_left, test_data_right))
                total_loss += criterion(preds, test_label.long()).item() * batch_size
                preds = preds.detach()
                predicted = preds.data.max(1)[1]
                test_acc_sum += predicted.eq(test_label).cpu().sum()
                all_labels.append(test_label.cpu().numpy())
                all_preds.append(predicted.cpu().numpy())

        end_time = time.time()

        inference_time = end_time - start_time
        print(f'Inference time: {inference_time} seconds')

        avg_loss = total_loss / (num_batches * batch_size)
        avg_acc = test_acc_sum / (num_batches * batch_size)


        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)


        f1 = f1_score(all_labels, all_preds, average='macro')
        kappa = cohen_kappa_score(all_labels, all_preds)


        conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = safe_divide(tn, tn + fp)
        sensitivity = safe_divide(tp, tp + fn)

        if len(np.unique(all_labels)) == 2:
            auc = roc_auc_score(all_labels, all_preds)
        else:
            auc = None

        return avg_loss, avg_acc, f1, kappa, auc, specificity, sensitivity

    epochs_without_improvement = 0
    best_epoch = 1
    best_valid = float('inf')
    for epoch in tqdm(range(1, args.max_epoch + 1), desc='Training Epoch', leave=False):
        train_loss, train_acc = train(model, optimizer, criterion)
        val_loss, val_acc, val_f1, val_kappa, val_auc, val_specificity, val_sensitivity = evaluate(model, criterion, test=False)

        print()
        print(
            'Epoch {:2d} Finish | Subject {} | Train Loss {:5.4f} | Train Acc {:5.4f} | Valid Loss {:5.4f} | Valid Acc {:5.4f} | '
            'F1 {:5.4f} | Kappa {:5.4f} | AUC {:5.4f} | Specificity {:5.4f} | Sensitivity {:5.4f}'.format(
            epoch,
            args.name,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_f1,
            val_kappa,
            val_auc or 0.0,
            val_specificity,
            val_sensitivity
            ))

        if val_loss < best_valid:
            best_valid = val_loss
            epochs_without_improvement = 0

            best_epoch = epoch
            print(f"Saved model at {CHECKPOINT_DIR / 'BeaST'}.")
            save_model(args, model, name=args.name)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > 10:
                break

    model = load_model(args, name=args.name)

    test_loss, test_acc,test_f1,test_kappa,test_auc,test_specificity,test_sensitivity = evaluate(model, criterion, test=True)
    print(f'Best epoch: {best_epoch}')
    test_auc_value = test_auc if test_auc is not None else 0.0
    print(f"Subject: {subject}, Acc: {test_acc:.2f},F1:{test_f1:.2f}, Kappa:{test_kappa:.2f}, AUC:{test_auc_value:.2f},Specificity:{test_specificity:.2f},Sensitivity:{test_sensitivity:.2f}")

    return test_loss, test_acc,test_f1,test_kappa,test_auc,test_specificity,test_sensitivity


def main(name="S1", time_len=2, dataset="DTU", method="BeaST", fold_id=None,
         max_epoch=100, seed=42, condition="NV", variant=None, label_mode="half",
         avgc_root=None):
    setup_seed(seed)
    print(name)
    args = DotMap()
    args.name = name
    args.dataset = dataset
    args.method = method
    args.time_len = time_len
    args.fold_id = fold_id
    args.condition = condition
    args.max_epoch = max_epoch
    args.random_seed = seed
    args.both_feature = False
    if dataset == "AVGC" and avgc_root is not None:
        global AVGC_DATA_DIR
        AVGC_DATA_DIR = avgc_root
    train_loader, valid_loader, test_loader = getData(
        name,
        time_len,
        dataset,
        fold_id=fold_id,
        condition=condition,
        variant=variant,
        label_mode=label_mode,
    )
    print('Train data:', len(train_loader.dataset))
    print(train_loader.dataset.data_left.shape)

    loss, acc, f1, kappa, auc, specificity, sensitivity = initiate(
        args, train_loader, valid_loader, test_loader, args.name
    )

    print(f"Loss: {loss}, Acc: {acc.item()}, F1: {f1}, Kappa: {kappa}, AUC: {auc}, Specificity: {specificity}, Sensitivity: {sensitivity}")

    fold_msg = f'_fold{fold_id}' if fold_id is not None else ''
    condition_msg = f'_{condition}' if dataset == "AVGC" else ''
    info_msg = (f'{method}_{dataset}{condition_msg}_{name}_{str(time_len)}s{fold_msg} '
                f'loss:{str(loss)} acc:{str(acc.item())} '
                f'f1:{str(f1)} kappa:{str(kappa)} auc:{str(auc)} '
                f'specificity:{str(specificity)} sensitivity:{str(sensitivity)}')

    result_logger.info(info_msg)

    return loss, acc, f1, kappa, auc, specificity, sensitivity
