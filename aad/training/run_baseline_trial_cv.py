import argparse
import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score
from torch.optim import Adam
from tqdm import tqdm

from aad.training.trial_cv_baseline_data import get_trial_cv_loaders


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results" / "trial_cv"
MODEL_DIR = PROJECT_DIR / "checkpoints" / "baselines"

MODEL_ALIASES = {
    "eegnet": "EEGNet",
    "conformer": "Conformer",
    "g3d": "G3D",
    "stgcn": "STGCN",
    "stanet": "STANet",
    "dgcnn": "DGCNN",
    "cnn": "CNN",
    "tseption": "Tseption",
    "agslnet": "AGSLNet",
    "xanet": "XANet",
    "graph_eeg": "Graph_EEG",
    "grapheeg": "Graph_EEG",
}

INPUT_MODES = {
    "EEGNet": "image_time_channels",
    "Conformer": "image_channels_time",
    "G3D": "graph",
    "STGCN": "graph",
    "STANet": "image_channels_time",
    "DGCNN": "channels_time",
    "CNN": "channels_time",
    "Tseption": "image_channels_time",
    "AGSLNet": "graph",
    "XANet": "image_channels_time",
    "Graph_EEG": "graph",
}

DEFAULT_SUBJECTS = {
    "DTU": [f"S{i}" for i in range(1, 19)],
    "KUL": [f"S{i}" for i in range(1, 17)],
}

CSV_FIELDS = [
    "dataset",
    "method",
    "subject",
    "time_len",
    "fold",
    "protocol",
    "loss",
    "acc",
    "f1",
    "kappa",
    "auc",
    "specificity",
    "sensitivity",
    "best_epoch",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Trial-independent cross-validation runner for baseline models.")
    parser.add_argument("--dataset", choices=["KUL", "DTU", "all"], default="all")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["EEGNet", "Conformer", "G3D", "STGCN", "STANet", "DGCNN", "CNN", "Tseption"],
    )
    parser.add_argument("--time-lens", nargs="+", type=float, default=[2, 1, 0.5, 0.25])
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="Run only one batch through each requested model.")
    return parser.parse_args()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def canonical_model_name(name):
    key = name.lower()
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model {name}. Available: {', '.join(MODEL_ALIASES.values())}")
    return MODEL_ALIASES[key]


def build_model(model_name, time_len, sample_count):
    if model_name == "EEGNet":
        from aad.models.baselines.EEGNet import EEGNet

        return EEGNet()
    if model_name == "Conformer":
        from aad.models.baselines.Conformer import Conformer

        return Conformer()
    if model_name == "G3D":
        from aad.graphs.all import AdjMatrixGraph_all
        from aad.models.baselines.msg3d import Model

        return Model(
            num_class=2,
            num_point=64,
            num_g3d_scales=2,
            num_gcn_scales=2,
            in_channels=1,
            graph=AdjMatrixGraph_all().A_binary,
        )
    if model_name == "STGCN":
        from aad.models.baselines.STGCN import Model

        return Model(
            in_channels=1,
            num_class=2,
            graph_args={"layout": "all", "strategy": "spatial"},
            edge_importance_weighting=True,
        )
    if model_name == "STANet":
        from aad.models.baselines.STANet import STANet

        return STANet(n_classes=2)
    if model_name == "DGCNN":
        from aad.models.baselines.model_DGCNN import DGCNN

        return DGCNN(
            in_channels=sample_count,
            num_electrodes=64,
            k_adj=3,
            out_channels=64,
            num_classes=2,
        )
    if model_name == "CNN":
        from aad.models.baselines.CNN import AuditoryAttentionCNN2

        return AuditoryAttentionCNN2(samples=sample_count, n_chans=64)
    if model_name == "Tseption":
        from aad.models.baselines.Tseption import TSception

        return TSception(
            num_classes=2,
            input_size=(1, 64, sample_count),
            sampling_rate=128,
            num_T=15,
            num_S=15,
            hidden=8,
            dropout_rate=0.5,
        )
    if model_name == "AGSLNet":
        from aad.models.baselines.trial_cv_extra_models import AGSLNet

        return AGSLNet(num_channels=64, num_classes=2)
    if model_name == "XANet":
        from aad.models.baselines.trial_cv_extra_models import XANet

        return XANet(num_classes=2)
    if model_name == "Graph_EEG":
        from aad.models.baselines.trial_cv_extra_models import GraphEEG

        return GraphEEG(num_channels=64, num_classes=2)
    raise ValueError(model_name)


def count_parameters(model):
    total = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        try:
            total += parameter.numel()
        except ValueError:
            continue
    return total


def get_logits(output):
    if isinstance(output, (tuple, list)):
        output = torch.stack([get_logits(item) for item in output]).mean(dim=0)
    if output.ndim > 2:
        output = output.flatten(1)
    if output.shape[1] == 1:
        output = torch.cat([-output, output], dim=1)
    return output


def safe_divide(num, den):
    return float(num / den) if den else float("nan")


def compute_metrics(labels, logits):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    logits = np.asarray(logits)
    preds = np.argmax(logits, axis=1)
    acc = float(np.mean(preds == labels)) if len(labels) else float("nan")
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = safe_divide(tn, tn + fp)
    sensitivity = safe_divide(tp, tp + fn)
    if len(np.unique(labels)) == 2:
        try:
            prob = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()[:, 1]
            auc = roc_auc_score(labels, prob)
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")
    return acc, f1, kappa, auc, specificity, sensitivity


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_labels = []
    all_logits = []

    with torch.set_grad_enabled(is_train):
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)
            logits = get_logits(model(data))
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            all_labels.append(labels.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())

    labels_np = np.concatenate(all_labels)
    logits_np = np.concatenate(all_logits)
    metrics = compute_metrics(labels_np, logits_np)
    avg_loss = total_loss / len(labels_np)
    return (avg_loss,) + metrics


def checkpoint_path(model_dir, dataset, model_name, subject, time_len, fold):
    return model_dir / model_name / f"{dataset}_{model_name}_{subject}_{time_len:g}s_fold{fold}.pt"


def train_one(args, dataset, model_name, subject, time_len, fold):
    input_mode = INPUT_MODES[model_name]
    sample_count = int(math.ceil(128 * float(time_len)))
    train_loader, valid_loader, test_loader, protocol, n_folds = get_trial_cv_loaders(
        dataset=dataset,
        subject=subject,
        time_len=time_len,
        fold_id=fold,
        input_mode=input_mode,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, time_len, sample_count).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=3e-4)
    print(f"{dataset} {model_name} {subject} {time_len:g}s fold {fold + 1}/{n_folds}")
    print(f"Device: {device}; trainable parameters: {count_parameters(model):,}")

    if args.smoke_test:
        data, _ = next(iter(train_loader))
        with torch.no_grad():
            logits = get_logits(model(data.to(device)))
        print(f"Smoke-test output shape: {tuple(logits.shape)}")
        return None

    best_valid = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    ckpt_path = checkpoint_path(Path(args.model_dir), dataset, model_name, subject, time_len, fold)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(1, args.max_epoch + 1), desc="Training Epoch", leave=False):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        valid_metrics = run_epoch(model, valid_loader, criterion, device)
        print(
            f"Epoch {epoch:03d} | train loss {train_metrics[0]:.4f} acc {train_metrics[1]:.4f} | "
            f"valid loss {valid_metrics[0]:.4f} acc {valid_metrics[1]:.4f}"
        )
        if valid_metrics[0] < best_valid:
            best_valid = valid_metrics[0]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > args.patience:
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    loss, acc, f1, kappa, auc, specificity, sensitivity = run_epoch(model, test_loader, criterion, device)
    return {
        "dataset": dataset,
        "method": model_name,
        "subject": subject,
        "time_len": time_len,
        "fold": fold,
        "protocol": protocol,
        "loss": loss,
        "acc": acc,
        "f1": f1,
        "kappa": kappa,
        "auc": auc,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "best_epoch": best_epoch,
    }


def result_file(results_dir, dataset, model_name):
    return results_dir / f"{dataset}_{model_name}_fold_results.csv"


def load_completed(path):
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row["dataset"], row["method"], row["subject"], float(row["time_len"]), int(row["fold"])))
    return completed


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    setup_seed(args.seed)
    results_dir = Path(args.results_dir)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    datasets = ["KUL", "DTU"] if args.dataset == "all" else [args.dataset]
    models = [canonical_model_name(name) for name in args.models]

    for dataset in datasets:
        subjects = args.subjects or DEFAULT_SUBJECTS[dataset]
        for model_name in models:
            out_csv = result_file(results_dir, dataset, model_name)
            completed = load_completed(out_csv)
            for subject in subjects:
                for time_len in args.time_lens:
                    fold_ids = args.folds
                    if fold_ids is None:
                        fold_ids = list(range(16 if dataset == "KUL" else 10))
                    for fold in fold_ids:
                        key = (dataset, model_name, subject, float(time_len), int(fold))
                        if key in completed:
                            print(f"Skip completed: {key}")
                            continue
                        row = train_one(args, dataset, model_name, subject, time_len, fold)
                        if row is not None:
                            append_row(out_csv, row)


if __name__ == "__main__":
    main()
