# Trial-Independent EEG Auditory Attention Detection

This repository contains the trial-independent implementation of **BeaST** and the comparison methods.

The within-trial protocol is intentionally excluded.

## Experimental protocol

For every subject, complete trials are split into train/validation/test (or trial-level cross-validation) **before** windowing. Windows from one trial never appear in different splits. Any train-fitted preprocessing must be fitted using training trials only.

Pipeline:

```text
raw/preprocessed EEG -> trial-level split -> sliding windows -> model training
                     -> validation model selection -> held-out test metrics
```

## Repository layout

```text
aad/
  data/                 # trial splitting and dataset-specific loaders
  graphs/               # EEG electrode adjacency definitions
  models/
    proposed/           # our model and its building blocks
    baselines/          # DARNet and comparison models
  training/             # training and evaluation routines
cli/                    # executable commands only
configs/                # experiment templates
```

`aad/models/proposed/msg3d5.py::BeaST` is the proposed model. Its loader is `aad/data/proposed_loader.py`, and its optimisation/validation/test loop is `aad/training/proposed.py`.

## Data paths

Raw EEG data are not distributed here. Set paths in the local environment:

```bash
export AAD_DTU_PATH=/path/to/DTU/DATA_preproc
export AAD_KUL_PATH=/path/to/KUL/KUL_single_single3
export AAD_AVGC_PATH=/path/to/AV-GC/gaze_KUL
```

Expected formats are documented by the loader: DTU uses `<subject>_data_preproc.mat`; KUL uses `No/<subject>Tra*.csv` and `csv/<subject>No.csv`.
AV-GC uses `<condition>/<variant>/<subject>/`, with `csv/<label-file>.csv` and
`<subject>Tra*.csv` containing 64 EEG channels. Supported conditions are `NV`,
`SV`, `MV`, and `MTN`; raw AV-GC recordings are not included in GitHub.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run our method

The entry point is intentionally limited to one subject and one fold per command:

```bash
python -m cli.train_my --dataset KUL --subject S1 --time-len 2 --fold 0
```

For AV-GC at 0.25 seconds, fold 0 holds out trial 1 and fold 1 holds out trial
2. Validation trials are selected from the remaining trials before windowing:

```bash
python -m cli.train_my --dataset AVGC --subject S1 --time-len 0.25 \
  --fold 0 --condition NV --avgc-root /path/to/AV-GC/gaze_KUL
python -m cli.train_my --dataset AVGC --subject S1 --time-len 0.25 \
  --fold 1 --condition NV --avgc-root /path/to/AV-GC/gaze_KUL
```

Omit `--fold` for one trial-independent train/validation/test split. Checkpoints are saved in `checkpoints/BeaST/` and ignored by Git.

## Run baselines

Baseline implementations live under `aad/models/baselines/`, independently of the proposed-model training pipeline. This keeps their data shape conversions and hyperparameters explicit per method.

## Citation and license

Add the paper citation and the license selected by the authors before publishing. Dataset access and redistribution follow the original dataset terms.
