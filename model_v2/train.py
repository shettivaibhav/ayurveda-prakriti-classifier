"""Cross-validated training for model_v2.

Evaluates three configs on the 7-class dual-dosha Prakriti target:
  - single: single-view baseline (palm_open_back only) - control, matches
            the architecture of the currently-deployed model but retrained
            on the correct 7-class practitioner labels.
  - mean:   4-view MobileNetV2 with mean-pooled fusion.
  - concat: 4-view MobileNetV2 with concatenated fusion (primary design).

Uses RepeatedStratifiedKFold at the subject level. n_splits is capped by the
smallest class count (currently 4, for P-K) so folds stay stratifiable.
Reports mean +/- std accuracy and macro-F1 across all fold-repeats, since a
single hold-out split is unreliable at this sample size.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from data import LABEL_COL, build_dataset
from datasets import (
    WristMultiViewDataset,
    WristSingleViewDataset,
    build_class_mappings,
    train_transforms,
    val_transforms,
)
from models import WristClassifier, WristMultiViewClassifier

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(config: str, num_classes: int) -> nn.Module:
    if config == "single":
        return WristClassifier(num_classes=num_classes, pretrained=True)
    if config == "mean":
        return WristMultiViewClassifier(num_classes=num_classes, fusion_mode="mean", pretrained=True)
    if config == "concat":
        return WristMultiViewClassifier(num_classes=num_classes, fusion_mode="concat", pretrained=True)
    raise ValueError(f"Unknown config: {config}")


def make_dataset(config: str, df, class_to_idx, transform):
    if config == "single":
        return WristSingleViewDataset(df, class_to_idx, transform=transform, view="palm_open_back")
    return WristMultiViewDataset(df, class_to_idx, transform=transform)


def train_one_fold(config: str, train_df, val_df, class_to_idx, num_classes, max_epochs, patience):
    train_ds = make_dataset(config, train_df, class_to_idx, train_transforms)
    val_ds = make_dataset(config, val_df, class_to_idx, val_transforms)

    train_labels = train_df[LABEL_COL].map(class_to_idx).to_numpy()

    # Class-weighted loss handles the imbalance; a WeightedRandomSampler on
    # top of that double-corrects and destabilizes training on this few
    # samples/class - use loss weighting only, plain shuffling for the loader.
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=train_labels)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    model = make_model(config, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc = -1.0
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_acc, best_val_f1


def run_cv(config: str, n_splits: int, n_repeats: int, max_epochs: int, patience: int, seed: int = 42):
    df = build_dataset()
    class_to_idx, idx_to_class = build_class_mappings(df[LABEL_COL])
    num_classes = len(class_to_idx)
    labels = df[LABEL_COL].to_numpy()

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    accs, f1s = [], []
    fold_log = []
    t_start = time.time()

    for i, (train_idx, val_idx) in enumerate(rskf.split(df, labels)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        val_acc, val_f1 = train_one_fold(
            config, train_df, val_df, class_to_idx, num_classes, max_epochs, patience
        )
        accs.append(val_acc)
        f1s.append(val_f1)
        fold_log.append({"fold": i, "val_acc": val_acc, "val_f1": val_f1})

        elapsed = time.time() - t_start
        print(
            f"[{config}] fold {i + 1}/{n_splits * n_repeats}: "
            f"acc={val_acc:.4f} f1={val_f1:.4f} (elapsed {elapsed:.0f}s)",
            flush=True,
        )

    summary = {
        "config": config,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "n_folds_total": n_splits * n_repeats,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
        "idx_to_class": idx_to_class,
        "fold_log": fold_log,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["single", "mean", "concat"], required=True)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    summary = run_cv(
        args.config, args.n_splits, args.n_repeats, args.max_epochs, args.patience
    )

    print(f"\n[{args.config}] acc = {summary['acc_mean']:.4f} +/- {summary['acc_std']:.4f}")
    print(f"[{args.config}] macro-F1 = {summary['f1_macro_mean']:.4f} +/- {summary['f1_macro_std']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "cv_results.json"
    all_results = {}
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    all_results[args.config] = summary
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote results to {results_path}")


if __name__ == "__main__":
    main()
