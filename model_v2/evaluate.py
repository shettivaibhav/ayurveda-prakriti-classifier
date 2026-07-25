"""Final refit: trains the winning config (from cv_results.json) on all 80
image-matched subjects and saves a deployable checkpoint.

Run after train.py has produced results/cv_results.json for at least one
config. Picks the config with the highest acc_mean unless --config is given
explicitly.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from data import LABEL_COL, VIEW_ORDER, build_dataset
from datasets import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_class_mappings,
    train_transforms,
)
from train import make_dataset, make_model

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_best_config() -> str:
    cv_path = RESULTS_DIR / "cv_results.json"
    if not cv_path.exists():
        raise FileNotFoundError(f"{cv_path} not found - run train.py for at least one config first")
    with open(cv_path) as f:
        cv_results = json.load(f)
    best = max(cv_results.items(), key=lambda kv: kv[1]["acc_mean"])
    print(f"Best config from CV: {best[0]} (acc_mean={best[1]['acc_mean']:.4f})")
    return best[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["single", "mean", "concat"], default=None)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    config = args.config or pick_best_config()

    df = build_dataset()
    class_to_idx, idx_to_class = build_class_mappings(df[LABEL_COL])
    num_classes = len(class_to_idx)
    labels = df[LABEL_COL].map(class_to_idx).to_numpy()

    train_ds = make_dataset(config, df, class_to_idx, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)

    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    model = make_model(config, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    print(f"Refitting '{config}' on all {len(df)} subjects for {args.epochs} epochs...")
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for images, batch_labels in train_loader:
            images, batch_labels = images.to(DEVICE), batch_labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
        print(f"  epoch {epoch + 1}/{args.epochs}: loss={epoch_loss / len(df):.4f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CHECKPOINT_DIR / "model_v2_final.pth"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "model_architecture": "MobileNetV2",
        "fusion_mode": config if config != "single" else None,
        "num_views": 1 if config == "single" else len(VIEW_ORDER),
        "view_order": None if config == "single" else VIEW_ORDER,
        "num_classes": num_classes,
        "input_size": (224, 224),
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    }
    torch.save(checkpoint, save_path)
    print(f"\nSaved final checkpoint to {save_path}")
    print(f"Classes: {idx_to_class}")


if __name__ == "__main__":
    main()
