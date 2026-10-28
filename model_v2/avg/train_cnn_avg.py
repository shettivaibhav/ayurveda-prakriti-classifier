"""Frozen-backbone MobileNetV2 averaging model — CV evaluation + deployable checkpoint.

This is the fix for the old model_v2/train.py, which fine-tuned the WHOLE MobileNetV2
backbone (~2.2M params) on ~68 subjects/fold and overfit to ~chance. Here the
ImageNet backbone stays FROZEN and only a small classification head is trained — the
right capacity for n=92. Because the backbone is frozen, training the head on the
cached MobileNetV2 embeddings (features/mobilenet_v2.npz) is identical to training the
full frozen model, and far faster.

Two "average the 4 views" designs, both under subject-level CV:
  scheme_A : one head trained on all training-subject images (views pooled); at test,
             average a subject's per-image softmaxes  -> the deployed design.
  scheme_B : one head per hand side (open_front/open_back/closed_front/closed_back);
             at test, average the per-view heads' softmaxes.

Then a final head is trained on all 366 images and assembled with the frozen backbone
into a single-view WristClassifier saved to mini/model_v2.pth, with
inference_mode="average_predictions" for the archived CNN inference workflow.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torchvision import models

HERE = Path(__file__).resolve().parent
FEAT = HERE / "features" / "mobilenet_v2.npz"
DEPLOY_PATH = HERE.parents[1] / "mini" / "model_v2.pth"
RESULTS = HERE / "results_cnn_avg.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VIEW_ORDER = ["palm_open_front", "palm_open_back", "palm_closed_front", "palm_closed_back"]
VIEW_KEY = {"open_front": "palm_open_front", "open_back": "palm_open_back",
            "closed_front": "palm_closed_front", "closed_back": "palm_closed_back"}
N_SPLITS, N_REPEATS = 5, 5
RNG = 42


# ---- deployable single-view model: FROZEN MobileNetV2 backbone + trainable head ----
class WristClassifier(nn.Module):
    """Single-view classifier whose backbone/head names match the saved checkpoints."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        in_f = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Identity()
        self.head = make_head(in_f, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def make_head(dim, num_classes):
    return nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )


def train_head(X, y, dim, num_classes, class_w, epochs=120, lr=1e-3, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    head = make_head(dim, num_classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32))
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    head.train()
    for _ in range(epochs):  # full-batch GD; data is tiny
        opt.zero_grad()
        loss = crit(head(Xt), yt)
        loss.backward()
        opt.step()
    head.eval()
    return head


@torch.no_grad()
def head_proba(head, X):
    return torch.softmax(head(torch.tensor(X, dtype=torch.float32)), dim=1).numpy()


def _metrics(ytrue, ypred):
    return {
        "acc": accuracy_score(ytrue, ypred),
        "bal": balanced_accuracy_score(ytrue, ypred),
        "f1": f1_score(ytrue, ypred, average="macro", zero_division=0),
    }


def load(dominant=False):
    z = np.load(FEAT, allow_pickle=True)
    emb = z["emb"]
    subj = z["subject_id"].astype(str)
    view = z["view"].astype(str)
    label = z["label"].astype(str)
    if dominant:
        # collapse dual-dosha Prakriti to the DOMINANT dosha (first letter): K/P/V.
        # 3-class is the framing the old, practitioner-validated model used.
        label = np.array([l[0] for l in label])
    subjects = np.array(list(dict.fromkeys(subj)))
    y_subj_str = np.array([label[subj == s][0] for s in subjects])
    return emb, subj, view, label, subjects, y_subj_str


def cv_eval(dominant):
    emb, subj, view, label, subjects, y_subj_str = load(dominant)
    le = LabelEncoder().fit(y_subj_str)
    y_subj = le.transform(y_subj_str)
    y_img = le.transform(label)
    nc, dim = len(le.classes_), emb.shape[1]
    views = list(VIEW_KEY.keys())

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RNG)
    A = {"acc": [], "bal": [], "f1": []}
    B = {"acc": [], "bal": [], "f1": []}
    cmA = np.zeros((nc, nc), int)

    for fold, (tr_i, te_i) in enumerate(cv.split(subjects, y_subj)):
        train_subj = set(subjects[tr_i])
        tr = np.isin(subj, list(train_subj))
        te = ~tr
        test_subjects = subjects[te_i]
        yte = y_subj[te_i]

        # --- scheme A: one pooled head, average per-subject image softmaxes ---
        cw = compute_class_weight("balanced", classes=np.arange(nc), y=y_img[tr])
        headA = train_head(emb[tr], y_img[tr], dim, nc, cw, seed=fold)
        pA = head_proba(headA, emb[te])
        te_subj = subj[te]
        predA = np.array([pA[te_subj == s].mean(0).argmax() for s in test_subjects])
        m = _metrics(yte, predA)
        for k in A:
            A[k].append(m[k])
        cmA += confusion_matrix(yte, predA, labels=range(nc))

        # --- scheme B: per-view heads, average the views ---
        headsB = {}
        for v in views:
            trv = tr & (view == v)
            cwv = compute_class_weight("balanced", classes=np.arange(nc), y=y_img[trv])
            headsB[v] = train_head(emb[trv], y_img[trv], dim, nc, cwv, seed=fold)
        predB = []
        for s in test_subjects:
            probs = []
            for v in views:
                mm = te & (view == v) & (subj == s)
                if mm.any():
                    probs.append(head_proba(headsB[v], emb[mm])[0])
            predB.append(np.mean(probs, axis=0).argmax())
        m = _metrics(yte, np.array(predB))
        for k in B:
            B[k].append(m[k])

    def summ(d):
        return {k: [float(np.mean(v)), float(np.std(v))] for k, v in d.items()}

    maj = np.bincount(y_subj).max() / len(y_subj)
    return {
        "n_subjects": int(len(subjects)), "n_images": int(len(subj)),
        "classes": list(le.classes_),
        "baselines": {"random": 1 / nc, "majority": float(maj)},
        "scheme_A_pooled_avg": summ(A),
        "scheme_B_perview_avg": summ(B),
        "scheme_A_confusion": cmA.tolist(),
    }, le


def fit_and_save(le, dominant):
    """Train the final head on ALL images and save a deployable checkpoint."""
    emb, subj, view, label, subjects, _ = load(dominant)
    y_img = le.transform(label)
    nc, dim = len(le.classes_), emb.shape[1]
    cw = compute_class_weight("balanced", classes=np.arange(nc), y=y_img)
    head = train_head(emb, y_img, dim, nc, cw, seed=RNG)

    model = WristClassifier(num_classes=nc, pretrained=True)
    model.head.load_state_dict(head.state_dict())
    model.eval()

    # cast every label to a native python str/int so the checkpoint contains no
    # numpy scalars (torch>=2.6 loads with weights_only=True by default and rejects them)
    idx_to_class = {int(i): str(c) for i, c in enumerate(le.classes_)}
    ckpt = {
        "model_state_dict": model.state_dict(),
        "arch": "single_view_mobilenet_v2",
        "num_classes": int(nc),
        "idx_to_class": idx_to_class,
        "class_to_idx": {str(c): int(i) for i, c in enumerate(le.classes_)},
        "input_size": [224, 224],
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "view_order": VIEW_ORDER,
        "inference_mode": "average_predictions",
        "dosha_level": "dominant" if nc == 3 else "prakriti",
        "notes": "Frozen ImageNet MobileNetV2 backbone + trained head. At inference, run "
                 "each hand view and average the softmaxes. dosha_level=dominant predicts "
                 "the single dominant dosha (Kapha/Pitta/Vata).",
    }
    torch.save(ckpt, DEPLOY_PATH)
    return idx_to_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["prakriti7", "dominant3"], default="dominant3",
                    help="prakriti7 = 7-class dual dosha; dominant3 = 3-class Kapha/Pitta/Vata")
    args = ap.parse_args()
    dominant = args.target == "dominant3"

    print(f"Cross-validating frozen-backbone MobileNetV2 averaging (subject-level) "
          f"[target={args.target}]...")
    res, le = cv_eval(dominant)
    a, b = res["scheme_A_pooled_avg"], res["scheme_B_perview_avg"]
    print(f"\nn={res['n_subjects']} subjects / {res['n_images']} images   classes={res['classes']}")
    print(f"baselines: random={res['baselines']['random']:.3f}  majority={res['baselines']['majority']:.3f}\n")
    print(f"{'scheme':<26}{'acc':>8}{'bal_acc':>9}{'macroF1':>9}")
    print(f"{'A: pooled + avg preds':<26}{a['acc'][0]:>8.3f}{a['bal'][0]:>9.3f}{a['f1'][0]:>9.3f}")
    print(f"{'B: per-view + avg':<26}{b['acc'][0]:>8.3f}{b['bal'][0]:>9.3f}{b['f1'][0]:>9.3f}")

    idx_to_class = fit_and_save(le, dominant)
    res["target"] = args.target
    res["deployed"] = {"path": str(DEPLOY_PATH), "idx_to_class": idx_to_class,
                       "scheme": "A_pooled_avg"}
    out = HERE / (f"results_cnn_avg_{'3class' if dominant else '7class'}.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"\nSaved deployable checkpoint -> {DEPLOY_PATH}  (num_classes={len(idx_to_class)})")
    print(f"Saved CV results -> {out}")


if __name__ == "__main__":
    main()
