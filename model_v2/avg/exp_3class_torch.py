"""Find the best DEPLOYABLE single-torch-model config for image-only 3-class dosha.

Compares, under subject-level 5x5 CV:
  fusion of views : meanpool (average the 4 view embeddings, then head)  vs
                    avgprob  (head per image, average the softmaxes)
  head            : linear   vs  mlp(256)
  loss            : weighted (balanced) vs plain
Picks the highest-accuracy config that stays a clean torch model (backbone + head).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

FEAT = Path(__file__).resolve().parent / "features" / "mobilenet_v2.npz"
VIEWS = ["open_front", "open_back", "closed_front", "closed_back"]


def make_head(kind, dim, nc):
    if kind == "linear":
        return nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, nc))
    return nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, 256), nn.ReLU(),
                         nn.Dropout(0.3), nn.Linear(256, nc))


def train_head(X, y, kind, dim, nc, cw, epochs=150, lr=1e-3, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    h = make_head(kind, dim, nc)
    opt = torch.optim.Adam(h.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(weight=None if cw is None else torch.tensor(cw, dtype=torch.float32))
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    h.train()
    for _ in range(epochs):
        opt.zero_grad(); crit(h(Xt), yt).backward(); opt.step()
    h.eval()
    return h


@torch.no_grad()
def proba(h, X):
    return torch.softmax(h(torch.tensor(X, dtype=torch.float32)), 1).numpy()


def main():
    z = np.load(FEAT, allow_pickle=True)
    emb, subj = z["emb"], z["subject_id"].astype(str)
    label = np.array([l[0] for l in z["label"].astype(str)])  # dominant dosha
    subjects = np.array(list(dict.fromkeys(subj)))
    le = LabelEncoder().fit(label)
    y_img = le.transform(label)
    y_subj = np.array([y_img[subj == s][0] for s in subjects])
    mean_emb = np.stack([emb[subj == s].mean(0) for s in subjects])  # subject meanpool
    nc, dim = len(le.classes_), emb.shape[1]

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    configs = [(fuse, head, wl) for fuse in ["meanpool", "avgprob"]
               for head in ["linear", "mlp"] for wl in ["weighted", "plain"]]
    scores = {c: {"acc": [], "bal": [], "f1": []} for c in configs}

    for fold, (tr, te) in enumerate(cv.split(subjects, y_subj)):
        tr_subj = set(subjects[tr])
        img_tr = np.isin(subj, list(tr_subj))
        for (fuse, head, wl) in configs:
            cw = None
            if fuse == "meanpool":
                Xtr, ytr = mean_emb[tr], y_subj[tr]
            else:
                Xtr, ytr = emb[img_tr], y_img[img_tr]
            if wl == "weighted":
                cw = compute_class_weight("balanced", classes=np.arange(nc), y=ytr)
            h = train_head(Xtr, ytr, head, dim, nc, cw, seed=fold)
            if fuse == "meanpool":
                pred = proba(h, mean_emb[te]).argmax(1)
            else:
                p_img = proba(h, emb[~img_tr])
                te_subj = subj[~img_tr]
                pred = np.array([p_img[te_subj == s].mean(0).argmax() for s in subjects[te]])
            yte = y_subj[te]
            scores[(fuse, head, wl)]["acc"].append(accuracy_score(yte, pred))
            scores[(fuse, head, wl)]["bal"].append(balanced_accuracy_score(yte, pred))
            scores[(fuse, head, wl)]["f1"].append(f1_score(yte, pred, average="macro", zero_division=0))

    print(f"3-class dominant dosha (K/P/V)  |  subject-level 5x5 CV  |  majority=64.1%")
    print(f"{'fuse':<10}{'head':<8}{'loss':<10}{'accuracy':>10}{'balanced':>10}{'macroF1':>9}")
    ranked = sorted(configs, key=lambda c: -np.mean(scores[c]["acc"]))
    for c in ranked:
        s = scores[c]
        print(f"{c[0]:<10}{c[1]:<8}{c[2]:<10}"
              f"{np.mean(s['acc']):>9.1%}{np.mean(s['bal']):>10.1%}{np.mean(s['f1']):>9.1%}")
    best = ranked[0]
    print(f"\nBEST accuracy: fuse={best[0]} head={best[1]} loss={best[2]} "
          f"-> {np.mean(scores[best]['acc']):.1%} acc / {np.mean(scores[best]['bal']):.1%} balanced")


if __name__ == "__main__":
    main()
