"""Ceiling for a FULLY-AUTOMATED model: inputs = 4 hand images + age + sex only.
No practitioner features (not recoverable from photos). 3-class dominant dosha.

Tries: each backbone alone, the best backbone + age/sex, and an ALL-BACKBONE ensemble
(concatenate the 5 frozen embeddings) + age/sex — to extract every bit of automated
signal. Subject-level 5x5 CV, accuracy + balanced-accuracy, vs majority 64.1%.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
RNG = 42
ROOT = Path(__file__).resolve().parents[2]
MANI = ROOT / "model_v2" / "results_full" / "manifest.csv"
FDIR = ROOT / "model_v2" / "avg" / "features"
BACKBONES = ["mobilenet_v2", "resnet50", "efficientnet_b0", "convnext_tiny", "dinov2_vits14"]


def meanpool(name, ids):
    z = np.load(FDIR / f"{name}.npz", allow_pickle=True)
    emb, subj = z["emb"], z["subject_id"].astype(str)
    return np.stack([emb[subj == s].mean(0) for s in ids])


def models(highdim):
    m = {
        "RandomForest": RandomForestClassifier(n_estimators=600,
                        class_weight="balanced_subsample", random_state=RNG),
        "LogReg": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RNG),
        "SVM-RBF": SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=RNG),
    }
    if highdim:
        pca = lambda c: Pipeline([("sc", StandardScaler()),
                                  ("pca", PCA(n_components=40, random_state=RNG)), ("c", c)])
        return {k: pca(v) for k, v in m.items()}
    return {"RandomForest": m["RandomForest"],
            "LogReg": Pipeline([("sc", StandardScaler()), ("c", m["LogReg"])]),
            "SVM-RBF": Pipeline([("sc", StandardScaler()), ("c", m["SVM-RBF"])])}


def evalX(name, X, y, highdim=True):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
    best = (None, -1, -1)
    for mn, mdl in models(highdim).items():
        r = cross_validate(mdl, X, y, cv=cv, scoring=["accuracy", "balanced_accuracy"], n_jobs=-1)
        a, b = r["test_accuracy"].mean(), r["test_balanced_accuracy"].mean()
        if a > best[1]:
            best = (mn, a, b)
    print(f"{name:<34}{best[1]:>9.1%}{best[2]:>10.1%}   [{best[0]}]")
    return best


def main():
    df = pd.read_csv(MANI).reset_index(drop=True)
    ids = df["id"].astype(str).tolist()
    y = df["ydom"].to_numpy()
    agesex = df[["Age", "Sex_M"]].to_numpy(float)
    maj = pd.Series(y).value_counts().max() / len(y)

    emb = {b: meanpool(b, ids) for b in BACKBONES}
    allb = np.hstack([emb[b] for b in BACKBONES])

    print(f"3-class dominant dosha | fully automated inputs | majority = {maj:.1%}")
    print(f"{'inputs':<34}{'accuracy':>9}{'balanced':>10}")
    for b in BACKBONES:
        evalX(f"{b}", emb[b], y)
    print("-" * 55)
    evalX("mobilenet + age/sex", np.hstack([emb["mobilenet_v2"], agesex]), y)
    evalX("ALL 5 backbones (ensemble)", allb, y)
    evalX("ALL 5 backbones + age/sex", np.hstack([allb, agesex]), y)
    evalX("age + sex ONLY (no image)", agesex, y, highdim=False)


if __name__ == "__main__":
    main()
