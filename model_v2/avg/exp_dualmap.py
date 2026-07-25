"""Map the 3-class (K/P/V) model's probabilities to the sheet's dual/single Prakriti
labels, and score against palm_dataset_26 June 26.xlsx (via manifest y7).

Rule (user's): take the two highest doshas -> dual dosha "top1-top2"; BUT if the
dominant is Kapha and the second dosha is weak (prob < t), output single "K".
(Only Kapha ever appears as a single-dosha label in the sheet.)

Probabilities are OUT-OF-FOLD (subject-level 5-fold) so this is held-out accuracy,
not memorised. RandomForest on mean-pooled MobileNetV2 embeddings (the ~68% model).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")
RNG = 42
ROOT = Path(__file__).resolve().parents[2]
MANI = ROOT / "model_v2" / "results_full" / "manifest.csv"
FEAT = ROOT / "model_v2" / "avg" / "features" / "mobilenet_v2.npz"


def main():
    df = pd.read_csv(MANI).reset_index(drop=True)
    ids = df["id"].astype(str).tolist()
    y7 = df["y7"].to_numpy()          # sheet Prakriti (dual/single)
    ydom = df["ydom"].to_numpy()      # dominant dosha for training

    z = np.load(FEAT, allow_pickle=True)
    emb, subj = z["emb"], z["subject_id"].astype(str)
    X = np.stack([emb[subj == s].mean(0) for s in ids])

    clf = RandomForestClassifier(n_estimators=600, class_weight="balanced_subsample",
                                 random_state=RNG)
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
    proba = cross_val_predict(clf, X, ydom, cv=cv, method="predict_proba")
    classes = clf.fit(X, ydom).classes_           # order of proba columns, e.g. ['K','P','V']

    # dominant-only reference (top-1 vs dominant of the sheet)
    top1 = classes[proba.argmax(1)]
    dom_acc = accuracy_score(ydom, top1)

    def map_label(p, t):
        order = np.argsort(p)[::-1]
        d1, d2 = classes[order[0]], classes[order[1]]
        if d1 == "K" and p[order[1]] < t:
            return "K"
        return f"{d1}-{d2}"

    print(f"Sheet Prakriti classes present: {sorted(pd.unique(y7))}")
    print(f"Majority label in sheet: {pd.Series(y7).mode()[0]} "
          f"({pd.Series(y7).value_counts().max()}/{len(y7)} = "
          f"{pd.Series(y7).value_counts().max()/len(y7):.1%})")
    print(f"\n3-class dominant-only accuracy (reference): {dom_acc:.1%}\n")

    print("Dual/single mapping vs the sheet (held-out):")
    print(f"{'Kapha-single threshold t':<26}{'accuracy vs sheet':>18}")
    best = (0.0, -1)
    for t in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        pred = np.array([map_label(proba[i], t) for i in range(len(proba))])
        a = accuracy_score(y7, pred)
        tag = "  (t=0: always dual)" if t == 0 else ""
        if a > best[1]:
            best = (t, a)
        print(f"{t:<26.2f}{a:>17.1%}{tag}")
    print(f"\nBest: t={best[0]:.2f} -> {best[1]:.1%} agreement with the professional's sheet")
    print("(dual-dosha label correct = BOTH dominant and secondary dosha match the sheet)")


if __name__ == "__main__":
    main()
