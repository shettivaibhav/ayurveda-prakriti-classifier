"""Concept-bottleneck test: image -> practitioner concepts -> dosha.

Can we AUTO-FILL the tabular features from the hand images (a 'hair-check', 'vein-check',
'colour-check' model per feature), then let the strong tabular model classify dosha?

Stage 1: for each practitioner feature (SC skin-colour, ST texture, SH hair, SHC hair-
         colour, VV veins, PC palm-colour, Mole), train an image->feature classifier on
         frozen MobileNetV2 embeddings. Report how recoverable each feature is.
Stage 2: out-of-fold predict every feature from images, assemble a PREDICTED tabular
         matrix (Age/Sex are self-reported, kept true), and predict 3-class dominant
         dosha from it. Compare to the ceiling (TRUE features) and floor (image->dosha).

All subject-level 5x5 CV. Also tries a simple palm-colour signal (mean HSV) to see if
raw colour beats deep embeddings for the colour features.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RNG = 42
ROOT = Path(__file__).resolve().parents[2]
MANI = ROOT / "model_v2" / "results_full" / "manifest.csv"
FEAT = ROOT / "model_v2" / "avg" / "features" / "mobilenet_v2.npz"

GROUPS = {
    "SC": ["SC_Fair", "SC_Medium", "SC_Dark"],
    "ST": ["ST_Rough", "ST_Smooth", "ST_Dry", "ST_Oily"],
    "SH": ["SH_Thick", "SH_Thin", "SH_Med"],   # all-zero -> 'None' (no hair)
    "SHC": ["SHC_Brown", "SHC_Black"],          # all-zero -> 'None'
    "VV": ["VV_Low", "VV_Med", "VV_High"],
    "PC": ["PC_PinkRed", "PC_PinkWhite", "PC_Brown"],
}


def cat_from_onehot(df, cols):
    """One-hot block -> single categorical index; all-zero -> extra 'None' class."""
    M = df[cols].to_numpy()
    y = np.where(M.sum(1) == 0, len(cols), M.argmax(1))
    return y, len(cols) + 1


def color_stats(paths):
    """Mean H,S,V of each subject's open_back palm image (cheap raw-colour signal)."""
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((64, 64)).convert("HSV")
        out.append(np.array(im).reshape(-1, 3).mean(0))
    return np.array(out)


def cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)


def clf():
    return RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                  random_state=RNG)


def main():
    df = pd.read_csv(MANI).reset_index(drop=True)
    z = np.load(FEAT, allow_pickle=True)
    emb, subj = z["emb"], z["subject_id"].astype(str)
    Ximg = np.stack([emb[subj == s].mean(0) for s in df["id"].astype(str)])
    Xcol = color_stats(df["img_open_back"])
    ydom = df["ydom"].to_numpy()

    print("Stage 1 — how recoverable is each feature from the images? (subject-level CV)")
    print(f"{'feature':<7}{'#cls':>5}{'majority':>10}{'img-acc':>9}{'colour-acc':>11}")
    pred_feats = {}      # out-of-fold predicted category per feature (from images)
    for g, cols in GROUPS.items():
        y, ncls = cat_from_onehot(df, cols)
        maj = np.bincount(y).max() / len(y)
        acc_img = accuracy_score(y, cross_val_predict(clf(), Ximg, y, cv=cv()))
        acc_col = accuracy_score(y, cross_val_predict(clf(), Xcol, y, cv=cv()))
        pred_feats[g] = (cross_val_predict(clf(), Ximg, y, cv=cv()), cols)
        star = "  <- colour" if acc_col > acc_img else ""
        print(f"{g:<7}{ncls:>5}{maj:>10.1%}{acc_img:>9.1%}{acc_col:>11.1%}{star}")
    # Mole (binary)
    ym = df["Mole"].to_numpy()
    acc_m = accuracy_score(ym, cross_val_predict(clf(), Ximg, ym, cv=cv()))
    print(f"{'Mole':<7}{2:>5}{max(ym.mean(),1-ym.mean()):>10.1%}{acc_m:>9.1%}{'-':>11}")

    # ---- Stage 2: assemble predicted tabular matrix, predict dosha ----
    def onehot(pred, cols):
        M = np.zeros((len(pred), len(cols)))
        for i, c in enumerate(pred):
            if c < len(cols):
                M[i, c] = 1
        return M

    age_sex = df[["Age", "Sex_M"]].to_numpy(float)   # self-reported -> keep true
    pred_blocks = [onehot(pred_feats[g][0], cols) for g, (_, cols) in
                   [(g, pred_feats[g]) for g in GROUPS]]
    mole_pred = cross_val_predict(clf(), Ximg, ym, cv=cv()).reshape(-1, 1)
    X_pred = np.hstack([age_sex, mole_pred] + pred_blocks)

    TAB = ["Age", "Sex_M", "Mole"] + [c for cols in GROUPS.values() for c in cols]
    X_true = df[TAB].to_numpy(float)

    def dosha_acc(X):
        p = cross_val_predict(clf(), X, ydom, cv=cv())
        return accuracy_score(ydom, p), balanced_accuracy_score(ydom, p)

    a_true = dosha_acc(X_true)
    a_pred = dosha_acc(X_pred)
    a_img = dosha_acc(Ximg)
    maj = np.bincount(pd.factorize(ydom)[0]).max() / len(ydom)

    print("\nStage 2 — 3-class dominant dosha (majority = {:.1%})".format(maj))
    print(f"  CEILING  true practitioner features -> {a_true[0]:.1%} acc / {a_true[1]:.1%} bal")
    print(f"  PIPELINE image-derived features     -> {a_pred[0]:.1%} acc / {a_pred[1]:.1%} bal")
    print(f"  FLOOR    image embeddings -> dosha   -> {a_img[0]:.1%} acc / {a_img[1]:.1%} bal")


if __name__ == "__main__":
    main()
