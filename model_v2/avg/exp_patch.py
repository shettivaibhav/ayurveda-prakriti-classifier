"""Test the fixed-patch colour approach AND establish the dual-dosha ceiling.

Patch idea:
  skin colour (SC) <- average colour of a CENTRE patch of the back-open-palm view
  palm colour (PC) <- average colour of a patch a little ABOVE centre of the
                      front-open-palm view

Also computes the hard ceiling: TRUE practitioner features -> dosha, for both the
3-class dominant and 7-class dual targets (this bounds ANY feature-based method).
Everything subject-level CV, scored against the sheet labels (manifest y7 / ydom).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")
RNG = 42
ROOT = Path(__file__).resolve().parents[2]
MANI = ROOT / "model_v2" / "results_full" / "manifest.csv"
FEAT = ROOT / "model_v2" / "avg" / "features" / "mobilenet_v2.npz"

GROUPS = {"SC": ["SC_Fair", "SC_Medium", "SC_Dark"],
          "PC": ["PC_PinkRed", "PC_PinkWhite", "PC_Brown"]}
ALLTAB = ["Age", "Sex_M", "Mole", "SC_Fair", "SC_Medium", "SC_Dark",
          "ST_Rough", "ST_Smooth", "ST_Dry", "ST_Oily", "SH_Thick", "SH_Thin", "SH_Med",
          "SHC_Brown", "SHC_Black", "VV_Low", "VV_Med", "VV_High",
          "PC_PinkRed", "PC_PinkWhite", "PC_Brown"]


def patch_color(path, cy_frac, half=0.16):
    """Mean colour (RGB+Lab+chroma) of a square patch centred at (W/2, H*cy_frac)."""
    if not isinstance(path, str) or not Path(path).exists():
        return None
    im = np.asarray(Image.open(path).convert("RGB")).astype(float)
    H, W = im.shape[:2]
    cx, cy = W // 2, int(H * cy_frac)
    hw, hh = int(W * half), int(H * half)
    patch = im[max(0, cy - hh):cy + hh, max(0, cx - hw):cx + hw].reshape(-1, 3) / 255.0
    rgb = patch.mean(0)
    lab = rgb2lab(patch.reshape(-1, 1, 3)).reshape(-1, 3).mean(0)
    s = patch.sum(1) + 1e-6
    chroma = np.stack([patch[:, 0] / s, patch[:, 1] / s, patch[:, 2] / s], 1).mean(0)
    return np.concatenate([rgb, lab, chroma, [(patch[:, 0] - patch[:, 1]).mean()]])


def cat(df, cols):
    M = df[cols].to_numpy()
    return np.where(M.sum(1) == 0, len(cols), M.argmax(1))


def rf():
    return RandomForestClassifier(n_estimators=600, class_weight="balanced_subsample",
                                  random_state=RNG)


def cvk():
    return StratifiedKFold(5, shuffle=True, random_state=RNG)


def oof_acc(X, y):
    return accuracy_score(y, cross_val_predict(rf(), X, y, cv=cvk()))


def main():
    df = pd.read_csv(MANI).reset_index(drop=True)
    z = np.load(FEAT, allow_pickle=True)
    emb, subj = z["emb"], z["subject_id"].astype(str)
    Ximg = np.stack([emb[subj == s].mean(0) for s in df["id"].astype(str)])

    # patches: SC from back centre, PC from front just-above-centre
    def col(view, cyf):
        out = [patch_color(p, cyf) for p in df[view]]
        # fill any missing (2 subjects miss open_front) with the column mean
        good = np.array([o for o in out if o is not None])
        mean = good.mean(0)
        return np.array([o if o is not None else mean for o in out])

    SCpatch = col("img_open_back", 0.50)    # centre of back
    PCpatch = col("img_open_front", 0.40)   # a little above centre of palm

    print("=== 1) Does the fixed-patch colour recover SC / PC? (held-out) ===")
    print(f"{'feature':<6}{'majority':>10}{'embeddings':>12}{'PATCH colour':>14}{'patch+emb':>12}")
    for g, cols in GROUPS.items():
        y = cat(df, cols)
        maj = np.bincount(y).max() / len(y)
        patch = SCpatch if g == "SC" else PCpatch
        print(f"{g:<6}{maj:>10.1%}{oof_acc(Ximg, y):>12.1%}{oof_acc(patch, y):>14.1%}"
              f"{oof_acc(np.hstack([patch, Ximg]), y):>12.1%}")

    print("\n=== 2) CEILING: TRUE practitioner features -> dosha (bounds any method) ===")
    ydom, y7 = df["ydom"].to_numpy(), df["y7"].to_numpy()
    Xtrue = df[ALLTAB].to_numpy(float)
    print(f"  true features -> 3-class dominant : {oof_acc(Xtrue, ydom):.1%}")
    print(f"  true features -> 7-class DUAL     : {oof_acc(Xtrue, y7):.1%}   "
          f"(majority {pd.Series(y7).value_counts().max()/len(y7):.1%})")

    print("\n=== 3) Pipeline: patch colour for SC/PC (+emb others) -> DUAL dosha ===")
    OTHERS = {"ST": ["ST_Rough", "ST_Smooth", "ST_Dry", "ST_Oily"],
              "SH": ["SH_Thick", "SH_Thin", "SH_Med"], "SHC": ["SHC_Brown", "SHC_Black"],
              "VV": ["VV_Low", "VV_Med", "VV_High"]}

    def onehot(pred, k):
        M = np.zeros((len(pred), k))
        for i, c in enumerate(pred):
            if c < k:
                M[i, c] = 1
        return M

    blocks = [df[["Age", "Sex_M"]].to_numpy(float),
              onehot(cross_val_predict(rf(), np.hstack([SCpatch, Ximg]), cat(df, GROUPS["SC"]), cv=cvk()), 3),
              onehot(cross_val_predict(rf(), np.hstack([PCpatch, Ximg]), cat(df, GROUPS["PC"]), cv=cvk()), 3),
              cross_val_predict(rf(), Ximg, df["Mole"].to_numpy(), cv=cvk()).reshape(-1, 1)]
    for g, cols in OTHERS.items():
        blocks.append(onehot(cross_val_predict(rf(), Ximg, cat(df, cols), cv=cvk()), len(cols)))
    Xpred = np.hstack(blocks)
    print(f"  pipeline (auto features) -> 7-class DUAL : {oof_acc(Xpred, y7):.1%}")
    print(f"  direct image             -> 7-class DUAL : {oof_acc(Ximg, y7):.1%}")


if __name__ == "__main__":
    main()
