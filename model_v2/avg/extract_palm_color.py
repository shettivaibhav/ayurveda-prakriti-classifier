"""Proper palm-colour extraction, then re-test recovery of the colour features
(palm colour PC, skin colour SC) and the concept-bottleneck pipeline.

Fixes the earlier crude signal (wrong view + whole-image mean):
  1. use the PALM-FACING view (open_front) for palm colour, back (open_back) for skin.
  2. white-balance each photo (neutralise the light-background colour cast).
  3. skin-segment (YCbCr) and keep the largest hand blob -> measure only skin pixels.
  4. describe colour in perceptual Lab + HSV + normalised-rgb (discriminates
     PinkRed / PinkWhite / Brown far better than raw RGB).

Compares PC/SC recovery for: crude-HSV (old) vs deep embeddings vs new colour vs
colour+embeddings, then reruns the image->features->dosha pipeline. Subject-level CV.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2lab, rgb2ycbcr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")
RNG = 42
ROOT = Path(__file__).resolve().parents[2]
MANI = ROOT / "model_v2" / "results_full" / "manifest.csv"
FEAT = ROOT / "model_v2" / "avg" / "features" / "mobilenet_v2.npz"

GROUPS = {
    "SC": ["SC_Fair", "SC_Medium", "SC_Dark"],
    "PC": ["PC_PinkRed", "PC_PinkWhite", "PC_Brown"],
}


def white_balance(arr):
    """Neutralise colour cast using bright (assumed light-background) pixels."""
    lum = arr.mean(2)
    thr = np.percentile(lum, 95)
    bright = arr[lum >= thr]
    if len(bright) < 50:
        return arr
    wp = np.clip(bright.mean(0), 1, None)          # estimated white point
    return np.clip(arr * (wp.mean() / wp), 0, 255)


def skin_mask(arr):
    """YCbCr skin threshold, keep largest connected blob, drop background-bright pixels."""
    y = rgb2ycbcr(arr.astype(np.uint8))
    Cb, Cr = y[..., 1], y[..., 2]
    m = (Cb >= 77) & (Cb <= 127) & (Cr >= 133) & (Cr <= 173)
    m &= arr.mean(2) < np.percentile(arr.mean(2), 92)   # exclude blown-out background
    if m.sum() < 300:
        return np.ones(arr.shape[:2], bool)             # fallback: whole image
    lbl, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        m = lbl == (np.argmax(sizes) + 1)
    return m


def color_features(path):
    if not isinstance(path, str) or not Path(path).exists():
        return None
    arr = np.asarray(Image.open(path).convert("RGB").resize((160, 200))).astype(float)
    wb = white_balance(arr)
    m = skin_mask(wb)
    px = wb[m] / 255.0                        # skin pixels, [0,1] RGB, white-balanced
    if len(px) < 50:
        px = wb.reshape(-1, 3) / 255.0
    rgb_mean = px.mean(0)
    lab = rgb2lab(px.reshape(-1, 1, 3)).reshape(-1, 3).mean(0)          # L, a, b
    hsv = np.asarray(Image.fromarray((px.reshape(1, -1, 3) * 255).astype(np.uint8))
                     .convert("HSV")).reshape(-1, 3).mean(0) / 255.0    # H, S, V
    s = px.sum(1) + 1e-6
    chroma = np.stack([px[:, 0] / s, px[:, 1] / s, px[:, 2] / s], 1).mean(0)  # r,g,b norm
    redness = (px[:, 0] - px[:, 1]).mean()
    return np.concatenate([rgb_mean, lab, hsv, chroma, [redness]])       # 13 features


def build_color_matrix(df):
    feats_front, feats_back = [], []
    for _, r in df.iterrows():
        ff = color_features(r.get("img_open_front"))
        fb = color_features(r.get("img_open_back"))
        if ff is None:
            ff = fb
        if fb is None:
            fb = ff
        feats_front.append(ff)
        feats_back.append(fb)
    return np.array(feats_front), np.array(feats_back)


def cat(df, cols):
    M = df[cols].to_numpy()
    return np.where(M.sum(1) == 0, len(cols), M.argmax(1))


def rf():
    return RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                  random_state=RNG)


def acc(X, y):
    return accuracy_score(y, cross_val_predict(rf(), X, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=RNG)))


def main():
    df = pd.read_csv(MANI).reset_index(drop=True)
    z = np.load(FEAT, allow_pickle=True)
    emb, subj = z["emb"], z["subject_id"].astype(str)
    Ximg = np.stack([emb[subj == s].mean(0) for s in df["id"].astype(str)])

    print("Extracting palm/skin colour (white-balance + skin-segment)...")
    Cfront, Cback = build_color_matrix(df)
    Ccol = np.hstack([Cfront, Cback])
    Cboth = np.hstack([Ccol, Ximg])

    print("\nRecovery of the colour features (subject-level 5-fold):")
    print(f"{'feature':<6}{'majority':>10}{'embeddings':>12}{'NEW colour':>12}{'colour+emb':>12}")
    for g, cols in GROUPS.items():
        y = cat(df, cols)
        maj = np.bincount(y).max() / len(y)
        best_src = Cfront if g == "PC" else Cback   # PC<-palm, SC<-back-of-hand
        print(f"{g:<6}{maj:>10.1%}{acc(Ximg, y):>12.1%}{acc(best_src, y):>12.1%}"
              f"{acc(np.hstack([best_src, Ximg]), y):>12.1%}")

    # pipeline with improved colour for PC & SC (+ embeddings for the rest, +true Age/Sex)
    ydom = df["ydom"].to_numpy()
    OTHERS = {"ST": ["ST_Rough", "ST_Smooth", "ST_Dry", "ST_Oily"],
              "SH": ["SH_Thick", "SH_Thin", "SH_Med"], "SHC": ["SHC_Brown", "SHC_Black"],
              "VV": ["VV_Low", "VV_Med", "VV_High"]}
    cvk = StratifiedKFold(5, shuffle=True, random_state=RNG)

    def onehot(pred, k):
        M = np.zeros((len(pred), k))
        for i, c in enumerate(pred):
            if c < k:
                M[i, c] = 1
        return M

    blocks = [df[["Age", "Sex_M"]].to_numpy(float)]
    # colour concepts from proper colour features
    blocks.append(onehot(cross_val_predict(rf(), np.hstack([Cback, Ximg]), cat(df, GROUPS["SC"]), cv=cvk), 3))
    blocks.append(onehot(cross_val_predict(rf(), np.hstack([Cfront, Ximg]), cat(df, GROUPS["PC"]), cv=cvk), 3))
    blocks.append(cross_val_predict(rf(), Ximg, df["Mole"].to_numpy(), cv=cvk).reshape(-1, 1))
    for g, cols in OTHERS.items():
        blocks.append(onehot(cross_val_predict(rf(), Ximg, cat(df, cols), cv=cvk), len(cols)))
    Xpred = np.hstack(blocks)

    TAB = ["Age", "Sex_M"] + GROUPS["SC"] + GROUPS["PC"] + ["Mole"] + [c for cs in OTHERS.values() for c in cs]
    Xtrue = df[TAB].to_numpy(float)

    def dosha(X):
        p = cross_val_predict(rf(), X, ydom, cv=cvk)
        return accuracy_score(ydom, p), balanced_accuracy_score(ydom, p)

    print("\n3-class dominant dosha pipeline (majority 64.1%):")
    print(f"  CEILING true features                 -> {dosha(Xtrue)[0]:.1%} acc / {dosha(Xtrue)[1]:.1%} bal")
    print(f"  PIPELINE (NEW colour for PC/SC)        -> {dosha(Xpred)[0]:.1%} acc / {dosha(Xpred)[1]:.1%} bal")
    print(f"  FLOOR image embeddings -> dosha        -> {dosha(Ximg)[0]:.1%} acc / {dosha(Ximg)[1]:.1%} bal")


if __name__ == "__main__":
    main()
