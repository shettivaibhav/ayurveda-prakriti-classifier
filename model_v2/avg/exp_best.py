"""Best fully-automated 3-class dominant-dosha model.

Feature stack: mean-pooled frozen embeddings + patch colour (skin=back centre,
palm=front just-above-centre) + age/sex. Scored vs the sheet (dominant dosha),
subject-level 5x5 CV. Reports what each ingredient adds, for MobileNetV2 and for
the all-5-backbone ensemble.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab
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


def patch_color(path, cy_frac, half=0.16):
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


def meanpool(name, ids):
    z = np.load(FDIR / f"{name}.npz", allow_pickle=True)
    emb, subj = z["emb"], z["subject_id"].astype(str)
    return np.stack([emb[subj == s].mean(0) for s in ids])


def col(df, view, cyf):
    out = [patch_color(p, cyf) for p in df[view]]
    good = np.array([o for o in out if o is not None]); mean = good.mean(0)
    return np.array([o if o is not None else mean for o in out])


def best_acc(X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
    npca = min(40, X.shape[1] - 1)   # adaptive: colour-only stacks have few features
    cand = {
        "RF": RandomForestClassifier(n_estimators=600, class_weight="balanced_subsample", random_state=RNG),
        "SVM": Pipeline([("sc", StandardScaler()), ("pca", PCA(npca, random_state=RNG)),
                         ("c", SVC(kernel="rbf", C=2.0, class_weight="balanced", random_state=RNG))]),
        "LogReg": Pipeline([("sc", StandardScaler()), ("pca", PCA(npca, random_state=RNG)),
                            ("c", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RNG))]),
    }
    best = (None, -1, -1)
    for n, m in cand.items():
        r = cross_validate(m, X, y, cv=cv, scoring=["accuracy", "balanced_accuracy"], n_jobs=-1)
        a, b = r["test_accuracy"].mean(), r["test_balanced_accuracy"].mean()
        if a > best[1]:
            best = (n, a, b)
    return best


def main():
    df = pd.read_csv(MANI).reset_index(drop=True)
    ids = df["id"].astype(str).tolist()
    y = df["ydom"].to_numpy()
    agesex = df[["Age", "Sex_M"]].to_numpy(float)
    SCp = col(df, "img_open_back", 0.50)    # skin colour: back centre
    PCp = col(df, "img_open_front", 0.40)   # palm colour: front just-above-centre
    patch = np.hstack([SCp, PCp])
    maj = pd.Series(y).value_counts().max() / len(y)

    mob = meanpool("mobilenet_v2", ids)
    allb = np.hstack([meanpool(b, ids) for b in BACKBONES])

    combos = [
        ("patch colour + age/sex", np.hstack([patch, agesex])),
        ("MobileNet only", mob),
        ("MobileNet + age/sex", np.hstack([mob, agesex])),
        ("MobileNet + patch + age/sex", np.hstack([mob, patch, agesex])),
        ("ALL-backbones + patch + age/sex", np.hstack([allb, patch, agesex])),
    ]
    print(f"3-class dominant dosha vs sheet | subject-level 5x5 CV | majority = {maj:.1%}")
    print(f"{'feature stack':<34}{'accuracy':>10}{'balanced':>10}{'clf':>8}")
    for name, X in combos:
        clf, a, b = best_acc(X, y)
        print(f"{name:<34}{a:>9.1%}{b:>10.1%}{clf:>8}")


if __name__ == "__main__":
    main()
