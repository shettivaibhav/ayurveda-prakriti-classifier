"""Train + save the deployable colour model: patch colour + age/sex -> RandomForest,
3-class dominant dosha. Saves mini/model_color.joblib. Uses the SAME feature
extraction as the app (mini/palm_color.py)."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate)

ROOT = Path(__file__).resolve().parents[2]
MINI = ROOT / "mini"
sys.path.insert(0, str(MINI))
from palm_color import patch_color, FEATURE_NAMES, SKIN_CY, PALM_CY  # noqa: E402

MANI = ROOT / "model_v2" / "results_full" / "manifest.csv"
OUT = MINI / "model_color.joblib"
RNG = 42


def load_patch(path, cy):
    if not isinstance(path, str) or not Path(path).exists():
        return None
    return patch_color(Image.open(path), cy)


def fill(arrs):
    good = np.array([a for a in arrs if a is not None])
    m = good.mean(0)
    return np.array([a if a is not None else m for a in arrs])


def main():
    df = pd.read_csv(MANI).reset_index(drop=True)
    SC = fill([load_patch(p, SKIN_CY) for p in df["img_open_back"]])
    PC = fill([load_patch(p, PALM_CY) for p in df["img_open_front"]])
    agesex = df[["Age", "Sex_M"]].to_numpy(float)
    X = np.hstack([SC, PC, agesex])
    y = df["ydom"].to_numpy()
    assert X.shape[1] == len(FEATURE_NAMES), (X.shape, len(FEATURE_NAMES))

    rf = RandomForestClassifier(n_estimators=600, class_weight="balanced_subsample",
                                random_state=RNG)

    # honest CV estimate (5x5 repeated for stable metrics; single fold for confusion)
    cvr = cross_validate(rf, X, y, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG),
                         scoring=["accuracy", "balanced_accuracy"], n_jobs=-1)
    print(f"CV vs sheet (3-class dominant, {len(y)} subjects):")
    print(f"  accuracy  : {cvr['test_accuracy'].mean():.1%}")
    print(f"  balanced  : {cvr['test_balanced_accuracy'].mean():.1%}   (majority "
          f"{pd.Series(y).value_counts().max()/len(y):.1%})")
    lab = sorted(pd.unique(y))
    yp = cross_val_predict(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=RNG), n_jobs=-1)
    cm = confusion_matrix(y, yp, labels=lab)
    print("  confusion (rows=truth, cols=pred):  " + " ".join(lab))
    for i, l in enumerate(lab):
        print(f"    {l}: " + " ".join(f"{v:3d}" for v in cm[i]))

    # final fit on all subjects + save
    rf.fit(X, y)
    bundle = {
        "model": rf,
        "classes": list(rf.classes_),                 # e.g. ['K','P','V']
        "feature_names": FEATURE_NAMES,
        "target": "dominant_dosha",
        "skin_view": "palm_open_back", "palm_view": "palm_open_front",
        "skin_cy": SKIN_CY, "palm_cy": PALM_CY,
        "notes": "patch colour (skin=back centre, palm=front above-centre) + age + sex "
                 "-> RandomForest. dominant dosha K/P/V.",
    }
    joblib.dump(bundle, OUT)
    print(f"\nSaved -> {OUT}  (classes={list(rf.classes_)})")


if __name__ == "__main__":
    main()
