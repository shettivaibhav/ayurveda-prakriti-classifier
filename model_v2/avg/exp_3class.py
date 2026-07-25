"""Can we hit ~70%? Reframe to 3-class DOMINANT dosha (Kapha/Pitta/Vata) — the same
framing as the old deployed model that worked in practice — and compare:

    tabular  : 21 practitioner skin/palm features
    image    : 4-view averaged MobileNetV2 features (the approach we just built)
    fusion   : tabular + image

Subject-level 5x5 CV, accuracy + balanced-accuracy + macro-F1, vs random/majority.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
RNG = 42
ROOT = Path(__file__).resolve().parents[2]
MANI = ROOT / "model_v2" / "results_full" / "manifest.csv"
FEAT = ROOT / "model_v2" / "avg" / "features" / "mobilenet_v2.npz"

TAB = ["Age", "Sex_M", "Mole", "SC_Fair", "SC_Medium", "SC_Dark", "ST_Rough",
       "ST_Smooth", "ST_Dry", "ST_Oily", "SH_Thick", "SH_Thin", "SH_Med",
       "SHC_Brown", "SHC_Black", "VV_Low", "VV_Med", "VV_High",
       "PC_PinkRed", "PC_PinkWhite", "PC_Brown"]


def clfs(image_like):
    base = {
        "RandomForest": RandomForestClassifier(n_estimators=400,
                        class_weight="balanced_subsample", random_state=RNG),
        "LogReg": Pipeline([("sc", StandardScaler()),
                  ("c", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG))]),
        "SVM-RBF": Pipeline([("sc", StandardScaler()),
                   ("c", SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=RNG))]),
    }
    return base


def run(name, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
    print(f"\n### {name}  (dim={X.shape[1]}) ###")
    print(f"{'model':<14}{'accuracy':>10}{'balanced':>10}{'macroF1':>9}")
    best = None
    for mname, model in clfs(name != "tabular").items():
        r = cross_validate(model, X, y, cv=cv,
                           scoring=["accuracy", "balanced_accuracy", "f1_macro"], n_jobs=-1)
        acc, bal, f1 = r["test_accuracy"].mean(), r["test_balanced_accuracy"].mean(), r["test_f1_macro"].mean()
        print(f"{mname:<14}{acc:>9.1%}{bal:>10.1%}{f1:>9.1%}")
        if best is None or acc > best[1]:
            best = (mname, acc, bal)
    return best


def main():
    df = pd.read_csv(MANI)
    df["dom"] = df["ydom"]
    y = df["dom"].to_numpy()
    print("3-class DOMINANT dosha  |  92 subjects  |  subject-level 5x5 CV")
    vc = pd.Series(y).value_counts()
    print("distribution:", dict(vc))
    print(f"baselines: random = {1/3:.1%}   majority ({vc.idxmax()}) = {vc.max()/len(y):.1%}")

    # tabular
    Xtab = df[TAB].to_numpy(float)

    # image: mean-pooled MobileNetV2 embedding per subject, aligned to manifest order
    z = np.load(FEAT, allow_pickle=True)
    emb, subj = z["emb"], z["subject_id"].astype(str)
    Ximg = np.stack([emb[subj == s].mean(0) for s in df["id"].astype(str)])

    Xfus = np.hstack([Xtab, Ximg])

    b_tab = run("tabular", Xtab, y)
    b_img = run("image (4-view avg)", Ximg, y)
    b_fus = run("fusion (tab+image)", Xfus, y)

    print("\n" + "=" * 60)
    print("BEST ACCURACY per approach (3-class dominant dosha):")
    for nm, b in [("tabular", b_tab), ("image avg", b_img), ("fusion", b_fus)]:
        print(f"  {nm:<12} {b[1]:.1%} accuracy  ({b[2]:.1%} balanced)  [{b[0]}]")


if __name__ == "__main__":
    main()
