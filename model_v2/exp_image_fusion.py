"""Image-only and tabular+image fusion experiments.

Uses the cached frozen MobileNetV2 embeddings. Because the embeddings are high
dimensional (1280 single-view / 5120 multi-view) and n is tiny, every pipeline
does StandardScaler -> PCA inside the CV so there is no leakage. Compares:

    tabular            (21 practitioner features)         <- reference
    image_single       (open_back embedding)
    image_multi        (4-view embeddings concatenated)
    fusion             (tabular + 4-view embeddings)

across the 7-class and 3-class-dominant framings, same CV protocol as
exp_tabular.py, with random/majority baselines and a permutation test on the
best pipeline per (feature-set, framing).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     permutation_test_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
RNG = 42
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "model_v2" / "results_full"
VIEWS = ["open_front", "open_back", "closed_front", "closed_back"]
TAB = ["Age", "Sex_M", "Mole", "SC_Fair", "SC_Medium", "SC_Dark",
       "ST_Rough", "ST_Smooth", "ST_Dry", "ST_Oily", "SH_Thick", "SH_Thin",
       "SH_Med", "SHC_Brown", "SHC_Black", "VV_Low", "VV_Med", "VV_High",
       "PC_PinkRed", "PC_PinkWhite", "PC_Brown"]


def mean_impute(mat, mask):
    """Fill embeddings of missing views with the column mean of present ones."""
    mat = mat.copy()
    if mask.sum() < len(mask):
        col_mean = mat[mask].mean(axis=0)
        mat[~mask] = col_mean
    return mat


def pca_pipe(clf, n_pca):
    return Pipeline([("sc", StandardScaler()),
                     ("pca", PCA(n_components=n_pca, random_state=RNG)),
                     ("clf", clf)])


def model_suite(image_like, n_pca):
    """Same classifiers everywhere; PCA front-end only for image/fusion sets."""
    base = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample", random_state=RNG),
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG),
        "SVM-RBF": SVC(kernel="rbf", C=2.0, gamma="scale",
                       class_weight="balanced", random_state=RNG),
    }
    if not image_like:
        # tabular: scale for the linear/kernel models, raw for the forest
        return {
            "RandomForest": base["RandomForest"],
            "LogReg": Pipeline([("sc", StandardScaler()), ("clf", base["LogReg"])]),
            "SVM-RBF": Pipeline([("sc", StandardScaler()), ("clf", base["SVM-RBF"])]),
        }
    return {name: pca_pipe(clf, n_pca) for name, clf in base.items()}


def evaluate(feat_name, X, y_str, image_like, n_pca=40):
    le = LabelEncoder(); y = le.fit_transform(y_str); names = list(le.classes_)
    counts = pd.Series(y_str).value_counts()
    maj = counts.max() / len(y)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
    rows, best = {}, (None, -1, None)
    for mname, model in model_suite(image_like, n_pca).items():
        r = cross_validate(model, X, y, cv=cv,
                           scoring=["accuracy", "balanced_accuracy", "f1_macro"],
                           n_jobs=-1)
        rows[mname] = {
            "acc": [r["test_accuracy"].mean(), r["test_accuracy"].std()],
            "bal_acc": [r["test_balanced_accuracy"].mean(), r["test_balanced_accuracy"].std()],
            "f1_macro": [r["test_f1_macro"].mean(), r["test_f1_macro"].std()],
        }
        if rows[mname]["bal_acc"][0] > best[1]:
            best = (mname, rows[mname]["bal_acc"][0], model)
    # permutation test on best pipeline
    skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
    score, _, pval = permutation_test_score(
        best[2], X, y, cv=skf, scoring="balanced_accuracy",
        n_permutations=100, n_jobs=-1, random_state=RNG)
    yp = cross_val_predict(best[2], X, y, cv=skf, n_jobs=-1)
    cm = confusion_matrix(y, yp, labels=list(range(len(names)))).tolist()
    return {
        "feature_set": feat_name, "dim": X.shape[1],
        "baselines": {"random": 1/len(names), "majority": maj},
        "models": rows, "best_model": best[0],
        "permutation": {"score": score, "pvalue": pval},
        "confusion_labels": names, "confusion": cm,
    }


def main():
    df = pd.read_csv(OUT / "manifest.csv")
    z = np.load(OUT / "embeddings.npz", allow_pickle=True)
    assert list(z["ids"]) == list(df["id"]), "embedding/manifest id mismatch"

    emb = {v: mean_impute(z[f"emb_{v}"], z[f"mask_{v}"]) for v in VIEWS}
    X_tab = df[TAB].values.astype(float)
    X_single = emb["open_back"]
    X_multi = np.hstack([emb[v] for v in VIEWS])
    X_fusion = np.hstack([X_tab, X_multi])

    feature_sets = [
        ("tabular", X_tab, False),
        ("image_single", X_single, True),
        ("image_multi", X_multi, True),
        ("fusion", X_fusion, True),
    ]
    framings = [("7class", df["y7"].values), ("3class_dominant", df["ydom"].values)]

    results = {}
    for fr_name, y in framings:
        print(f"\n{'#'*72}\n# {fr_name}\n{'#'*72}")
        print(f"{'feature_set':<14}{'dim':>6}{'best':>14}{'acc':>10}{'bal_acc':>10}"
              f"{'f1_macro':>10}{'perm_p':>9}")
        results[fr_name] = {}
        for feat_name, X, image_like in feature_sets:
            res = evaluate(feat_name, X, y, image_like)
            results[fr_name][feat_name] = res
            b = res["models"][res["best_model"]]
            print(f"{feat_name:<14}{res['dim']:>6}{res['best_model']:>14}"
                  f"{b['acc'][0]:>10.3f}{b['bal_acc'][0]:>10.3f}{b['f1_macro'][0]:>10.3f}"
                  f"{res['permutation']['pvalue']:>9.3f}")
            print(f"   baselines: random={res['baselines']['random']:.3f} "
                  f"majority={res['baselines']['majority']:.3f}")

    (OUT / "image_fusion_results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved -> {OUT/'image_fusion_results.json'}")


if __name__ == "__main__":
    main()
