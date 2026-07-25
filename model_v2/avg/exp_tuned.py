"""Rigorous, nested-CV hyperparameter tuning for the deployed colour model.

- Features: patch colour (skin=back centre, palm=front above-centre) + age/sex.
- NESTED CV: inner RandomizedSearchCV (tunes each family) wrapped in an outer
  RepeatedStratifiedKFold -> UNBIASED accuracy/balanced-accuracy (no selection leak).
- Families: current default RF (reference), tuned RF, ExtraTrees, HistGradientBoosting,
  SVM-RBF, LogReg. Inner scoring = balanced_accuracy (honest under 64% Kapha imbalance).
- Baselines: majority + stratified dummy. Permutation test on the winner.
- Refit the winner on all data, out-of-fold confusion, and redeploy to
  mini/model_color.joblib if it beats the current default.
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict, cross_validate,
                                     permutation_test_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
RNG = 42
ROOT = Path(__file__).resolve().parents[2]
MINI = ROOT / "mini"
sys.path.insert(0, str(MINI))
from palm_color import patch_color, SKIN_CY, PALM_CY, FEATURE_NAMES  # noqa: E402

MANI = ROOT / "model_v2" / "results_full" / "manifest.csv"


def load_patch(path, cy):
    if not isinstance(path, str) or not Path(path).exists():
        return None
    return patch_color(Image.open(path), cy)


def fill(arrs):
    good = np.array([a for a in arrs if a is not None]); m = good.mean(0)
    return np.array([a if a is not None else m for a in arrs])


def build_xy():
    df = pd.read_csv(MANI).reset_index(drop=True)
    SC = fill([load_patch(p, SKIN_CY) for p in df["img_open_back"]])
    PC = fill([load_patch(p, PALM_CY) for p in df["img_open_front"]])
    X = np.hstack([SC, PC, df[["Age", "Sex_M"]].to_numpy(float)])
    return X, df["ydom"].to_numpy()


def families():
    rf_space = {"n_estimators": [200, 400, 600, 800], "max_depth": [None, 3, 5, 8, 12],
                "min_samples_leaf": [1, 2, 4], "min_samples_split": [2, 5, 10],
                "max_features": ["sqrt", "log2", 0.5],
                "class_weight": ["balanced", "balanced_subsample"]}
    return {
        "RF-default(current)": (RandomForestClassifier(n_estimators=600,
            class_weight="balanced_subsample", random_state=RNG), None),
        "RF-tuned": (RandomForestClassifier(random_state=RNG), rf_space),
        "ExtraTrees": (ExtraTreesClassifier(random_state=RNG), rf_space),
        "HistGB": (HistGradientBoostingClassifier(random_state=RNG),
                   {"learning_rate": [0.03, 0.05, 0.1], "max_depth": [None, 3, 5],
                    "max_iter": [200, 400], "min_samples_leaf": [5, 10, 20],
                    "l2_regularization": [0.0, 1.0], "class_weight": ["balanced", None]}),
        "SVM-RBF": (Pipeline([("sc", StandardScaler()), ("c", SVC(random_state=RNG))]),
                    {"c__C": [0.5, 1, 2, 5, 10], "c__gamma": ["scale", "auto", 0.01, 0.1],
                     "c__class_weight": ["balanced", None]}),
        "LogReg": (Pipeline([("sc", StandardScaler()),
                             ("c", LogisticRegression(max_iter=3000, random_state=RNG))]),
                   {"c__C": [0.01, 0.1, 1, 10], "c__class_weight": ["balanced", None]}),
    }


def main():
    X, y = build_xy()
    maj = pd.Series(y).value_counts().max() / len(y)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RNG)
    inner = StratifiedKFold(4, shuffle=True, random_state=RNG)
    print(f"n={len(y)} subjects, 3-class dominant dosha. majority={maj:.1%}, random=33.3%")
    print(f"NESTED CV (outer 5x3, inner RandomizedSearchCV cv=4 -> balanced_accuracy)\n")

    # baselines
    for nm, strat in [("majority", "most_frequent"), ("stratified-random", "stratified")]:
        r = cross_validate(DummyClassifier(strategy=strat, random_state=RNG), X, y, cv=outer,
                           scoring=["accuracy", "balanced_accuracy"])
        print(f"  baseline {nm:<18} acc={r['test_accuracy'].mean():.1%}  "
              f"bal={r['test_balanced_accuracy'].mean():.1%}")

    print(f"\n{'family':<22}{'nested acc':>12}{'nested bal':>12}{'bal std':>9}")
    results = {}
    for name, (est, space) in families().items():
        model = est if space is None else RandomizedSearchCV(
            est, space, n_iter=25, cv=inner, scoring="balanced_accuracy",
            random_state=RNG, n_jobs=-1)
        r = cross_validate(model, X, y, cv=outer, scoring=["accuracy", "balanced_accuracy"], n_jobs=1)
        a, b, bs = r["test_accuracy"].mean(), r["test_balanced_accuracy"].mean(), r["test_balanced_accuracy"].std()
        results[name] = (a, b)
        print(f"{name:<22}{a:>11.1%}{b:>12.1%}{bs:>9.1%}")

    winner = max(results, key=lambda k: results[k][1])
    cur = results["RF-default(current)"]
    print(f"\nWinner by nested balanced-acc: {winner}  "
          f"(acc {results[winner][0]:.1%} / bal {results[winner][1]:.1%})")
    print(f"Current deployed (RF-default): acc {cur[0]:.1%} / bal {cur[1]:.1%}")

    # --- refit winner on ALL data (full inner search for its hyperparams) ---
    est, space = families()[winner]
    if space is None:
        best = est
        best_params = "fixed defaults"
    else:
        search = RandomizedSearchCV(est, space, n_iter=40, cv=inner,
                                    scoring="balanced_accuracy", random_state=RNG, n_jobs=-1)
        search.fit(X, y)
        best_params = search.best_params_
        best = search.best_estimator_
        # SVC needs probability=True for the app's predict_proba
        if winner == "SVM-RBF":
            p = {k.replace("c__", ""): v for k, v in best_params.items()}
            best = Pipeline([("sc", StandardScaler()),
                             ("c", SVC(probability=True, random_state=RNG, **p))]).fit(X, y)
    print(f"Winner best params: {best_params}")

    # permutation test (significance) on the winner
    score, _, pval = permutation_test_score(
        best, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=RNG),
        scoring="balanced_accuracy", n_permutations=200, n_jobs=-1, random_state=RNG)
    print(f"Permutation test: balanced-acc {score:.1%}, p={pval:.3f} "
          f"({'significant' if pval < 0.05 else 'NOT significant'})")

    # out-of-fold confusion
    yp = cross_val_predict(best, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=RNG))
    lab = sorted(pd.unique(y))
    cm = confusion_matrix(y, yp, labels=lab)
    print("  confusion (rows=truth, cols=pred):  " + " ".join(lab))
    for i, l in enumerate(lab):
        print(f"    {l}: " + " ".join(f"{v:3d}" for v in cm[i]))

    # --- redeploy if winner >= current on balanced-acc ---
    if results[winner][1] >= cur[1] - 1e-9:
        best.fit(X, y)
        bundle = {
            "model": best, "classes": list(best.classes_), "feature_names": FEATURE_NAMES,
            "target": "dominant_dosha", "skin_view": "palm_open_back", "palm_view": "palm_open_front",
            "skin_cy": SKIN_CY, "palm_cy": PALM_CY,
            "tuning": {"winner": winner, "best_params": str(best_params),
                       "nested_acc": results[winner][0], "nested_bal": results[winner][1],
                       "perm_pvalue": pval},
            "notes": "patch colour + age/sex -> tuned classifier (nested-CV selected). dominant dosha K/P/V.",
        }
        joblib.dump(bundle, MINI / "model_color.joblib")
        print(f"\nREDEPLOYED -> mini/model_color.joblib  (winner={winner})")
    else:
        print("\nKept current model (winner did not beat it).")


if __name__ == "__main__":
    main()
