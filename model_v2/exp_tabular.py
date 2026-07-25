"""Tabular ML experiments on the practitioner's 24 annotation features.

Compares RandomForest / XGBoost / LogisticRegression / SVM across three label
framings (7-class, 3-class dominant dosha, multi-label dosha presence) with
RepeatedStratifiedKFold CV. Reports accuracy, balanced accuracy and macro-F1
against random + majority baselines, plus a label-permutation test to check
whether each model is genuinely above chance on this tiny dataset.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     permutation_test_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RNG = 42
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "model_v2" / "results_full"

FEATURES = ["Age", "Sex_M", "Mole",
            "SC_Fair", "SC_Medium", "SC_Dark",
            "ST_Rough", "ST_Smooth", "ST_Dry", "ST_Oily",
            "SH_Thick", "SH_Thin", "SH_Med",
            "SHC_Brown", "SHC_Black",
            "VV_Low", "VV_Med", "VV_High",
            "PC_PinkRed", "PC_PinkWhite", "PC_Brown"]


def models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            min_samples_leaf=1, random_state=RNG),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            random_state=RNG, verbosity=0),
        "LogReg": Pipeline([("sc", StandardScaler()),
                            ("clf", LogisticRegression(
                                max_iter=2000, class_weight="balanced",
                                C=1.0, random_state=RNG))]),
        "SVM-RBF": Pipeline([("sc", StandardScaler()),
                             ("clf", SVC(kernel="rbf", C=2.0, gamma="scale",
                                         class_weight="balanced",
                                         probability=False, random_state=RNG))]),
    }


def cv_scores(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
    r = cross_validate(model, X, y, cv=cv,
                       scoring=["accuracy", "balanced_accuracy", "f1_macro"],
                       n_jobs=-1, error_score="raise")
    return {
        "acc": (r["test_accuracy"].mean(), r["test_accuracy"].std()),
        "bal_acc": (r["test_balanced_accuracy"].mean(), r["test_balanced_accuracy"].std()),
        "f1_macro": (r["test_f1_macro"].mean(), r["test_f1_macro"].std()),
    }


def run_multiclass(name, X, y_str):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_str)          # integer labels (xgboost needs these)
    class_names = list(le.classes_)
    print(f"\n{'='*72}\n{name}  (n={len(y)}, classes={class_names})\n{'='*72}")
    counts = pd.Series(y_str).value_counts()
    maj = counts.max() / len(y)
    print(f"baselines: random={1/len(set(y)):.3f}  majority={maj:.3f} "
          f"(always '{counts.idxmax()}')")
    print(f"{'model':<14}{'acc':>16}{'bal_acc':>16}{'f1_macro':>16}")
    results = {"baselines": {"random": 1/len(set(y)), "majority": maj},
               "models": {}}
    best = (None, -1)
    for mname, model in models().items():
        s = cv_scores(model, X, y)
        results["models"][mname] = {k: list(v) for k, v in s.items()}
        print(f"{mname:<14}"
              f"{s['acc'][0]:>8.3f}±{s['acc'][1]:.3f}"
              f"{s['bal_acc'][0]:>8.3f}±{s['bal_acc'][1]:.3f}"
              f"{s['f1_macro'][0]:>8.3f}±{s['f1_macro'][1]:.3f}")
        if s["bal_acc"][0] > best[1]:
            best = (mname, s["bal_acc"][0])
    # detail on best model: confusion matrix + permutation test
    bname = best[0]
    print(f"\n--- best by balanced-acc: {bname} ---")
    bmodel = models()[bname]
    skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
    yp = cross_val_predict(bmodel, X, y, cv=skf, n_jobs=-1)
    labels = list(range(len(class_names)))
    print("confusion matrix (rows=true, cols=pred), labels:", class_names)
    print(confusion_matrix(y, yp, labels=labels))
    print(classification_report(y, yp, labels=labels, target_names=class_names,
                                zero_division=0, digits=3))
    score, _, pval = permutation_test_score(
        bmodel, X, y, cv=skf, scoring="balanced_accuracy",
        n_permutations=200, n_jobs=-1, random_state=RNG)
    print(f"permutation test (balanced-acc): observed={score:.3f}  p={pval:.4f}")
    results["best_model"] = bname
    results["best_confusion_labels"] = class_names
    results["best_confusion"] = confusion_matrix(y, yp, labels=labels).tolist()
    results["permutation"] = {"score": score, "pvalue": pval}
    return results


def run_multilabel(X, df):
    print(f"\n{'='*72}\nMULTI-LABEL dosha presence (has_K / has_P / has_V)\n{'='*72}")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
    out = {}
    print(f"{'dosha':<8}{'prevalence':>12}{'model':>14}{'bal_acc':>12}{'f1':>10}{'roc_auc':>10}")
    for dosha in ("has_K", "has_P", "has_V"):
        y = df[dosha].values
        prev = y.mean()
        for mname in ("RandomForest", "XGBoost"):
            model = models()[mname]
            r = cross_validate(model, X, y, cv=cv,
                               scoring=["balanced_accuracy", "f1"], n_jobs=-1)
            # roc-auc via predict_proba
            proba = cross_val_predict(models()[mname], X, y,
                                      cv=StratifiedKFold(5, shuffle=True, random_state=RNG),
                                      method="predict_proba", n_jobs=-1)[:, 1]
            auc = roc_auc_score(y, proba)
            ba = r["test_balanced_accuracy"].mean()
            f1 = r["test_f1"].mean()
            print(f"{dosha:<8}{prev:>12.2f}{mname:>14}{ba:>12.3f}{f1:>10.3f}{auc:>10.3f}")
            out.setdefault(dosha, {})[mname] = {
                "prevalence": prev, "bal_acc": ba, "f1": f1, "roc_auc": auc}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT / "manifest.csv")
    X = df[FEATURES].values.astype(float)
    all_results = {"n": len(df), "features": FEATURES}
    all_results["7class"] = run_multiclass("7-CLASS PRAKRITI (y7)", X, df["y7"].values)
    all_results["3class_dominant"] = run_multiclass("3-CLASS DOMINANT DOSHA (ydom)", X, df["ydom"].values)
    all_results["multilabel"] = run_multilabel(X, df)
    (OUT / "tabular_results.json").write_text(json.dumps(all_results, indent=2, default=float))
    print(f"\nSaved -> {OUT/'tabular_results.json'}")


if __name__ == "__main__":
    main()
