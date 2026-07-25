"""Tabular "diagnosis-reconstruction" baseline - NOT the deployed model.

Trains RandomForest / LogisticRegression on the practitioner's own manual
visual codes (skin color/texture, vein visibility, palm color, mole, age,
sex) to predict Prakriti, using Leave-One-Out CV (cheap at n~80-92).

This measures how well the label can be reconstructed from the practitioner's
own notes -- it is NOT an achievable ceiling for the photo-only image model.
The image model's task (raw pixels -> label) is strictly harder than this
task (already-extracted symbolic codes -> label). This baseline exists for
the paper's discussion/interpretability section only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder

from data import LABEL_COL, TABULAR_FEATURE_COLS, load_labels

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list, LabelEncoder]:
    feature_cols = TABULAR_FEATURE_COLS + ["Age"]
    sex_dummies = pd.get_dummies(df["Sex"], prefix="Sex")
    X_df = pd.concat([df[feature_cols], sex_dummies], axis=1)
    # One-hot indicator columns: a missing value means "not recorded", treat as 0.
    X_df[TABULAR_FEATURE_COLS] = X_df[TABULAR_FEATURE_COLS].fillna(0)
    feature_names = list(X_df.columns)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[LABEL_COL])

    return X_df.to_numpy(dtype=float), y, feature_names, label_encoder


def loo_cv_accuracy(model_fn, X: np.ndarray, y: np.ndarray) -> dict:
    loo = LeaveOneOut()
    correct = 0
    preds = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])[0]
        preds[test_idx[0]] = pred
        correct += int(pred == y[test_idx[0]])

    accuracy = correct / len(y)
    return {"accuracy": accuracy, "n": len(y), "predictions": preds.tolist()}


def run_lime_explanation(X: np.ndarray, y: np.ndarray, feature_names: list, class_names: list):
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(X, y)

    explainer = LimeTabularExplainer(
        X,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
    )

    instance = X[0]
    exp = explainer.explain_instance(instance, model.predict_proba, num_features=min(10, len(feature_names)))

    out_html = RESULTS_DIR / "tabular_lime_explanation.html"
    exp.save_to_file(str(out_html))

    return {
        "explained_instance_index": 0,
        "top_features": exp.as_list(),
        "html_path": str(out_html),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Uses all 92 labeled subjects, not just the 80 with photos - this
    # baseline only needs the xlsx's tabular features, never touches images.
    df = load_labels()

    X, y, feature_names, label_encoder = build_feature_matrix(df)
    class_names = list(label_encoder.classes_)

    print(f"Subjects: {len(df)} | Features: {len(feature_names)} | Classes: {class_names}")
    print("\nNOTE: this baseline measures diagnosis-reconstruction from the")
    print("practitioner's own manual codes, not a ceiling for the photo model.\n")

    results = {}

    print("Running LOO-CV: RandomForest...")
    rf_result = loo_cv_accuracy(
        lambda: RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        X, y,
    )
    print(f"  RandomForest LOO accuracy: {rf_result['accuracy']:.4f} (n={rf_result['n']})")
    results["random_forest"] = {"accuracy": rf_result["accuracy"], "n": rf_result["n"]}

    print("Running LOO-CV: LogisticRegression...")
    lr_result = loo_cv_accuracy(
        lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
        X, y,
    )
    print(f"  LogisticRegression LOO accuracy: {lr_result['accuracy']:.4f} (n={lr_result['n']})")
    results["logistic_regression"] = {"accuracy": lr_result["accuracy"], "n": lr_result["n"]}

    print("\nGenerating LIME explanation...")
    lime_result = run_lime_explanation(X, y, feature_names, class_names)
    results["lime"] = lime_result
    print(f"  Saved LIME explanation to {lime_result['html_path']}")
    print("  Top features for explained instance:")
    for feat, weight in lime_result["top_features"]:
        print(f"    {feat}: {weight:.4f}")

    out_path = RESULTS_DIR / "tabular_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
