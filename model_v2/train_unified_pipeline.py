"""Offline Model Training and Comparative Evaluation for the Unified Ayurvedic Prakriti Pipeline.

Evaluates:
  1. Class-weighted RBF-SVM (with StandardScaler)
  2. Class-weighted ExtraTrees
  3. Class-weighted Logistic Regression (with StandardScaler)
  4. Reference Baseline (2-view, 22-feature model)

Validation:
  - Repeated Stratified K-Fold (5 folds x 5 repeats = 25 iterations)
  - Participant-level validation with 0 data leakage
  - Target: 3-class dominant dosha (K: 60, P: 16, V: 17 on the 93 verified cohort)
  - Metrics: Balanced Accuracy, Macro F1, Accuracy, Out-of-Fold Confusion Matrix

Saves the winning pipeline bundle to:
  mini/unified_prakriti_pipeline.joblib
"""

import sys
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINI_DIR = PROJECT_ROOT / "mini"
sys.path.insert(0, str(MINI_DIR))

from unified_cv import (
    extract_participant_features, FEATURE_NAMES, VIEW_KEYS
)
from palm_color import (
    patch_color as baseline_patch_color,
    SKIN_CY as BASE_SKIN_CY,
    PALM_CY as BASE_PALM_CY,
    FEATURE_NAMES as BASE_FEATURE_NAMES
)

DATASET_XLSX = PROJECT_ROOT / "dataset" / "palm_dataset_26 June 26.xlsx"
WRIST_DIR = PROJECT_ROOT / "dataset" / "Wrist_Dataset"
OUT_MODEL_PATH = MINI_DIR / "unified_prakriti_pipeline.joblib"
RNG = 42


def find_subject_images(subj_id: str, wrist_dir: Path) -> dict:
    """Finds image paths for the 4 required views for a subject ID."""
    sid_clean = str(subj_id).strip().lower()
    subj_folder = wrist_dir / sid_clean
    if not subj_folder.exists():
        matches = [d for d in wrist_dir.iterdir() if d.is_dir() and d.name.lower() == sid_clean]
        if matches:
            subj_folder = matches[0]
        else:
            return {}

    filenames = {
        "palm_open_front": ["palm_open_front.jpg"],
        "palm_open_back": ["palm_open_back.jpg"],
        "palm_closed_front": ["palm_close_front.jpg", "palm_closed_front.jpg"],
        "palm_closed_back": ["palm_close_back.jpg", "palm_closed_back.jpg"]
    }

    images = {}
    for vk, candidates in filenames.items():
        found_path = None
        for c in candidates:
            p = subj_folder / c
            if p.exists():
                found_path = p
                break
        if found_path:
            try:
                images[vk] = Image.open(found_path)
            except Exception:
                images[vk] = None
        else:
            images[vk] = None
    return images


def load_verified_cohort(n_verified: int = 93):
    """Loads feature matrix X (70-D) and labels y for the verified cohort."""
    df = pd.read_excel(DATASET_XLSX)
    verified_df = df.iloc[:n_verified].copy()

    X_list = []
    X_baseline_list = []
    y_list = []
    subj_ids = []

    print(f"Extracting features for {n_verified} verified subjects from {WRIST_DIR}...", flush=True)
    for idx in range(n_verified):
        row = verified_df.iloc[idx]
        sid = str(row["id"]).strip()
        age = row["Age"]
        sex = row["Sex"]
        is_male = 1.0 if str(sex).strip().upper().startswith("M") else 0.0

        prakriti = str(row["Prakriti"]).strip()
        ydom = prakriti[0].upper()  # K, P, or V
        assert ydom in ["K", "P", "V"], f"Invalid dominant dosha {ydom} for {sid}"

        images = find_subject_images(sid, WRIST_DIR)

        # 1. Unified 70-D Feature Vector
        feat_vec = extract_participant_features(images, age=age, sex_m=is_male)
        X_list.append(feat_vec)

        # 2. Baseline 22-D Feature Vector
        sc_img = images.get("palm_open_back")
        pc_img = images.get("palm_open_front")
        sc_patch = baseline_patch_color(sc_img, BASE_SKIN_CY) if sc_img else np.zeros(10)
        pc_patch = baseline_patch_color(pc_img, BASE_PALM_CY) if pc_img else np.zeros(10)
        base_vec = np.concatenate([sc_patch, pc_patch, [float(age), is_male]])
        X_baseline_list.append(base_vec)

        y_list.append(ydom)
        subj_ids.append(sid)
        if (idx + 1) % 15 == 0 or idx == n_verified - 1:
            print(f"  Processed {idx + 1}/{n_verified} subjects...", flush=True)

    X = np.array(X_list, dtype=float)
    X_base = np.array(X_baseline_list, dtype=float)
    y = np.array(y_list)

    print(f"Extracted X shape: {X.shape}, Baseline shape: {X_base.shape}", flush=True)
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}", flush=True)
    return X, X_base, y, subj_ids


def evaluate_pipeline(name: str, pipeline, X: np.ndarray, y: np.ndarray, rskf: RepeatedStratifiedKFold):
    """Evaluates a model pipeline over repeated stratified CV folds."""
    accs, bal_accs, macro_f1s = [], [], []

    for train_idx, val_idx in rskf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_val)

        accs.append(accuracy_score(y_val, preds))
        bal_accs.append(balanced_accuracy_score(y_val, preds))
        macro_f1s.append(f1_score(y_val, preds, average="macro", zero_division=0))

    # Single-fold OOF predictions for confusion matrix
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    oof_preds = cross_val_predict(pipeline, X, y, cv=skf, n_jobs=-1)
    cm = confusion_matrix(y, oof_preds, labels=["K", "P", "V"])

    results = {
        "name": name,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "balanced_acc_mean": float(np.mean(bal_accs)),
        "balanced_acc_std": float(np.std(bal_accs)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "confusion_matrix": cm.tolist(),
        "classes": ["K", "P", "V"]
    }
    return results, oof_preds


def main():
    X, X_base, y, subj_ids = load_verified_cohort(n_verified=93)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)

    # Define Candidate Pipelines
    pipelines = {
        "Baseline (2-view ExtraTrees)": (
            ExtraTreesClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=RNG, n_jobs=-1),
            X_base
        ),
        "Unified RBF-SVM (Balanced)": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(C=2.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=RNG))
            ]),
            X
        ),
        "Unified ExtraTrees (Balanced)": (
            Pipeline([
                ("clf", ExtraTreesClassifier(n_estimators=400, max_depth=8, min_samples_leaf=2,
                                             class_weight="balanced_subsample", random_state=RNG, n_jobs=-1))
            ]),
            X
        ),
        "Unified Random Forest (Balanced)": (
            Pipeline([
                ("clf", RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=2,
                                               class_weight="balanced_subsample", random_state=RNG, n_jobs=-1))
            ]),
            X
        ),
        "Unified Logistic Regression (Balanced)": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=RNG))
            ]),
            X
        ),
    }

    print("\n" + "=" * 78)
    print(f"{'Model':<40} | {'Balanced Acc':<12} | {'Macro F1':<10} | {'Raw Acc':<10}")
    print("=" * 78)

    all_results = {}
    best_name = None
    best_score = -1.0
    best_pipe = None

    for name, (pipe, X_data) in pipelines.items():
        res, oof_preds = evaluate_pipeline(name, pipe, X_data, y, rskf)
        all_results[name] = res

        print(
            f"{name:<40} | "
            f"{res['balanced_acc_mean']*100:5.1f}% ±{res['balanced_acc_std']*100:3.1f}% | "
            f"{res['macro_f1_mean']*100:5.1f}% ±{res['macro_f1_std']*100:3.1f}% | "
            f"{res['accuracy_mean']*100:5.1f}%"
        )

        # We prioritize Balanced Accuracy on unified 70-D models
        if "Unified" in name and res["balanced_acc_mean"] > best_score:
            best_score = res["balanced_acc_mean"]
            best_name = name
            best_pipe = pipe

    print("=" * 78)
    print(f"\nWinning Unified Model: {best_name} (Balanced Acc: {best_score*100:.1f}%)")

    # Detailed report for winning model
    win_res = all_results[best_name]
    cm = np.array(win_res["confusion_matrix"])
    print("\nOut-of-Fold Confusion Matrix (Rows=Truth, Cols=Predicted: K, P, V):")
    print("      K   P   V")
    for i, cls in enumerate(["K", "P", "V"]):
        print(f"  {cls}: {cm[i, 0]:3d} {cm[i, 1]:3d} {cm[i, 2]:3d}")

    # Retrain winning model on all 93 verified participants
    print(f"\nFitting {best_name} on all 93 verified subjects...")
    best_pipe.fit(X, y)

    # -------------------------------------------------------------
    # Evaluate Winning Model on Independent Held-Out Test Cohort (N=61)
    # Strictly kept untouched until this final evaluation step
    # -------------------------------------------------------------
    print("\nEvaluating on Independent Held-Out Test Cohort (rows 93 to 154)...")
    df_all = pd.read_excel(DATASET_XLSX)
    held_out_df = df_all.iloc[93:].copy()
    X_test_list, y_test_list = [], []
    for _, row in held_out_df.iterrows():
        sid = str(row["id"]).strip()
        age = row["Age"]
        sex = row["Sex"]
        is_male = 1.0 if str(sex).strip().upper().startswith("M") else 0.0
        prakriti = str(row["Prakriti"]).strip()
        ydom = prakriti[0].upper()
        if ydom not in ["K", "P", "V"]:
            continue
        imgs = find_subject_images(sid, WRIST_DIR)
        feat_vec = extract_participant_features(imgs, age=age, sex_m=is_male)
        X_test_list.append(feat_vec)
        y_test_list.append(ydom)

    X_test = np.array(X_test_list, dtype=float)
    y_test = np.array(y_test_list)

    test_preds = best_pipe.predict(X_test)
    test_bal_acc = float(balanced_accuracy_score(y_test, test_preds))
    test_macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))
    test_acc = float(accuracy_score(y_test, test_preds))
    test_cm = confusion_matrix(y_test, test_preds, labels=["K", "P", "V"]).tolist()

    held_out_metrics = {
        "n_held_out": len(y_test),
        "test_balanced_accuracy": test_bal_acc,
        "test_macro_f1": test_macro_f1,
        "test_accuracy": test_acc,
        "test_confusion_matrix": test_cm,
        "test_class_distribution": pd.Series(y_test).value_counts().to_dict()
    }
    all_results["Held-Out Test Evaluation"] = held_out_metrics

    print("-" * 78)
    print("HELD-OUT TEST EVALUATION RESULTS:")
    print(f"  * Held-Out Subjects     : {len(y_test)}")
    print(f"  * Test Balanced Accuracy: {test_bal_acc*100:.1f}%")
    print(f"  * Test Macro F1         : {test_macro_f1*100:.1f}%")
    print(f"  * Test Raw Accuracy     : {test_acc*100:.1f}%")
    print("-" * 78)

    # Save artifact bundle
    bundle = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "classes": ["K", "P", "V"],
        "feature_names": FEATURE_NAMES,
        "view_keys": VIEW_KEYS,
        "n_features": len(FEATURE_NAMES),
        "cv_results": all_results,
        "winning_metrics": win_res,
        "held_out_metrics": held_out_metrics,
        "notes": "Trained on 93 verified subjects using 4-view anatomical ROIs + demographics."
    }

    joblib.dump(bundle, OUT_MODEL_PATH)
    print(f"Successfully saved deployment bundle to: {OUT_MODEL_PATH}")

    # Also save evaluation metrics to JSON for documentation
    eval_json_path = PROJECT_ROOT / "model_v2" / "results" / "unified_pipeline_evaluation.json"
    eval_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved evaluation metrics to: {eval_json_path}")


if __name__ == "__main__":
    main()
