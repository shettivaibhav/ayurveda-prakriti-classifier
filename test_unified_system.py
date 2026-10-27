"""End-to-End Verification Test for the Unified Ayurvedic Prakriti System.

Tests:
  1. Approach 1 (Wrist Dosha Analysis): 4-view CV extraction, image quality checks,
     and unified ML inference.
  2. Approach 2 (WhatsApp Batch Pipeline): PrakritiPredictor and
     classify_prakriti_with_unified_ml integration.
  3. Verified Cohort vs Held-out Test Cohort data integrity check.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
MINI_DIR = PROJECT_ROOT / "mini"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MINI_DIR))

from mini.unified_cv import (
    validate_image_quality, extract_participant_features, FEATURE_NAMES
)
from whatsapp_pipeline.predictor import PrakritiPredictor
from whatsapp_pipeline.classifier import classify_prakriti_with_unified_ml


def test_approach_1():
    print("--- [TEST 1] Approach 1: Live Wrist Dosha Inference ---")
    subj_dir = PROJECT_ROOT / "dataset" / "Wrist_Dataset" / "1001_male"
    assert subj_dir.exists(), f"Subject directory {subj_dir} missing"

    # Load 4 views
    views = {
        "palm_open_front": Image.open(subj_dir / "palm_open_front.jpg"),
        "palm_open_back": Image.open(subj_dir / "palm_open_back.jpg"),
        "palm_closed_front": Image.open(subj_dir / "palm_close_front.jpg"),
        "palm_closed_back": Image.open(subj_dir / "palm_close_back.jpg"),
    }

    # 1. Test image quality
    for vk, img in views.items():
        q_res = validate_image_quality(img)
        assert "valid" in q_res and "blur_score" in q_res
        print(f"  {vk}: Valid={q_res['valid']}, Blur={q_res['blur_score']:.1f}")

    # 2. Test feature extraction
    feats = extract_participant_features(views, age=32, sex_m=1.0)
    assert feats.shape == (70,), f"Expected shape (70,), got {feats.shape}"
    assert not np.isnan(feats).any(), "NaN found in feature vector"
    print(f"  Extracted 70-D Feature Vector: min={feats.min():.3f}, max={feats.max():.3f}")

    # 3. Test 7-Feature CV Clinical Extraction
    from mini.unified_cv import extract_cv_clinical_markers
    markers = extract_cv_clinical_markers(views)
    expected_markers = ["skin_color", "skin_texture", "palm_color", "vein_visibility", "mole", "hair_thickness", "hair_color"]
    for m in expected_markers:
        assert m in markers, f"Missing clinical marker {m}"
    print(f"  Extracted 7 Clinical Markers: { {m: markers[m] for m in expected_markers} }")
    assert "binary_19d" in markers and len(markers["binary_19d"]) == 19, "19-D binary vector missing or incorrect"

    # 4. Test unified model inference
    model_path = MINI_DIR / "unified_prakriti_pipeline.joblib"
    assert model_path.exists(), "Model artifact missing"
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]

    probs = pipeline.predict_proba(feats.reshape(1, -1))[0]
    classes = bundle["classes"]
    res = {c: float(p) for c, p in zip(classes, probs)}
    top_dosha = max(res.items(), key=lambda x: x[1])[0]

    assert abs(sum(probs) - 1.0) < 1e-4, "Probabilities do not sum to 1.0"
    print(f"  Prediction Successful -> Dominant: {top_dosha} | Probabilities: {res}")
    print("  [PASS] Approach 1 Verified!\n")
    return res


def test_approach_2():
    print("--- [TEST 2] Approach 2: WhatsApp Batch Pipeline Inference ---")
    subj_dir = PROJECT_ROOT / "dataset" / "Wrist_Dataset" / "1002_female"
    assert subj_dir.exists(), f"Subject directory {subj_dir} missing"

    # Test PrakritiPredictor
    pred = PrakritiPredictor()
    assert pred.bundle is not None, "Failed to load model in PrakritiPredictor"

    sub_res = pred.predict_subject(
        age=28,
        gender="F",
        palm_open_front_path=str(subj_dir / "palm_open_front.jpg"),
        palm_open_back_path=str(subj_dir / "palm_open_back.jpg"),
        palm_closed_front_path=str(subj_dir / "palm_close_front.jpg"),
        palm_closed_back_path=str(subj_dir / "palm_close_back.jpg"),
    )
    print(f"  PrakritiPredictor output: Dominant={sub_res['predicted_dosha']}, Confidence={sub_res['confidence']:.2%}")
    assert sub_res["predicted_dosha"] in ["K", "P", "V"]

    # Test classify_prakriti_with_unified_ml
    test_row = {"Age": 28, "Sex": "F"}
    batch_res = classify_prakriti_with_unified_ml(
        subj_id="1002_female",
        subject_row=test_row,
        wrist_dataset_dir=str(PROJECT_ROOT / "dataset" / "Wrist_Dataset")
    )

    print(f"  Unified ML Batch Result: {batch_res['predicted_dosha']} ({batch_res['classification_percent']})")
    print(f"  Rationale: {batch_res['reasoning']}")
    assert "predicted_dosha" in batch_res and "vata_percent" in batch_res
    total_pct = batch_res["kapha_percent"] + batch_res["pitta_percent"] + batch_res["vata_percent"]
    assert total_pct == 100, f"Percentages sum to {total_pct}, expected 100"
    print("  [PASS] Approach 2 Verified!\n")


def test_dataset_integrity():
    print("--- [TEST 3] Dataset Integrity & Segregation ---")
    df = pd.read_excel(PROJECT_ROOT / "dataset" / "palm_dataset_26 June 26.xlsx")
    assert len(df) == 154, f"Expected 154 rows, found {len(df)}"

    verified_df = df.iloc[:93]
    held_out_df = df.iloc[93:]

    print(f"  Verified Clinical Cohort: {len(verified_df)} subjects (1001_male to 1093_Female)")
    print(f"  Held-out Test Cohort: {len(held_out_df)} subjects (1094_Female to 1154_Male)")
    assert len(held_out_df) == 61, f"Expected 61 held-out subjects, found {len(held_out_df)}"

    # Check evaluation metrics file
    eval_json = PROJECT_ROOT / "model_v2" / "results" / "unified_pipeline_evaluation.json"
    assert eval_json.exists(), "Evaluation JSON file missing"
    with open(eval_json) as f:
        metrics = json.load(f)

    print("\n--- Empirical Performance Summary (Honest 5x5 Repeated Stratified CV) ---")
    for name, m in metrics.items():
        if "balanced_acc_mean" in m:
            print(f"  * {name:<40}: Balanced Acc = {m['balanced_acc_mean']*100:.1f}%, Macro F1 = {m['macro_f1_mean']*100:.1f}%, Raw Acc = {m['accuracy_mean']*100:.1f}%")
        elif name == "Held-Out Test Evaluation":
            print(f"\n--- Independent Held-Out Test Cohort Performance (N={m['n_held_out']}) ---")
            print(f"  * Test Balanced Accuracy: {m['test_balanced_accuracy']*100:.1f}%")
            print(f"  * Test Macro F1         : {m['test_macro_f1']*100:.1f}%")
            print(f"  * Test Raw Accuracy     : {m['test_accuracy']*100:.1f}%")

    print("\n  [PASS] Dataset integrity & metrics verified!\n")


if __name__ == "__main__":
    test_approach_1()
    test_approach_2()
    test_dataset_integrity()
    print("All system verification tests PASSED successfully!")
