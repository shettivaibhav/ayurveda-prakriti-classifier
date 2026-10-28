"""
Doctor Portal Backend Engine
Handles:
  1. Queue of pending participants awaiting clinical verification.
  2. Loading 4 high-res hand photos and pre-populating user-reported traits (vein, mole, skin texture) + CV estimates.
  3. Single-click 3-way synchronization (Excel, Local Folders, Supabase Cloud).
  4. Instant model retraining on all verified records with feature vector caching.
"""

import sys
import json
import io
import shutil
from pathlib import Path
from PIL import Image as PILImage
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.drawing.image import Image as OpenpyxlImage
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent if (BACKEND_DIR.parent / "dataset").exists() else BACKEND_DIR
WRIST_DIR = PROJECT_ROOT / "dataset" / "Wrist_Dataset"
MASTER_XLSX = (PROJECT_ROOT / "doctor_validation_prakriti_master.xlsx") if (PROJECT_ROOT / "doctor_validation_prakriti_master.xlsx").exists() else (BACKEND_DIR / "doctor_validation_prakriti_master.xlsx")
DATASET_MASTER_XLSX = PROJECT_ROOT / "dataset" / "doctor_validation_prakriti_master.xlsx"
FEATURE_CACHE_PATH = (PROJECT_ROOT / "dataset" / "verified_features_cache.joblib") if (PROJECT_ROOT / "dataset").exists() else (BACKEND_DIR / "verified_features_cache.joblib")
MODEL_PATH = (BACKEND_DIR / "unified_prakriti_pipeline.joblib") if (BACKEND_DIR / "unified_prakriti_pipeline.joblib").exists() else (PROJECT_ROOT / "mini" / "unified_prakriti_pipeline.joblib")

for path in [BACKEND_DIR, PROJECT_ROOT, PROJECT_ROOT / "mini", PROJECT_ROOT / "backend"]:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from unified_cv import extract_participant_features, FEATURE_NAMES, VIEW_KEYS
from supabasehelper import supabase_client


def get_queue_summary():
    """Returns lists of pending, verified, and duplicate participant IDs."""
    if not MASTER_XLSX.exists():
        return [], [], []

    wb = openpyxl.load_workbook(MASTER_XLSX, data_only=True)
    ws = wb.active
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    col_map = {name: c for c, name in enumerate(headers, 1) if name}

    pending = []
    verified = []
    duplicates = []

    for r in range(2, ws.max_row + 1):
        uid = ws.cell(r, col_map["Participant ID"]).value
        if not uid:
            continue
        uid = str(uid).strip()
        name = str(ws.cell(r, col_map.get("Participant Name", 2)).value or "").strip()
        status = str(ws.cell(r, col_map["Validation Status"]).value or "").strip().lower()

        item = {
            "row": r,
            "id": uid,
            "name": name,
            "status": ws.cell(r, col_map["Validation Status"]).value,
            "ml_dosha": ws.cell(r, col_map.get("ML Predicted Dosha", 19)).value,
            "doc_dosha": ws.cell(r, col_map.get("Doctor Verified Prakriti (Ground Truth)", 10)).value
        }

        if "duplicate" in status:
            duplicates.append(item)
        elif "verified" in status:
            verified.append(item)
        else:
            pending.append(item)

    return pending, verified, duplicates


def get_subject_data(subj_id: str):
    """
    Loads images, demographics, pre-filled user traits, and ML prediction for a subject.
    """
    wb = openpyxl.load_workbook(MASTER_XLSX, data_only=True)
    ws = wb.active
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    col_map = {name: c for c, name in enumerate(headers, 1) if name}

    row_data = {}
    row_idx = None
    for r in range(2, ws.max_row + 1):
        uid = str(ws.cell(r, col_map["Participant ID"]).value or "").strip()
        if uid.lower() == str(subj_id).strip().lower():
            row_idx = r
            for k, c in col_map.items():
                row_data[k] = ws.cell(r, c).value
            break

    # Find folder and photos
    target_folder = WRIST_DIR / str(subj_id).strip().lower()
    if not target_folder.exists():
        for d in WRIST_DIR.iterdir():
            if d.is_dir() and d.name.lower() == str(subj_id).strip().lower():
                target_folder = d
                break

    images = {}
    view_files = {
        "palm_open_front": "palm_open_front.jpg",
        "palm_open_back": "palm_open_back.jpg",
        "palm_closed_front": "palm_close_front.jpg",
        "palm_closed_back": "palm_close_back.jpg"
    }

    if target_folder.exists():
        for vk, fname in view_files.items():
            fpath = target_folder / fname
            if fpath.exists():
                try:
                    images[vk] = PILImage.open(fpath)
                except Exception:
                    images[vk] = None
            else:
                images[vk] = None

    # Load metadata.json if available
    meta = {}
    if target_folder.exists() and (target_folder / "metadata.json").exists():
        try:
            meta = json.loads((target_folder / "metadata.json").read_text())
        except Exception:
            pass

    # Pre-populate defaults:
    # 1. User given from metadata/Excel: Vein, Mole, Skin texture
    user_vein = row_data.get("Doctor Vein Visibility") or meta.get("doctor_verified_features", {}).get("vein_visibility") or "Low"
    user_mole = row_data.get("Doctor Mole Presence") or meta.get("doctor_verified_features", {}).get("mole") or "No"
    user_skin = row_data.get("Doctor Skin Texture") or meta.get("doctor_verified_features", {}).get("skin_texture") or "Smooth"

    # 2. Defaults for remaining 4
    doc_sc = row_data.get("Doctor Skin Color") or "Medium"
    doc_pc = row_data.get("Doctor Palm Color") or "PinkWhite"
    doc_sh = row_data.get("Doctor Hair Thickness") or "Thin"
    doc_shc = row_data.get("Doctor Hair Color") or "Brown"
    doc_dosha = row_data.get("Doctor Verified Prakriti (Ground Truth)") or row_data.get("ML Predicted Dosha") or "K-P"
    doc_notes = row_data.get("Doctor Clinical Notes") or ""

    return {
        "row_idx": row_idx,
        "id": subj_id,
        "name": row_data.get("Participant Name") or "Unknown",
        "age": row_data.get("Age") or 25,
        "sex": row_data.get("Sex") or ("F" if "female" in subj_id.lower() else "M"),
        "images": images,
        "ml_predicted_dosha": row_data.get("ML Predicted Dosha") or "K-P",
        "ml_confidence_str": row_data.get("ML Confidence / Percentages") or "",
        "defaults": {
            "dosha": doc_dosha,
            "vein": user_vein,
            "mole": user_mole,
            "skin_texture": user_skin,
            "skin_color": doc_sc,
            "palm_color": doc_pc,
            "hair_thickness": doc_sh,
            "hair_color": doc_shc,
            "notes": doc_notes
        }
    }


def save_doctor_verification(subj_id: str, verified_dosha: str, features: dict, clinical_notes: str = ""):
    """
    Performs 3-Way Synchronization:
      1. Master Excel (root & dataset/)
      2. Local Folder (dataset/Wrist_Dataset/{id}/metadata.json)
      3. Supabase Cloud (wrist_uploads table)
    """
    # 1. Update Master Excel Files
    for path in [DATASET_MASTER_XLSX, MASTER_XLSX]:
        if not path.exists():
            continue
        wb = openpyxl.load_workbook(path)
        ws = wb.active
        headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
        col_map = {name: c for c, name in enumerate(headers, 1) if name}

        fill_verified = PatternFill(start_color="F2F8F2", end_color="F2F8F2", fill_type="solid")
        f_bold = Font(name="Calibri", size=10, bold=True)

        for r in range(2, ws.max_row + 1):
            uid = str(ws.cell(r, col_map["Participant ID"]).value or "").strip()
            if uid.lower() == str(subj_id).strip().lower():
                # Status & Ground Truth
                ws.cell(r, col_map["Validation Status"], value="Verified (Clinical Ground Truth)")
                ws.cell(r, col_map["Doctor Verified Prakriti (Ground Truth)"], value=verified_dosha)
                ws.cell(r, col_map["Doctor Clinical Notes"], value=clinical_notes or "Clinically verified by Doctor")

                # 7 Clinical Markers
                ws.cell(r, col_map["Doctor Vein Visibility"], value=features.get("vein_visibility"))
                ws.cell(r, col_map["Doctor Mole Presence"], value=features.get("mole"))
                ws.cell(r, col_map["Doctor Skin Texture"], value=features.get("skin_texture"))
                ws.cell(r, col_map["Doctor Skin Color"], value=features.get("skin_color"))
                ws.cell(r, col_map["Doctor Palm Color"], value=features.get("palm_color"))
                ws.cell(r, col_map["Doctor Hair Thickness"], value=features.get("hair_thickness"))
                ws.cell(r, col_map["Doctor Hair Color"], value=features.get("hair_color"))

                # 19 One-hot columns
                sc = features.get("skin_color")
                ws.cell(r, col_map["SC_Fair"], value=1 if sc == "Fair" else 0)
                ws.cell(r, col_map["SC_Medium"], value=1 if sc == "Medium" else 0)
                ws.cell(r, col_map["SC_Dark"], value=1 if sc == "Dark" else 0)

                st = features.get("skin_texture")
                ws.cell(r, col_map["ST_Rough"], value=1 if st == "Rough" else 0)
                ws.cell(r, col_map["ST_Smooth"], value=1 if st == "Smooth" else 0)
                ws.cell(r, col_map["ST_Dry"], value=1 if st == "Dry" else 0)
                ws.cell(r, col_map["ST_Oily"], value=1 if st == "Oily" else 0)

                sh = features.get("hair_thickness")
                ws.cell(r, col_map["SH_Thick"], value=1 if sh == "Thick" else 0)
                ws.cell(r, col_map["SH_Thin"], value=1 if sh == "Thin" else 0)
                ws.cell(r, col_map["SH_Med"], value=1 if sh == "Med" else 0)

                shc = features.get("hair_color")
                ws.cell(r, col_map["SHC_Brown"], value=1 if shc == "Brown" else 0)
                ws.cell(r, col_map["SHC_Black"], value=1 if shc == "Black" else 0)

                ws.cell(r, col_map["Mole on skin"], value=1 if features.get("mole") == "Yes" else 0)

                vv = features.get("vein_visibility")
                ws.cell(r, col_map["VV_Low"], value=1 if vv == "Low" else 0)
                ws.cell(r, col_map["VV_Med"], value=1 if vv == "Med" else 0)
                ws.cell(r, col_map["VV_High"], value=1 if vv == "High" else 0)

                pc = features.get("palm_color")
                ws.cell(r, col_map["PC_PinkRed"], value=1 if pc == "PinkRed" else 0)
                ws.cell(r, col_map["PC_PinkWhite"], value=1 if pc == "PinkWhite" else 0)
                ws.cell(r, col_map["PC_Brown"], value=1 if pc == "Brown" else 0)

                for c in range(1, ws.max_column + 1):
                    ws.cell(r, c).fill = fill_verified
                ws.cell(r, col_map["Validation Status"]).font = f_bold
                ws.cell(r, col_map["Doctor Verified Prakriti (Ground Truth)"]).font = f_bold
                break
        wb.save(path)

    # 2. Update Local Folder metadata.json
    target_folder = WRIST_DIR / str(subj_id).strip().lower()
    if not target_folder.exists():
        for d in WRIST_DIR.iterdir():
            if d.is_dir() and d.name.lower() == str(subj_id).strip().lower():
                target_folder = d
                break

    if target_folder.exists():
        meta_file = target_folder / "metadata.json"
        meta = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
            except Exception:
                pass
        meta["validation_status"] = "Verified (Clinical Ground Truth)"
        meta["doctor_verified_prakriti"] = verified_dosha
        meta["dominant_dosha"] = verified_dosha[0].upper()
        meta["subdominant_dosha"] = verified_dosha.split("-")[1].upper() if "-" in verified_dosha else None
        meta["doctor_verified_features"] = features
        meta["doctor_notes"] = clinical_notes or "Clinically verified by Doctor"
        meta["is_duplicate"] = False
        meta_file.write_text(json.dumps(meta, indent=4))

    # 3. Update Supabase Cloud
    if supabase_client:
        gender = "female" if "female" in subj_id.lower() else "male"
        payload = {
            "user_id": subj_id,
            "gender": gender,
            "angle": "doctor_ground_truth_portal",
            "predicted_dosha": verified_dosha,
            "confidence": 1.0,
            "image_data": json.dumps({
                "validation_status": "Verified (Clinical Ground Truth)",
                "doctor_verified_prakriti": verified_dosha,
                "dominant_dosha": verified_dosha[0].upper(),
                "subdominant_dosha": verified_dosha.split("-")[1].upper() if "-" in verified_dosha else None,
                "clinical_features": features,
                "doctor_notes": clinical_notes,
                "source": "Doctor Clinical Portal"
            })
        }
        try:
            supabase_client.table("wrist_uploads").insert(payload).execute()
        except Exception as e:
            print(f"[Doctor Portal Supabase Error]: {e}")

    return True


def retrain_model_pipeline(progress_callback=None):
    """
    Retrains the unified ML pipeline on all Verified records.
    Uses cached feature vectors for instant retraining.
    """
    if progress_callback: progress_callback(0.1, "Reading verified clinical cohort...")

    wb = openpyxl.load_workbook(MASTER_XLSX, data_only=True)
    ws = wb.active
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    col_map = {name: c for c, name in enumerate(headers, 1) if name}

    # Load cache if available
    feature_cache = {}
    if FEATURE_CACHE_PATH.exists():
        try:
            feature_cache = joblib.load(FEATURE_CACHE_PATH)
        except Exception:
            feature_cache = {}

    X_list = []
    y_list = []
    subj_list = []

    # Gather all verified records
    verified_rows = []
    for r in range(2, ws.max_row + 1):
        status = str(ws.cell(r, col_map["Validation Status"]).value or "").strip().lower()
        if "duplicate" in status:
            continue
        if "verified" in status:
            uid = str(ws.cell(r, col_map["Participant ID"]).value).strip()
            age = ws.cell(r, col_map["Age"]).value or 25
            sex = str(ws.cell(r, col_map["Sex"]).value or "M").strip()
            dosha = str(ws.cell(r, col_map["Doctor Verified Prakriti (Ground Truth)"]).value or "").strip()
            if dosha and dosha[0].upper() in ["K", "P", "V"]:
                verified_rows.append({
                    "id": uid,
                    "age": float(age),
                    "is_male": 1.0 if sex.upper().startswith("M") else 0.0,
                    "label": dosha[0].upper() # Dominant dosha
                })

    n_samples = len(verified_rows)
    if n_samples < 20:
        return {"success": False, "error": f"Not enough verified records (found {n_samples})"}

    if progress_callback: progress_callback(0.3, f"Extracting/loading 70-D features for {n_samples} verified subjects...")

    cache_updated = False
    for i, item in enumerate(verified_rows):
        sid = item["id"]
        if sid in feature_cache:
            feat_vec = feature_cache[sid]
        else:
            # Extract from images
            target_folder = WRIST_DIR / sid.lower()
            if not target_folder.exists():
                for d in WRIST_DIR.iterdir():
                    if d.is_dir() and d.name.lower() == sid.lower():
                        target_folder = d
                        break

            view_files = {
                "palm_open_front": target_folder / "palm_open_front.jpg",
                "palm_open_back": target_folder / "palm_open_back.jpg",
                "palm_closed_front": target_folder / "palm_close_front.jpg",
                "palm_closed_back": target_folder / "palm_close_back.jpg"
            }
            images = {}
            for vk, fp in view_files.items():
                images[vk] = PILImage.open(fp) if fp.exists() else None

            feat_vec = extract_participant_features(images, age=item["age"], sex_m=item["is_male"])
            feature_cache[sid] = feat_vec
            cache_updated = True

        X_list.append(feat_vec)
        y_list.append(item["label"])

    if cache_updated:
        joblib.dump(feature_cache, FEATURE_CACHE_PATH)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list)

    if progress_callback: progress_callback(0.6, "Fitting class-weighted RBF-SVM & ExtraTrees pipelines...")

    # Define best pipeline: StandardScaler + class_weight='balanced' SVC(probability=True)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=2.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=42))
    ])

    # 5-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    bal_accs = cross_val_score(pipe, X, y, cv=skf, scoring="balanced_accuracy")
    raw_accs = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
    f1s = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro")

    # Fit final pipeline on 100% of verified data
    if progress_callback: progress_callback(0.9, "Fitting final production bundle...")
    pipe.fit(X, y)

    # Save to disk
    bundle = {
        "pipeline": pipe,
        "model_name": "Unified RBF-SVM (Balanced)",
        "classes": ["K", "P", "V"],
        "feature_names": FEATURE_NAMES,
        "view_keys": VIEW_KEYS,
        "n_verified_train": n_samples,
        "metrics": {
            "balanced_accuracy": float(np.mean(bal_accs)),
            "accuracy": float(np.mean(raw_accs)),
            "macro_f1": float(np.mean(f1s))
        }
    }
    joblib.dump(bundle, MODEL_PATH)

    if progress_callback: progress_callback(1.0, "Model successfully updated and deployed!")

    return {
        "success": True,
        "n_samples": n_samples,
        "class_distribution": pd.Series(y).value_counts().to_dict(),
        "balanced_accuracy": float(np.mean(bal_accs)),
        "accuracy": float(np.mean(raw_accs)),
        "macro_f1": float(np.mean(f1s))
    }
