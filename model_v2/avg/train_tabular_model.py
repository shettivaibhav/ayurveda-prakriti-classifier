import openpyxl
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, cross_val_predict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
XLSX_PATH = PROJECT_ROOT / "dataset" / "palm_dataset_26 June 26.xlsx"
OUT_PATH = PROJECT_ROOT / "mini" / "model_tabular.joblib"
RNG = 42

FEATURES = [
    "Age", "Sex_M", "Mole",
    "SC_Fair", "SC_Medium", "SC_Dark",
    "ST_Rough", "ST_Smooth", "ST_Dry", "ST_Oily",
    "SH_Thick", "SH_Thin", "SH_Med",
    "SHC_Brown", "SHC_Black",
    "VV_Low", "VV_Med", "VV_High",
    "PC_PinkRed", "PC_PinkWhite", "PC_Brown"
]

def clean_val(v):
    if v is None:
        return 0
    try:
        return int(v)
    except ValueError:
        return 0

def main():
    print("Loading excel sheet...")
    wb = openpyxl.load_workbook(XLSX_PATH, read_only=True, data_only=True)
    ws = wb["Sheet"]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    
    # Map column headers to indices
    hidx = {name: idx for idx, name in enumerate(header) if name is not None}
    
    X_list = []
    y_list = []
    
    print("Extracting features and targets...")
    for r_idx, row in enumerate(rows[1:], start=1):
        if row[0] is None or row[hidx.get("Prakriti")] is None:
            continue
            
        prak = str(row[hidx["Prakriti"]]).strip()
        ydom = prak[0]  # First letter = dominant dosha (K, P, V)
        
        # Build features dict
        rec = {
            "Age": clean_val(row[hidx["Age"]]),
            "Sex_M": 1 if str(row[hidx["Sex"]]).strip().upper() == 'M' else 0,
            "Mole": clean_val(row[hidx["Mole on skin"]]),
            "SC_Fair": clean_val(row[hidx["SC_Fair"]]),
            "SC_Medium": clean_val(row[hidx["SC_Medium"]]),
            "SC_Dark": clean_val(row[hidx["SC_Dark"]]),
            "ST_Rough": clean_val(row[hidx["ST_Rough"]]),
            "ST_Smooth": clean_val(row[hidx["ST_Smooth"]]),
            "ST_Dry": clean_val(row[hidx["ST_Dry"]]),
            "ST_Oily": clean_val(row[hidx["ST_Oily"]]),
            "SH_Thick": clean_val(row[hidx["SH_Thick"]]),
            "SH_Thin": clean_val(row[hidx["SH_Thin"]]),
            "SH_Med": clean_val(row[hidx["SH_Med"]]),
            "SHC_Brown": clean_val(row[hidx["SHC_Brown"]]),
            "SHC_Black": clean_val(row[hidx["SHC_Black"]]),
            "VV_Low": clean_val(row[hidx["VV_Low"]]),
            "VV_Med": clean_val(row[hidx["VV_Med"]]),
            "VV_High": clean_val(row[hidx["VV_High"]]),
            "PC_PinkRed": clean_val(row[hidx["PC_PinkRed"]]),
            "PC_PinkWhite": clean_val(row[hidx["PC_PinkWhite"]]),
            "PC_Brown": clean_val(row[hidx["PC_Brown"]]),
        }
        
        # Feature vector in the exact order of FEATURES
        feats = [rec[f] for f in FEATURES]
        X_list.append(feats)
        y_list.append(ydom)
        
    X = np.array(X_list, dtype=float)
    y = np.array(y_list)
    
    print(f"Dataset loaded. Total samples: {len(y)}")
    
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                min_samples_leaf=1, random_state=RNG)
                                
    # Evaluate model
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
    cvr = cross_validate(rf, X, y, cv=cv, scoring=["accuracy", "balanced_accuracy"])
    
    print(f"\nCV results (3-class dominant, {len(y)} subjects):")
    print(f"  Accuracy  : {cvr['test_accuracy'].mean():.1%}")
    print(f"  Balanced  : {cvr['test_balanced_accuracy'].mean():.1%}")
    
    # Stratified K-Fold for Confusion Matrix
    skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
    yp = cross_val_predict(rf, X, y, cv=skf)
    classes = sorted(pd.unique(y))
    cm = confusion_matrix(y, yp, labels=classes)
    
    print("\nConfusion Matrix (rows=truth, cols=pred):")
    print("  " + " ".join(classes))
    for i, c in enumerate(classes):
        print(f"  {c}: " + " ".join(f"{v:3d}" for v in cm[i]))
        
    # Fit model on all data and save
    print("\nTraining on all data...")
    rf.fit(X, y)
    
    bundle = {
        "model": rf,
        "classes": list(rf.classes_),
        "feature_names": FEATURES,
        "target": "dominant_dosha",
        "notes": "Tabular model trained on manual features: Age, Sex, Mole, Skin Color, Skin Texture, Hair Thickness, Hair Color, Vein Visibility, Palm Color -> dominant dosha K/P/V."
    }
    
    joblib.dump(bundle, OUT_PATH)
    print(f"Saved -> {OUT_PATH} (classes={bundle['classes']})")

if __name__ == "__main__":
    main()
