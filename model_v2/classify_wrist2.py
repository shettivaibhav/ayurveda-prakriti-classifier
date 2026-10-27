import os
import shutil
import openpyxl
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = PROJECT_ROOT / "dataset" / "palm_dataset_26 June 26.xlsx"
WRIST_DATASET = PROJECT_ROOT / "dataset" / "Wrist_Dataset"
OUT_DIR = PROJECT_ROOT / "dataset" / "WristClassifiedData2"

def normalize_id(subj_id):
    if not subj_id:
        return ""
    num, _, gender = str(subj_id).partition("_")
    return f"{num.strip()}_{gender.strip().lower()}"

def main():
    print("Loading excel sheet...")
    wb = openpyxl.load_workbook(XLSX_PATH, read_only=True, data_only=True)
    ws = wb["Sheet"]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    
    prak_idx = header.index("Prakriti")
    
    # Map normalized ID to Prakriti
    prak_map = {}
    for row in rows[1:]:
        rid = row[0]
        if rid is not None:
            norm_id = normalize_id(rid)
            prak_map[norm_id] = row[prak_idx]
    
    print(f"Loaded {len(prak_map)} subjects from Excel.")
    
    # Re-create output folder structure
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for folder in ["Kapha", "Pitta", "Vata"]:
        (OUT_DIR / folder).mkdir(parents=True, exist_ok=True)
        
    dosha_to_dir = {
        'K': 'Kapha',
        'P': 'Pitta',
        'V': 'Vata'
    }
    
    copied_count = 0
    # Walk Wrist_Dataset
    for item in sorted(WRIST_DATASET.iterdir()):
        if item.is_dir():
            subj_name = item.name
            norm_id = normalize_id(subj_name)
            prak = prak_map.get(norm_id)
            
            if not prak:
                print(f"Warning: Subject {subj_name} (normalized: {norm_id}) not found or has no Prakriti in Excel.")
                continue
            
            # Extract unique dosha letters, ignoring spaces and hyphens
            prak_clean = str(prak).replace("-", "").replace(" ", "").upper()
            doshas = set(prak_clean)
            
            for d in doshas:
                if d in dosha_to_dir:
                    target_sub = dosha_to_dir[d]
                    dest = OUT_DIR / target_sub / subj_name
                    # Copy the whole subject directory
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                    print(f"Copied {subj_name} -> {target_sub}/")
                    copied_count += 1
                else:
                    print(f"Warning: Unknown dosha letter '{d}' for subject {subj_name} in Prakriti '{prak}'.")
                    
    print(f"\nCompleted! Total directory copies created under WristClassifiedData2: {copied_count}")

if __name__ == "__main__":
    main()
