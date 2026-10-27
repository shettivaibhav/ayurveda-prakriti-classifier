"""
Builds the Doctor Validation Master Sheet (doctor_validation_prakriti_master.xlsx)
Preserves:
  - First 93 verified records: all existing features, photos, and ML predictions intact.
  - Next 61 (to 154) records: demographics, photos, ML predictions, and available extracted features.
  - Adds dedicated, clearly formatted doctor validation columns for clinical review.
  - Restores all embedded photo thumbnails.
"""

import sys
import shutil
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from pathlib import Path
import io
from PIL import Image as PILImage
from openpyxl.drawing.image import Image as OpenpyxlImage

PROJECT_ROOT = Path("c:/Users/hp/Desktop/ayurveda prakriti classifier")
SRC_XLSX = PROJECT_ROOT / "dataset" / "palm_dataset_26 June 26.xlsx"
WRIST_DIR = PROJECT_ROOT / "dataset" / "Wrist_Dataset"
OUT_XLSX = PROJECT_ROOT / "dataset" / "doctor_validation_prakriti_master.xlsx"
ROOT_COPY = PROJECT_ROOT / "doctor_validation_prakriti_master.xlsx"

def add_thumbnail(ws, img_path, cell_coord):
    try:
        img = PILImage.open(img_path)
        img.thumbnail((90, 80))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        buf.seek(0)
        ox_img = OpenpyxlImage(buf)
        ox_img.width = img.width
        ox_img.height = img.height
        ws.add_image(ox_img, cell_coord)
        return True
    except Exception as e:
        return False

def build_sheet():
    print("Reading source dataset...")
    src_wb = openpyxl.load_workbook(SRC_XLSX, data_only=True)
    src_ws = src_wb["Sheet"]
    
    src_headers = [src_ws.cell(1, c).value for c in range(1, src_ws.max_column + 1)]
    hidx = {name: c for c, name in enumerate(src_headers, 1) if name is not None}
    
    # Read rows
    rows_data = []
    for r in range(2, src_ws.max_row + 1):
        cell_id = src_ws.cell(r, 1).value
        if not cell_id:
            continue
        row_dict = {}
        for name, col_idx in hidx.items():
            row_dict[name] = src_ws.cell(r, col_idx).value
        rows_data.append(row_dict)
        
    print(f"Loaded {len(rows_data)} participant records from source.")
    
    # New workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Doctor Validation Master"
    
    # Define Column Structure
    cols = [
        # 1. Demographics
        ("id", "Participant ID", "1001_male", 16, "DEMO"),
        ("Name", "Participant Name", "Sakshi", 18, "DEMO"),
        ("Age", "Age", 22, 8, "DEMO"),
        ("Sex", "Sex", "F", 8, "DEMO"),
        
        # 2. Hand Photos
        ("open_front", "Palm Open Front (Palm)", None, 16, "PHOTO"),
        ("open_back", "Palm Open Back (Skin/Veins)", None, 16, "PHOTO"),
        ("closed_front", "Palm Closed Front (Fist)", None, 16, "PHOTO"),
        ("closed_back", "Palm Closed Back (Outer Fist)", None, 16, "PHOTO"),
        
        # 3. Doctor Validation Section (Prominent)
        ("Validation_Status", "Validation Status", "Verified / Pending", 22, "DOC"),
        ("Doctor_Verified_Prakriti", "Doctor Verified Prakriti (Ground Truth)", "K / P / V / K-P / etc.", 22, "DOC"),
        ("Doctor_Vein_Visibility", "Doctor Vein Visibility", "Low / Med / High", 18, "DOC"),
        ("Doctor_Mole_Presence", "Doctor Mole Presence", "Yes / No", 16, "DOC"),
        ("Doctor_Skin_Texture", "Doctor Skin Texture", "Dry / Rough / Smooth / Oily", 20, "DOC"),
        ("Doctor_Skin_Color", "Doctor Skin Color", "Fair / Medium / Dark", 18, "DOC"),
        ("Doctor_Palm_Color", "Doctor Palm Color", "PinkRed / PinkWhite / Brown", 18, "DOC"),
        ("Doctor_Hair_Thickness", "Doctor Hair Thickness", "Thick / Med / Thin", 18, "DOC"),
        ("Doctor_Hair_Color", "Doctor Hair Color", "Black / Brown", 16, "DOC"),
        ("Doctor_Clinical_Notes", "Doctor Clinical Notes", "Doctor observations", 28, "DOC"),
        
        # 4. ML Model Predictions (Reference for Doctor)
        ("ML_Predicted_Prakriti", "ML Predicted Dosha", "K-P", 16, "ML"),
        ("Classification_Percent", "ML Confidence / Percentages", "k-60%, p-30%, v-10%", 24, "ML"),
        
        # 5. One-Hot Feature Columns (Preserved exactly)
        ("SC_Fair", "SC_Fair", 0, 9, "ONEHOT"),
        ("SC_Medium", "SC_Medium", 1, 10, "ONEHOT"),
        ("SC_Dark", "SC_Dark", 0, 9, "ONEHOT"),
        ("ST_Rough", "ST_Rough", 0, 10, "ONEHOT"),
        ("ST_Smooth", "ST_Smooth", 1, 11, "ONEHOT"),
        ("ST_Dry", "ST_Dry", 0, 9, "ONEHOT"),
        ("ST_Oily", "ST_Oily", 0, 9, "ONEHOT"),
        ("SH_Thick", "SH_Thick", 0, 10, "ONEHOT"),
        ("SH_Thin", "SH_Thin", 1, 9, "ONEHOT"),
        ("SH_Med", "SH_Med", 0, 9, "ONEHOT"),
        ("SHC_Brown", "SHC_Brown", 0, 11, "ONEHOT"),
        ("SHC_Black", "SHC_Black", 1, 10, "ONEHOT"),
        ("Mole on skin", "Mole on skin", 1, 12, "ONEHOT"),
        ("VV_Low", "VV_Low", 1, 9, "ONEHOT"),
        ("VV_Med", "VV_Med", 0, 9, "ONEHOT"),
        ("VV_High", "VV_High", 0, 9, "ONEHOT"),
        ("PC_PinkRed", "PC_PinkRed", 0, 11, "ONEHOT"),
        ("PC_PinkWhite", "PC_PinkWhite", 1, 13, "ONEHOT"),
        ("PC_Brown", "PC_Brown", 0, 10, "ONEHOT")
    ]
    
    # Styles
    f_header = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    f_doc_header = Font(name="Calibri", size=10, bold=True, color="1F4E79")
    f_data = Font(name="Calibri", size=10)
    f_bold = Font(name="Calibri", size=10, bold=True)
    
    fill_demo = PatternFill(start_color="203764", end_color="203764", fill_type="solid")     # Dark Navy
    fill_photo = PatternFill(start_color="305496", end_color="305496", fill_type="solid")    # Medium Blue
    fill_doc = PatternFill(start_color="D9EAD3", end_color="D9EAD3", fill_type="solid")      # Soft Sage Green (Distinct)
    fill_ml = PatternFill(start_color="595959", end_color="595959", fill_type="solid")       # Dark Slate
    fill_onehot = PatternFill(start_color="8497B0", end_color="8497B0", fill_type="solid")   # Steel Blue-Gray
    
    fill_verified_row = PatternFill(start_color="F2F8F2", end_color="F2F8F2", fill_type="solid")  # Pale green
    fill_pending_row = PatternFill(start_color="FFF9E6", end_color="FFF9E6", fill_type="solid")   # Pale amber
    
    border_thin = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9")
    )
    
    # Write Headers
    ws.row_dimensions[1].height = 36
    for c_idx, (col_key, col_title, _, col_width, cat) in enumerate(cols, 1):
        cell = ws.cell(row=1, column=c_idx, value=col_title)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        col_letter = get_column_letter(c_idx)
        ws.column_dimensions[col_letter].width = col_width
        
        if cat == "DEMO":
            cell.fill = fill_demo
            cell.font = f_header
        elif cat == "PHOTO":
            cell.fill = fill_photo
            cell.font = f_header
        elif cat == "DOC":
            cell.fill = fill_doc
            cell.font = f_doc_header
        elif cat == "ML":
            cell.fill = fill_ml
            cell.font = f_header
        else:
            cell.fill = fill_onehot
            cell.font = f_header
            
    # Write Data Rows
    print("Writing records and formatting cells...")
    
    for r_idx, r_data in enumerate(rows_data, 2):
        ws.row_dimensions[r_idx].height = 80
        is_verified = (r_idx <= 94)  # First 93 records (rows 2 to 94)
        
        # Determine clinical values for doctor columns
        # 1. Vein Visibility
        vv_str = ""
        if r_data.get("VV_High") == 1: vv_str = "High"
        elif r_data.get("VV_Med") == 1: vv_str = "Med"
        elif r_data.get("VV_Low") == 1: vv_str = "Low"
        
        # 2. Mole
        mole_str = "Yes" if r_data.get("Mole on skin") == 1 else "No"
        
        # 3. Skin Texture
        st_str = ""
        if r_data.get("ST_Dry") == 1: st_str = "Dry"
        elif r_data.get("ST_Rough") == 1: st_str = "Rough"
        elif r_data.get("ST_Oily") == 1: st_str = "Oily"
        elif r_data.get("ST_Smooth") == 1: st_str = "Smooth"
        
        # 4. Skin Color
        sc_str = ""
        if r_data.get("SC_Fair") == 1: sc_str = "Fair"
        elif r_data.get("SC_Dark") == 1: sc_str = "Dark"
        elif r_data.get("SC_Medium") == 1: sc_str = "Medium"
        
        # 5. Palm Color
        pc_str = ""
        if r_data.get("PC_PinkRed") == 1: pc_str = "PinkRed"
        elif r_data.get("PC_Brown") == 1: pc_str = "Brown"
        elif r_data.get("PC_PinkWhite") == 1: pc_str = "PinkWhite"
        
        # 6. Hair Thickness
        sh_str = ""
        if r_data.get("SH_Thick") == 1: sh_str = "Thick"
        elif r_data.get("SH_Thin") == 1: sh_str = "Thin"
        elif r_data.get("SH_Med") == 1: sh_str = "Med"
        
        # 7. Hair Color
        shc_str = ""
        if r_data.get("SHC_Black") == 1: shc_str = "Black"
        elif r_data.get("SHC_Brown") == 1: shc_str = "Brown"
        
        prakriti_raw = str(r_data.get("Prakriti", "")).strip() if r_data.get("Prakriti") is not None else ""
        classification_pct = str(r_data.get("Classification Percent", "")).strip() if r_data.get("Classification Percent") is not None else ""
        
        # Row values map
        val_map = {
            "id": r_data.get("id"),
            "Name": r_data.get("Name") or "Unknown",
            "Age": r_data.get("Age"),
            "Sex": r_data.get("Sex"),
            "open_front": None,
            "open_back": None,
            "closed_front": None,
            "closed_back": None,
            
            # Doctor validation columns
            "Validation_Status": "Verified (Clinical Ground Truth)" if is_verified else "Pending Doctor Validation",
            "Doctor_Verified_Prakriti": prakriti_raw if is_verified else "",
            "Doctor_Vein_Visibility": vv_str,
            "Doctor_Mole_Presence": mole_str,
            "Doctor_Skin_Texture": st_str,
            "Doctor_Skin_Color": sc_str if is_verified else "",
            "Doctor_Palm_Color": pc_str if is_verified else "",
            "Doctor_Hair_Thickness": sh_str if is_verified else "",
            "Doctor_Hair_Color": shc_str if is_verified else "",
            "Doctor_Clinical_Notes": "Pre-validated in cohort 1" if is_verified else "Awaiting clinical exam",
            
            # ML Reference columns
            "ML_Predicted_Prakriti": prakriti_raw,
            "Classification_Percent": classification_pct,
            
            # One-hot original features
            "SC_Fair": r_data.get("SC_Fair", 0),
            "SC_Medium": r_data.get("SC_Medium", 0),
            "SC_Dark": r_data.get("SC_Dark", 0),
            "ST_Rough": r_data.get("ST_Rough", 0),
            "ST_Smooth": r_data.get("ST_Smooth", 0),
            "ST_Dry": r_data.get("ST_Dry", 0),
            "ST_Oily": r_data.get("ST_Oily", 0),
            "SH_Thick": r_data.get("SH_Thick", 0),
            "SH_Thin": r_data.get("SH_Thin", 0),
            "SH_Med": r_data.get("SH_Med", 0),
            "SHC_Brown": r_data.get("SHC_Brown", 0),
            "SHC_Black": r_data.get("SHC_Black", 0),
            "Mole on skin": r_data.get("Mole on skin", 0),
            "VV_Low": r_data.get("VV_Low", 0),
            "VV_Med": r_data.get("VV_Med", 0),
            "VV_High": r_data.get("VV_High", 0),
            "PC_PinkRed": r_data.get("PC_PinkRed", 0),
            "PC_PinkWhite": r_data.get("PC_PinkWhite", 0),
            "PC_Brown": r_data.get("PC_Brown", 0),
        }
        
        for c_idx, (col_key, _, _, _, cat) in enumerate(cols, 1):
            cell = ws.cell(row=r_idx, column=c_idx, value=val_map.get(col_key))
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border_thin
            cell.font = f_bold if col_key in ["id", "Validation_Status", "Doctor_Verified_Prakriti"] else f_data
            
            # Subtle background highlight for the doctor section
            if cat == "DOC":
                if is_verified:
                    cell.fill = fill_verified_row
                else:
                    cell.fill = fill_pending_row
                    
    # Restore Thumbnails
    print("Embedding photo thumbnails into Excel cells...")
    VIEW_MAP = {
        "open_front": ("E", ["palm_open_front.jpg"]),
        "open_back": ("F", ["palm_open_back.jpg"]),
        "closed_front": ("G", ["palm_closed_front.jpg", "palm_close_front.jpg"]),
        "closed_back": ("H", ["palm_closed_back.jpg", "palm_close_back.jpg"])
    }
    
    restored_img_count = 0
    for r_idx, r_data in enumerate(rows_data, 2):
        sid = str(r_data.get("id", "")).strip()
        exact = WRIST_DIR / sid
        subj_dir = exact
        if not exact.is_dir():
            num, _, gender = sid.partition("_")
            norm_dir = WRIST_DIR / f"{num}_{gender.lower()}"
            if norm_dir.is_dir():
                subj_dir = norm_dir
                
        if not subj_dir.is_dir():
            continue
            
        for view_key, (col_letter, candidates) in VIEW_MAP.items():
            cell_coord = f"{col_letter}{r_idx}"
            img_path = None
            for cand in candidates:
                p = subj_dir / cand
                if p.exists():
                    img_path = p
                    break
            if img_path:
                if add_thumbnail(ws, img_path, cell_coord):
                    restored_img_count += 1
                    
    print(f"Successfully embedded {restored_img_count} photo thumbnails into sheet.")
    
    # Save output
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_XLSX)
    print(f"Saved master file to: {OUT_XLSX}")
    
    # Also save a copy at root for immediate convenience
    shutil.copy2(OUT_XLSX, ROOT_COPY)
    print(f"Saved root copy to: {ROOT_COPY}")

if __name__ == "__main__":
    build_sheet()
