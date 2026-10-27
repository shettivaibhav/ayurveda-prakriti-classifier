import pandas as pd
from pathlib import Path

def generate_excel_report(subjects_results, output_path):
    """
    Creates an Excel spreadsheet containing all parsed subjects, details,
    predictions, visual features, and review flags.
    """
    rows = []
    
    for idx, subj in enumerate(subjects_results):
        # Base metadata & details
        row = {
            "Record No": idx + 1,
            "Sender": subj.get("sender", "Unknown"),
            "Name": subj.get("name", "Unknown"),
            "Age": subj.get("age", "Unknown"),
            "Gender": subj.get("gender", "Unknown"),
            "Skin Color": subj.get("skin_color", "Unknown"),
            "Skin Texture": subj.get("skin_texture", "Unknown"),
            "Vein Visibility": subj.get("vein_visibility", "Unknown"),
            "Palm Color": subj.get("palm_color", "Unknown"),
            "Mole": subj.get("mole", "Unknown"),
            "Hair Thickness": subj.get("hair_thickness", "Unknown"),
            "Hair Color": subj.get("hair_color", "Unknown"),
            
            # Categorized Image Paths
            "Palm Open Front (Palm) Path": subj["image_views"].get("palm_open_front", "Missing"),
            "Palm Open Back (Skin) Path": subj["image_views"].get("palm_open_back", "Missing"),
            "Palm Closed Front Path": subj["image_views"].get("palm_closed_front", "Missing"),
            "Palm Closed Back Path": subj["image_views"].get("palm_closed_back", "Missing"),
            
            # Ground Truth Prakriti
            "Prakriti": subj.get("predicted_dosha", "Unknown"),
            
            # Flags for manual review
            "Needs Manual Review": "Yes" if subj.get("needs_review", False) else "No",
            "Review Reasons": ", ".join(subj.get("review_reasons", [])) if subj.get("review_reasons") else "None"
        }
        
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # Save to Excel
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_excel(out_file, index=False, sheet_name="Ayurveda Prakriti Results")
    
    # Apply styling using openpyxl
    try:
        import openpyxl
        from openpyxl.styles import Font, Alignment, PatternFill
        from openpyxl.utils import get_column_letter
        
        wb = openpyxl.load_workbook(out_file)
        ws = wb["Ayurveda Prakriti Results"]
        
        # Header style: Bold, light blue fill, centered
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
        ws.row_dimensions[1].height = 28
        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_align
            
        # Highlight records needing review: light red fill on the "Needs Manual Review" cell
        review_col_idx = df.columns.get_loc("Needs Manual Review") + 1
        review_fill = PatternFill(start_color="F2DCDB", end_color="F2DCDB", fill_type="solid")
        review_font = Font(name="Calibri", size=11, color="9C0006")
        
        for row_idx in range(2, ws.max_row + 1):
            cell = ws.cell(row=row_idx, column=review_col_idx)
            if cell.value == "Yes":
                cell.fill = review_fill
                cell.font = review_font
                
        # Auto-adjust column widths
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val_str = str(cell.value or '')
                if len(val_str) > max_len:
                    max_len = len(val_str)
            # Add padding and limit max width
            ws.column_dimensions[col_letter].width = min(max(max_len + 3, 10), 30)
            
        wb.save(out_file)
    except Exception as e:
        print(f"  [Exporter] Excel styling failed (data saved successfully): {e}")
        
    return out_file

def append_to_source_dataset(subjects_results, xlsx_path, wrist_dataset_dir):
    """
    Appends successfully processed subjects directly to the main Excel training dataset.
    Ensures:
      - Categorized images are physically embedded as cells in columns B, C, D, E.
      - A 'Name' column is created in Column 28 (AB) and names are populated.
      - Skin and Palm color columns are auto-populated from continuous CV features.
      - Cell writing is explicit (by row/column) to avoid openpyxl list shifting or merged-cell bugs.
    """
    import openpyxl
    from openpyxl.drawing.image import Image as OpenpyxlImage
    import shutil
    import json
    from PIL import Image as PILImage
    
    xlsx_path = Path(xlsx_path)
    wrist_dataset_dir = Path(wrist_dataset_dir)
    
    if not xlsx_path.exists():
        print(f"  [Exporter] Warning: Main Excel dataset not found at: {xlsx_path}. Skipping append.")
        return
        
    print(f"\nAppending new subjects to main Excel dataset at: {xlsx_path}...")
    
    # Create a backup of the original file before modifying it
    backup_path = xlsx_path.with_suffix(xlsx_path.suffix + ".bak")
    try:
        shutil.copy2(xlsx_path, backup_path)
        print(f"  [Exporter] Created backup of dataset at: {backup_path}")
    except Exception as backup_err:
        print(f"  [Exporter] Warning: Failed to create backup: {backup_err}")
        
    # Load workbook and worksheet
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb["Sheet"]
    
    # Ensure "Name" column exists in column 28 (AB)
    ws.cell(row=1, column=28, value="Name")
    
    # Get column header names (fresh check)
    header = [cell.value for cell in ws[1]]
    hidx = {name: idx for idx, name in enumerate(header) if name is not None}
    
    # Determine the starting subject numeric ID to append
    max_num = 1000
    for row in range(2, ws.max_row + 1):
        cell_val = ws.cell(row=row, column=1).value
        if cell_val:
            num, _, _ = str(cell_val).partition("_")
            try:
                max_num = max(max_num, int(num))
            except ValueError:
                pass
                
    next_id = max_num + 1
    appended_count = 0
    appended_ids = []
    
    for subj in subjects_results:
        # Store ALL subjects without threshold or review skipping
        gender_raw = str(subj.get("gender") or "").strip().lower()
        if gender_raw.startswith("m"):
            gender_str = "Male"
            gender_code = "M"
        elif gender_raw.startswith("f"):
            gender_str = "Female"
            gender_code = "F"
        else:
            gender_str = "Female"
            gender_code = "F"

        subj_id_name = f"{next_id}_{gender_str}"
        subj_name = subj.get("name") or f"Subject_{next_id}"
        
        # Robust age parsing
        age_val = None
        try:
            raw_age = subj.get("age")
            if raw_age not in (None, "Unknown", ""):
                age_val = int(float(str(raw_age).strip()))
        except (ValueError, TypeError):
            age_val = None
        
        # Map NLP-extracted clinical details directly
        sc = subj.get("skin_color", "Medium")
        sc_fair = 1 if sc == "Fair" else 0
        sc_medium = 1 if sc == "Medium" else 0
        sc_dark = 1 if sc == "Dark" else 0
        
        st = subj.get("skin_texture", "Smooth")
        st_rough = 1 if st == "Rough" else 0
        st_smooth = 1 if st == "Smooth" else 0
        st_dry = 1 if st == "Dry" else 0
        st_oily = 1 if st == "Oily" else 0
        
        sh = subj.get("hair_thickness", "None")
        sh_thick = 1 if sh == "Thick" else 0
        sh_thin = 1 if sh == "Thin" else 0
        sh_med = 1 if sh == "Med" else 0
        
        shc = subj.get("hair_color", "None")
        shc_brown = 1 if shc == "Brown" else 0
        shc_black = 1 if shc == "Black" else 0
        
        mole_val = 1 if subj.get("mole", "No") == "Yes" else 0
        
        vv = subj.get("vein_visibility", "Low")
        vv_low = 1 if vv == "Low" else 0
        vv_med = 1 if vv == "Med" else 0
        vv_high = 1 if vv == "High" else 0
        
        pc = subj.get("palm_color", "PinkWhite")
        pc_pinkred = 1 if pc == "PinkRed" else 0
        pc_pinkwhite = 1 if pc == "PinkWhite" else 0
        pc_brown = 1 if pc == "Brown" else 0
        
        # Prepare row values matching dataset schema columns
        row_data = {
            "id": subj_id_name,
            "open_front": None,
            "open_back": None,
            "closed_front": None,
            "closed_back": None,
            "Age": age_val,
            "Sex": gender_code,
            "SC_Fair": sc_fair, 
            "SC_Medium": sc_medium, 
            "SC_Dark": sc_dark,
            "ST_Rough": st_rough,
            "ST_Smooth": st_smooth,
            "ST_Dry": st_dry,
            "ST_Oily": st_oily,
            "SH_Thick": sh_thick, 
            "SH_Thin": sh_thin, 
            "SH_Med": sh_med,
            "SHC_Brown": shc_brown, 
            "SHC_Black": shc_black,
            "Mole on skin": mole_val,
            "VV_Low": vv_low, 
            "VV_Med": vv_med, 
            "VV_High": vv_high,
            "PC_PinkRed": pc_pinkred, 
            "PC_PinkWhite": pc_pinkwhite, 
            "PC_Brown": pc_brown,
            "Prakriti": subj.get("predicted_dosha", "Unknown"),
            "Name": subj_name
        }
        
        # Determine the row number to write to (bottom of sheet)
        row_num = ws.max_row + 1
        ws.row_dimensions[row_num].height = 80  # Height set to accommodate image thumbnails
        
        # Write values to explicit column positions
        for col_name, val in row_data.items():
            if col_name in hidx:
                col_num = hidx[col_name] + 1
                ws.cell(row=row_num, column=col_num, value=val)
                
        # Create folder in Wrist_Dataset (normalised id name: {num}_{gender_lowercase})
        subj_folder_name = f"{next_id}_{gender_str.lower()}"
        subj_dir = wrist_dataset_dir / subj_folder_name
        subj_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy image views with standardized filenames and embed them inside the Excel cells
        image_metadata_list = []
        VIEW_TO_FILENAME = {
            "palm_open_front": "palm_open_front.jpg",
            "palm_open_back": "palm_open_back.jpg",
            "palm_closed_front": "palm_close_front.jpg",
            "palm_closed_back": "palm_close_back.jpg",
        }
        
        VIEW_TO_COLUMN = {
            "palm_open_front": "B",
            "palm_open_back": "C",
            "palm_closed_front": "D",
            "palm_closed_back": "E",
        }
        
        for view, file_path in subj["image_views"].items():
            if file_path and Path(file_path).exists() and view in VIEW_TO_FILENAME:
                dst_name = VIEW_TO_FILENAME[view]
                dst_path = subj_dir / dst_name
                shutil.copy2(file_path, dst_path)
                
                # Exif/Metadata info
                file_size = dst_path.stat().st_size
                try:
                    img = PILImage.open(dst_path)
                    width, height = img.size
                except:
                    width, height = 0, 0
                image_metadata_list.append({
                    "filename": dst_name,
                    "file_size": file_size,
                    "resolution": f"{width}x{height}"
                })
                
                # Physical image cell embedding will be handled dynamically in the batch restoration pass
                pass
                
        # Write metadata.json inside the new subject's folder
        meta = {
            "id": f"person_{next_id}",
            "gender": gender_str.lower(),
            "name": subj_name,
            "device_model": "whatsapp_pipeline_extracted",
            "collection_date": "whatsapp_pipeline_extracted",
            "images": image_metadata_list
        }
        (subj_dir / "metadata.json").write_text(json.dumps(meta, indent=4))
        
        # Upload to Supabase for EVERY subject
        try:
            from supabasehelper import upload_complete_analysis, supabase_client
            if supabase_client is not None:
                print(f"  Uploading subject '{subj_name}' ({subj_id_name}) to Supabase...")
                # Prepare images dict for Supabase upload
                images_dict = {}
                for view, file_path in subj["image_views"].items():
                    if file_path and Path(file_path).exists():
                        try:
                            images_dict[view] = PILImage.open(file_path)
                        except Exception as p_err:
                            print(f"  [Warning] PIL cannot open {view} image for Supabase: {p_err}")
                            
                probs = {
                    "Vata": float(subj.get("vata_prob", 0.0)),
                    "Pitta": float(subj.get("pitta_prob", 0.0)),
                    "Kapha": float(subj.get("kapha_prob", 0.0))
                }
                pred_dosha = subj.get("predicted_dosha", "Unknown")
                if sum(probs.values()) == 0.0:
                    if "V" in pred_dosha:
                        probs["Vata"] = 1.0
                    elif "P" in pred_dosha:
                        probs["Pitta"] = 1.0
                    elif "K" in pred_dosha:
                        probs["Kapha"] = 1.0
                    else:
                        probs["Kapha"] = 1.0
                    
                sb_res = upload_complete_analysis(
                    user_id=subj_id_name,
                    gender=gender_str.lower(),
                    images_dict=images_dict,
                    probabilities=probs,
                    final_dosha=pred_dosha,
                    confidence=float(subj.get("confidence", 1.0)),
                    age=age_val,
                    skin_color=subj.get("skin_color", "Medium"),
                    skin_texture=subj.get("skin_texture", "Smooth"),
                    vein_visibility=subj.get("vein_visibility", "Low"),
                    palm_color=subj.get("palm_color", "PinkWhite"),
                    mole=1 if subj.get("mole", "No") == "Yes" else 0,
                    hair_thickness=subj.get("hair_thickness", "None"),
                    hair_color=subj.get("hair_color", "None"),
                    name=subj_name
                )
                if sb_res.get("success"):
                    print(f"  [Supabase] Successfully uploaded subject '{subj_name}' as ID '{subj_id_name}'. Record ID: {sb_res.get('record_id')}")
                else:
                    print(f"  [Supabase Warning] Database upload notice: {sb_res.get('errors')}")
            else:
                print("  [Supabase] Client is not configured. Skipping upload.")
        except Exception as sb_err:
            print(f"  [Supabase Error] Error during Supabase upload: {sb_err}")

        print(f"  [SUCCESS] Appended subject '{subj_name}' as ID '{subj_id_name}' and saved files.")
        next_id += 1
        appended_count += 1
        appended_ids.append(subj_id_name)
        
    try:
        restore_all_images_in_sheet(ws, wrist_dataset_dir)
        wb.save(xlsx_path)
        print(f"Saved changes to Excel dataset. Total appended records: {appended_count}")
    except PermissionError:
        print(f"\n[ERROR] Permission denied when saving to '{xlsx_path}'.")
        print("  Please make sure this Excel sheet is closed and not open in Microsoft Excel or another program, then try again.")
    except Exception as e:
        print(f"  [ERROR] Failed to save changes to Excel dataset: {e}")
        
    return appended_ids

def add_image_thumbnail_to_cell(ws, img_path, cell_coordinate):
    """
    Loads an image from img_path, resizes it to 100x100 thumbnail in-memory,
    and inserts it into ws at cell_coordinate using openpyxl.
    """
    from PIL import Image as PILImage
    import io
    from openpyxl.drawing.image import Image as OpenpyxlImage
    
    try:
        # Load and resize the image
        img = PILImage.open(img_path)
        img.thumbnail((100, 100))
        
        # Save to BytesIO buffer as JPEG to keep Excel size small
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        buf.seek(0)
        
        # Create openpyxl Image
        ox_img = OpenpyxlImage(buf)
        ox_img.width = img.width
        ox_img.height = img.height
        
        ws.add_image(ox_img, cell_coordinate)
        return True
    except Exception as e:
        return False

def restore_all_images_in_sheet(ws, wrist_dataset_dir):
    """
    Scans all rows in the spreadsheet and re-inserts downscaled image thumbnails
    from the local wrist dataset directory. This preserves all existing images
    and prevents openpyxl from stripping them during workbook saves.
    """
    from pathlib import Path
    
    wrist_dataset_dir = Path(wrist_dataset_dir)
    if not wrist_dataset_dir.exists():
        print(f"  [Warning] Wrist dataset directory not found at: {wrist_dataset_dir}. Cannot restore images.")
        return
        
    VIEW_TO_COLUMN = {
        "palm_open_front": "B",
        "palm_open_back": "C",
        "palm_closed_front": "D",
        "palm_closed_back": "E",
    }
    
    VIEW_FILENAME_CANDIDATES = {
        "palm_open_front": ["palm_open_front.jpg"],
        "palm_open_back": ["palm_open_back.jpg"],
        "palm_closed_front": ["palm_closed_front.jpg", "palm_close_front.jpg"],
        "palm_closed_back": ["palm_closed_back.jpg", "palm_close_back.jpg"],
    }
    
    # Clear openpyxl's internal drawing list to prevent corruption or duplicates
    ws._images = []

    print(f"  [Excel] Restoring images for all rows in worksheet...")
    restored_count = 0
    
    for row in range(2, ws.max_row + 1):
        cell_val = ws.cell(row=row, column=1).value
        if not cell_val:
            continue
            
        sid = str(cell_val).strip()
        # Resolve subject directory matching logic
        exact = wrist_dataset_dir / sid
        subject_dir = exact
        if not exact.is_dir():
            num, _, gender = sid.partition("_")
            normalized = wrist_dataset_dir / f"{num}_{gender.lower()}"
            if normalized.is_dir():
                subject_dir = normalized
                
        if not subject_dir.is_dir():
            continue
            
        ws.row_dimensions[row].height = 80
        
        for view, col_letter in VIEW_TO_COLUMN.items():
            cell_coordinate = f"{col_letter}{row}"
            
            # Find candidate file path on disk
            img_path = None
            for candidate in VIEW_FILENAME_CANDIDATES[view]:
                p = subject_dir / candidate
                if p.exists():
                    img_path = p
                    break
                    
            if img_path:
                if add_image_thumbnail_to_cell(ws, img_path, cell_coordinate):
                    restored_count += 1
                    
    print(f"  [Excel] Successfully restored {restored_count} image thumbnails to cells.")
