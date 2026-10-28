import os
import json
import base64
import shutil
import urllib.request
import urllib.error
import time
import pandas as pd
from pathlib import Path

# Add root folder and mini folder to python path to resolve modules
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for sub in ["backend", "mini"]:
    sub_path = PROJECT_ROOT / sub
    if sub_path.exists() and str(sub_path) not in sys.path:
        sys.path.insert(0, str(sub_path))

import joblib
from PIL import Image

from supabasehelper import update_wrist_data_prakriti
from whatsapp_pipeline.exporter import restore_all_images_in_sheet

def load_unclassified_subjects(xlsx_path, subject_ids=None):
    """
    Loads the Excel training sheet and returns the dataframe and a list of row indices
    that do not yet have a Prakriti classification.
    If subject_ids is provided, it only returns unclassified indices matching those IDs.
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel dataset not found at: {xlsx_path}")
        
    df = pd.read_excel(xlsx_path)
    
    if subject_ids is not None:
        # Filter by specific subject IDs
        unclassified_indices = []
        for idx in range(len(df)):
            row_id = str(df.loc[idx, "id"]).strip()
            if row_id in subject_ids:
                val = str(df.loc[idx, "Prakriti"]).strip()
                if val == "Unknown" or pd.isna(df.loc[idx, "Prakriti"]) or val == "":
                    unclassified_indices.append(idx)
    else:
        # Default fallback: look for unclassified subjects from index 91 (row 93) onwards
        indices = range(91, len(df))
        unclassified_indices = []
        for idx in indices:
            val = str(df.loc[idx, "Prakriti"]).strip()
            if val == "Unknown" or pd.isna(df.loc[idx, "Prakriti"]) or val == "":
                unclassified_indices.append(idx)
            
    return df, unclassified_indices

def generate_few_shot_examples(df):
    """
    Generates text descriptions of clinic examples from the first 92 rows
    to use as context/few-shot training in the Gemini prompt.
    """
    examples = []
    # Use first 93 entries (indices 0 to 92)
    for idx in range(min(93, len(df))):
        row = df.iloc[idx]
        prakriti = str(row["Prakriti"]).strip()
        if not prakriti or prakriti == "Unknown" or pd.isna(row["Prakriti"]):
            continue
            
        gender = "female" if str(row["Sex"]).strip().upper() == "F" else "male"
        age = row["Age"]
        
        # Skin Color
        sc = "Fair" if row.get("SC_Fair") == 1 else ("Medium" if row.get("SC_Medium") == 1 else ("Dark" if row.get("SC_Dark") == 1 else "Unknown"))
        # Skin Texture
        st = "Rough" if row.get("ST_Rough") == 1 else ("Smooth" if row.get("ST_Smooth") == 1 else ("Dry" if row.get("ST_Dry") == 1 else ("Oily" if row.get("ST_Oily") == 1 else "Unknown")))
        # Hair Thickness
        ht = "Thick" if row.get("SH_Thick") == 1 else ("Thin" if row.get("SH_Thin") == 1 else ("Med" if row.get("SH_Med") == 1 else "Unknown"))
        # Vein Visibility
        vv = "Low" if row.get("VV_Low") == 1 else ("Med" if row.get("VV_Med") == 1 else ("High" if row.get("VV_High") == 1 else "Unknown"))
        # Palm Color
        pc = "PinkRed" if row.get("PC_PinkRed") == 1 else ("PinkWhite" if row.get("PC_PinkWhite") == 1 else ("Brown" if row.get("PC_Brown") == 1 else "Unknown"))
        
        examples.append(
            f"- Subject ID: {row['id']}\n"
            f"  Age: {age}, Gender: {gender}\n"
            f"  Observations: Skin Color={sc}, Skin Texture={st}, Hair Thickness={ht}, Vein Visibility={vv}, Palm Color={pc}\n"
            f"  Prakriti: {prakriti}"
        )
        if len(examples) >= 10:
            break
            
    return "\n".join(examples)

def call_gemini_with_retry(url, payload, max_retries=5, initial_backoff=2.0):
    """
    Calls the Gemini API and handles 429 Rate Limit errors with exponential backoff.
    """
    backoff = initial_backoff
    req_data = json.dumps(payload).encode("utf-8")
    
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=35) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                content = res_data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(content.strip())
        except urllib.error.HTTPError as http_err:
            if http_err.code == 429:
                # Rate limit hit, backoff and retry
                time.sleep(backoff)
                backoff *= 2.0
                continue
            else:
                # Other HTTP error
                raise http_err
        except Exception as e:
            # Network or other exceptions
            if attempt == max_retries - 1:
                raise e
            time.sleep(backoff)
            backoff *= 2.0
            
    raise urllib.error.HTTPError(url, 429, "Too Many Requests - Retries exhausted", {}, None)

def extract_visual_features_with_gemini(subject_row, image_dir, api_key):
    """
    Queries Gemini multimodal API to classify physical features based on the 4 hand images.
    """
    subj_id = subject_row["id"]
    
    subj_folder_name = str(subj_id).lower()
    subj_path = Path(image_dir) / subj_folder_name
    if not subj_path.exists():
        found = False
        for child in Path(image_dir).iterdir():
            if child.is_dir() and child.name.lower() == subj_folder_name:
                subj_path = child
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Subject folder '{subj_folder_name}' not found under {image_dir}")
            
    # Check for all 4 images
    image_filenames = {
        "palm_open_front": "palm_open_front.jpg",
        "palm_open_back": "palm_open_back.jpg",
        "palm_closed_front": "palm_close_front.jpg",
        "palm_closed_back": "palm_close_back.jpg",
    }
    
    parts = []
    prompt = (
        "You are an expert Ayurvedic practitioner and researcher specializing in Prakriti classification and physical feature extraction.\n"
        "Your task is to analyze the attached hand images of a subject and classify their physical characteristics.\n\n"
        "Analyze the images for the following features and choose the best matching value for each:\n"
        "1. skin_color: Choose from ['Fair', 'Medium', 'Dark'] (Use the back of hand or wrist to determine the base skin tone)\n"
        "2. skin_texture: Choose from ['Rough', 'Smooth', 'Dry', 'Oily']\n"
        "3. hair_thickness: Choose from ['Thick', 'Thin', 'Med'] (Analyze hair visible on the back of hand/wrist)\n"
        "4. hair_color: Choose from ['Brown', 'Black']\n"
        "5. mole_on_skin: Choose from [0, 1] (Set to 1 if a mole is visible anywhere on the hand skin, 0 otherwise)\n"
        "6. vein_visibility: Choose from ['Low', 'Med', 'High'] (Analyze the visibility of veins on the back of the hand or wrist)\n"
        "7. palm_color: Choose from ['PinkRed', 'PinkWhite', 'Brown'] (Determine the base color of the inner palm)\n\n"
        "Return a JSON object with these exact keys:\n"
        "- \"skin_color\": string\n"
        "- \"skin_texture\": string\n"
        "- \"hair_thickness\": string\n"
        "- \"hair_color\": string\n"
        "- \"mole_on_skin\": integer (0 or 1)\n"
        "- \"vein_visibility\": string\n"
        "- \"palm_color\": string\n"
        "- \"reasoning\": A brief explanation (1-2 sentences) describing the visual cues that led to these classification decisions."
    )
    parts.append({"text": prompt})
    
    for view, filename in image_filenames.items():
        img_path = subj_path / filename
        if img_path.exists():
            img_data = base64.b64encode(img_path.read_bytes()).decode("utf-8")
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": img_data
                }
            })
            parts.append({"text": f"Above image is {view} for subject {subj_id}"})
            
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": parts
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"}
    )
    
    import time
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            
            elapsed = time.time() - start_time
            print(f"  [Gemini Stats] Features Extracted in {elapsed:.3f}s")
            
            content = res_data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(content.strip())
            
            # Dynamic rate pacing to stay under Gemini 15 RPM (4.5s target)
            sleep_time = max(0.1, 4.5 - elapsed)
            time.sleep(sleep_time)
            
            return parsed
    except Exception as e:
        body = ""
        if hasattr(e, 'read'):
            try:
                body = e.read().decode('utf-8')
            except Exception:
                pass
        # Dynamic rate pacing on failure
        elapsed = time.time() - start_time
        sleep_time = max(0.1, 4.5 - elapsed)
        time.sleep(sleep_time)
        if body:
            raise RuntimeError(f"{e} | Body: {body}")
        raise e

def classify_prakriti_with_groq(subject_row, few_shot_text, api_key):
    """
    Query Groq Chat Completion API with demographic details and extracted physical features
    along with few-shot labeled examples to classify dominant or combination Prakriti.
    """
    subj_id = subject_row["id"]
    gender = "female" if str(subject_row["Sex"]).strip().upper() == "F" else "male"
    age = subject_row["Age"]
    
    # Decode one-hot fields
    sc = "Fair" if subject_row.get("SC_Fair") == 1 else ("Medium" if subject_row.get("SC_Medium") == 1 else ("Dark" if subject_row.get("SC_Dark") == 1 else "Unknown"))
    st = "Rough" if subject_row.get("ST_Rough") == 1 else ("Smooth" if subject_row.get("ST_Smooth") == 1 else ("Dry" if subject_row.get("ST_Dry") == 1 else ("Oily" if subject_row.get("ST_Oily") == 1 else "Unknown")))
    ht = "Thick" if subject_row.get("SH_Thick") == 1 else ("Thin" if subject_row.get("SH_Thin") == 1 else ("Med" if subject_row.get("SH_Med") == 1 else "Unknown"))
    hc = "Brown" if subject_row.get("SHC_Brown") == 1 else ("Black" if subject_row.get("SHC_Black") == 1 else "Unknown")
    mole = "Yes" if subject_row.get("Mole on skin") == 1 else "No"
    vv = "Low" if subject_row.get("VV_Low") == 1 else ("Med" if subject_row.get("VV_Med") == 1 else ("High" if subject_row.get("VV_High") == 1 else "Unknown"))
    pc = "PinkRed" if subject_row.get("PC_PinkRed") == 1 else ("PinkWhite" if subject_row.get("PC_PinkWhite") == 1 else ("Brown" if subject_row.get("PC_Brown") == 1 else "Unknown"))
    
    prompt = (
        "You are an expert Ayurvedic practitioner and researcher specializing in Prakriti classification (constitution analysis).\n"
        "Your task is to classify a subject's dominant or combination Prakriti (dosha constitution) into one of the categories:\n"
        "- Dominant: 'K' (Kapha), 'P' (Pitta), 'V' (Vata)\n"
        "- Combinations: 'K-V' (Kapha-Vata), 'K-P' (Kapha-Pitta), 'P-V' (Pitta-Vata), 'V-K' (Vata-Kapha), 'P-K' (Pitta-Kapha), 'V-P' (Vata-Pitta)\n\n"
        "Guidelines:\n"
        "1. Dominant Kapha (K) typically features smooth, oily, fair skin, low vein visibility, and thick hair.\n"
        "2. Dominant Pitta (P) typically features medium skin color, warm/oily texture, reddish/pinkish palm color, and medium hair/vein visibility.\n"
        "3. Dominant Vata (V) typically features rough, dry, dark skin, high vein visibility, thin hair, and brown/pale palm color.\n"
        "4. Most subjects are combination constitutions (e.g. K-P, K-V, P-V) with a primary and secondary dosha.\n\n"
        "Here are some labeled examples of subjects from the clinic:\n"
        f"{few_shot_text}\n\n"
        "Now, analyze the following subject's demographic details and extracted physical characteristics:\n"
        f"Subject ID: {subj_id}\n"
        f"Age: {age}\n"
        f"Gender: {gender}\n"
        f"Skin Color: {sc}\n"
        f"Skin Texture: {st}\n"
        f"Palm Color: {pc}\n"
        f"Vein Visibility: {vv}\n"
        f"Hair Thickness: {ht}\n"
        f"Hair Color: {hc}\n"
        f"Mole on Skin: {mole}\n\n"
        "Return a JSON object with these exact keys:\n"
        "- \"predicted_dosha\": The dominant or combination dosha code (e.g., \"K-V\", \"K-P\", \"P-V\", \"K\", \"P\", \"V\")\n"
        "- \"vata_percent\": The percentage of Vata in their constitution (Integer, 0 to 100)\n"
        "- \"pitta_percent\": The percentage of Pitta in their constitution (Integer, 0 to 100)\n"
        "- \"kapha_percent\": The percentage of Kapha in their constitution (Integer, 0 to 100)\n"
        "- \"classification_percent\": A formatted string representing the percentages, exactly like \"k-65%, p-15%, v-20%\"\n"
        "- \"reasoning\": A brief explanation (2-3 sentences) of your classification based on the physical features.\n\n"
        "CRITICAL RULE: You MUST directly output ONLY the JSON object. Do NOT output any chain-of-thought, thinking blocks, <think> tags, markdown code blocks, or preamble/postscript text. Output strictly raw JSON starting with '{' and ending with '}'."
    )
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": "qwen/qwen3.8-27b",
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "temperature": 0.0,
        "max_tokens": 1024
    }
    
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    )
    
    import time
    import re
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            
            # Print performance metrics
            usage = res_data.get("usage", {})
            total_time = usage.get("total_time", 0.0)
            total_tokens = usage.get("total_tokens", 0)
            print(f"  [Groq Stats] Prakriti Classified in {total_time:.3f}s (Tokens: {total_tokens})")
            
            content = res_data["choices"][0]["message"]["content"]
            
            # Strip out thinking blocks if present to avoid brace matching errors
            import re
            content_clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            if '<think>' in content_clean:
                content_clean = content_clean.split('</think>')[-1].strip()
            
            # Extract nested balanced brace blocks
            def find_json_blocks(text):
                blocks = []
                start = -1
                count = 0
                for i, char in enumerate(text):
                    if char == '{':
                        if count == 0:
                            start = i
                        count += 1
                    elif char == '}':
                        if count > 0:
                            count -= 1
                            if count == 0 and start != -1:
                                blocks.append(text[start:i+1])
                return blocks
                
            blocks = find_json_blocks(content_clean)
            parsed = None
            if blocks:
                # Try parsing from last to first
                for block in reversed(blocks):
                    try:
                        parsed = json.loads(block.strip())
                        break
                    except Exception:
                        pass
                        
            if parsed is None:
                raise ValueError("No valid balanced JSON block found in response.")
            
            # Dynamic rate pacing to stay under 30 RPM
            elapsed = time.time() - start_time
            sleep_time = max(0.1, 2.2 - elapsed)
            time.sleep(sleep_time)
            
            return parsed
    except Exception as e:
        print(f"  [Classifier Error] Groq Prakriti classification failed: {e}")
        # Dynamic rate pacing on failure
        elapsed = time.time() - start_time
        sleep_time = max(0.1, 2.2 - elapsed)
        time.sleep(sleep_time)
        raise e

def classify_prakriti_with_unified_ml(subj_id, subject_row, wrist_dataset_dir):
    """
    Classifies Prakriti using the unified 4-view CV + ML pipeline (RBF-SVM).
    Replaces Groq LLM hallucination with deterministic empirical predictions.
    """
    subj_folder_name = str(subj_id).lower()
    subj_path = Path(wrist_dataset_dir) / subj_folder_name
    if not subj_path.exists():
        for child in Path(wrist_dataset_dir).iterdir():
            if child.is_dir() and child.name.lower() == subj_folder_name:
                subj_path = child
                break
                
    filenames = {
        "palm_open_front": ["palm_open_front.jpg"],
        "palm_open_back": ["palm_open_back.jpg"],
        "palm_closed_front": ["palm_close_front.jpg", "palm_closed_front.jpg"],
        "palm_closed_back": ["palm_close_back.jpg", "palm_closed_back.jpg"],
    }
    
    images = {}
    for vk, candidates in filenames.items():
        found = None
        if subj_path.exists():
            for c in candidates:
                p = subj_path / c
                if p.exists():
                    found = p
                    break
        if found:
            try:
                images[vk] = Image.open(found).convert("RGB")
            except Exception:
                images[vk] = None
        else:
            images[vk] = None
            
    age = subject_row.get("Age", 35)
    sex = str(subject_row.get("Sex", "")).strip().upper()
    is_male = 1.0 if sex == "M" or "MALE" in sex else 0.0
    
    from unified_cv import extract_participant_features
    feats = extract_participant_features(images, age=age, sex_m=is_male).reshape(1, -1)
    
    model_path = PROJECT_ROOT / "mini" / "unified_prakriti_pipeline.joblib"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "mini" / "model_color.joblib"
    bundle = joblib.load(model_path)
    
    pipeline = bundle.get("pipeline") or bundle.get("model")
    probs = pipeline.predict_proba(feats)[0]
    classes = bundle["classes"]
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
    
    # Calculate integer percentages that sum to 100
    kapha_pct = int(round(prob_dict.get("K", 0.0) * 100))
    pitta_pct = int(round(prob_dict.get("P", 0.0) * 100))
    vata_pct = 100 - (kapha_pct + pitta_pct)
    if vata_pct < 0:
        vata_pct = 0
        
    ranked = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    top_dosha = ranked[0][0]
    sec_dosha = ranked[1][0] if len(ranked) > 1 else None
    
    # If secondary dosha is significant (>= 25%), format as combination (e.g. K-P, K-V, P-V)
    if sec_dosha and ranked[1][1] >= 0.25:
        combination_dosha = f"{top_dosha}-{sec_dosha}"
    else:
        combination_dosha = top_dosha
        
    class_pct = f"k-{kapha_pct}%, p-{pitta_pct}%, v-{vata_pct}%"
    reasoning = (
        f"Diagnosed using Unified CV + {bundle.get('model_name', 'RBF-SVM')} model. "
        f"Primary dosha: {top_dosha} ({prob_dict[top_dosha]*100:.1f}%), "
        f"Secondary: {sec_dosha} ({prob_dict.get(sec_dosha, 0.0)*100:.1f}%)."
    )
    
    return {
        "predicted_dosha": combination_dosha,
        "vata_percent": vata_pct,
        "pitta_percent": pitta_pct,
        "kapha_percent": kapha_pct,
        "classification_percent": class_pct,
        "reasoning": reasoning,
        "raw_probabilities": prob_dict
    }

def bifurcate_subject_images(subject_id, predicted_dosha, vata, pitta, kapha, wrist_dataset_dir, output_classified_dir):
    """
    Copies the subject's dataset folder to the target combination directory.
    """
    scores = {
        "V": vata,
        "P": pitta,
        "K": kapha
    }
    sorted_doshas = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top1 = sorted_doshas[0][0]
    top2 = sorted_doshas[1][0]
    
    top_set = {top1, top2}
    if top_set == {"K", "V"}:
        combination = "K-V"
    elif top_set == {"K", "P"}:
        combination = "K-P"
    elif top_set == {"P", "V"}:
        combination = "P-V"
    else:
        # Fallback if percentages are equal/missing
        if predicted_dosha == "K":
            combination = "K-P"
        elif predicted_dosha == "P":
            combination = "P-V"
        else:
            combination = "K-V"
            
    subj_folder_name = str(subject_id).lower()
    src_dir = Path(wrist_dataset_dir) / subj_folder_name
    if not src_dir.exists():
        for child in Path(wrist_dataset_dir).iterdir():
            if child.is_dir() and child.name.lower() == subj_folder_name:
                src_dir = child
                break
                
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory for {subject_id} not found in Wrist_Dataset")
        
    dst_base = Path(output_classified_dir) / combination
    dst_base.mkdir(parents=True, exist_ok=True)
    dst_dir = dst_base / src_dir.name
    
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
        
    shutil.copytree(src_dir, dst_dir)
    return combination

def run_classification_pipeline(xlsx_path, wrist_dataset_dir, output_classified_dir, progress_callback=None, subject_ids=None):
    """
    Loads unclassified subjects (optionally filtered by subject_ids), classifies features and Prakriti via Groq,
    updates Excel row cells using openpyxl (preserving images), updates Supabase,
    and copies images to classified combination directories.
    """
    import openpyxl
    xlsx_path = Path(xlsx_path)
    wrist_dataset_dir = Path(wrist_dataset_dir)
    output_classified_dir = Path(output_classified_dir)
    
    # 0. Backup original Excel file
    backup_path = xlsx_path.with_suffix(xlsx_path.suffix + ".bak")
    try:
        shutil.copy2(xlsx_path, backup_path)
        if progress_callback:
            progress_callback(f"Created a backup of the Excel file at: {backup_path.name}")
    except Exception as backup_err:
        if progress_callback:
            progress_callback(f"[Warning] Failed to create backup: {backup_err}")
            
    # 1. Load API Keys
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        try:
            from supabasehelper import get_groq_api_key
            groq_key = get_groq_api_key()
        except Exception:
            pass
            
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        try:
            from supabasehelper import get_gemini_api_key
            gemini_key = get_gemini_api_key()
        except Exception:
            pass
            
    if not gemini_key:
        raise ValueError("Gemini API Key is missing. Please set GEMINI_API_KEY in environment or secrets.toml")
        
    # 2. Load openpyxl workbook & sheet
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb["Sheet"]
    
    # Map headers to 1-based column indices
    header = [ws.cell(row=1, column=col).value for col in range(1, ws.max_column + 1)]
    hidx = {name: idx + 1 for idx, name in enumerate(header) if name is not None}
    
    if "Classification Percent" not in hidx:
        new_col = len(header) + 1
        ws.cell(row=1, column=new_col, value="Classification Percent")
        hidx["Classification Percent"] = new_col
        header.append("Classification Percent")
        
    # 3. Load subjects with pandas to check unclassified
    if progress_callback:
        progress_callback("Loading unclassified subjects from Excel...")
    df, unclassified_indices = load_unclassified_subjects(xlsx_path, subject_ids=subject_ids)
    
    if not unclassified_indices:
        if progress_callback:
            if subject_ids:
                progress_callback("No unclassified subjects matching the current batch found in Excel sheet.")
            else:
                progress_callback("No unclassified subjects found from Row 93 onwards in Excel sheet.")
        return []
        
    if progress_callback:
        progress_callback(f"Found {len(unclassified_indices)} unclassified subject(s). Generating few-shot context examples...")
        
    few_shot_text = generate_few_shot_examples(df)
    results = []
    
    # Ensure "Classification Percent" column exists in dataframe
    if "Classification Percent" not in df.columns:
        df["Classification Percent"] = ""
        
    for idx in unclassified_indices:
        row = df.iloc[idx]
        subj_id = row["id"]
        name = row["Name"] if not pd.isna(row["Name"]) else "Unknown"
        row_num = idx + 2  # 0-based pandas index corresponds to 2-based Excel row number
        
        if progress_callback:
            progress_callback(f"\nAnalyzing '{name}' (ID: {subj_id}) using Hybrid Models...")
            
        try:
            # Stage A: Extract visual features
            if progress_callback:
                progress_callback("  [Stage A] Extracting visual features from hand images using Gemini...")
            features = extract_visual_features_with_gemini(row, wrist_dataset_dir, gemini_key)
            
            # Log extracted features
            if progress_callback:
                progress_callback(f"    Extracted Features -> Skin Color: {features.get('skin_color')}, Skin Texture: {features.get('skin_texture')}, Palm Color: {features.get('palm_color')}, Veins: {features.get('vein_visibility')}")
                progress_callback(f"    Feature Extraction Reasoning: {features.get('reasoning')}")
                
            # Update one-hot columns in DataFrame and openpyxl ws
            def set_one_hot_fields(feature_key, selected_col, all_cols):
                for col in all_cols:
                    val = 1 if col == selected_col else 0
                    if col in df.columns:
                        df.loc[idx, col] = val
                    if col in hidx:
                        ws.cell(row=row_num, column=hidx[col], value=val)
 
            # Update Skin Color
            sc_val = features.get("skin_color")
            if sc_val in ["Fair", "Medium", "Dark"]:
                set_one_hot_fields("skin_color", f"SC_{sc_val}", ["SC_Fair", "SC_Medium", "SC_Dark"])
                
            # Update Skin Texture
            st_val = features.get("skin_texture")
            if st_val in ["Rough", "Smooth", "Dry", "Oily"]:
                set_one_hot_fields("skin_texture", f"ST_{st_val}", ["ST_Rough", "ST_Smooth", "ST_Dry", "ST_Oily"])
                
            # Update Hair Thickness
            ht_val = features.get("hair_thickness")
            if ht_val in ["Thick", "Thin", "Med"]:
                set_one_hot_fields("hair_thickness", f"SH_{ht_val}", ["SH_Thick", "SH_Thin", "SH_Med"])
                
            # Update Hair Color
            hc_val = features.get("hair_color")
            if hc_val in ["Brown", "Black"]:
                set_one_hot_fields("hair_color", f"SHC_{hc_val}", ["SHC_Brown", "SHC_Black"])
                
            # Update Mole on skin
            mole_val = features.get("mole_on_skin")
            if mole_val is not None:
                try:
                    m_int = int(mole_val)
                    df.loc[idx, "Mole on skin"] = m_int
                    if "Mole on skin" in hidx:
                        ws.cell(row=row_num, column=hidx["Mole on skin"], value=m_int)
                except ValueError:
                    pass
                    
            # Update Vein Visibility
            vv_val = features.get("vein_visibility")
            if vv_val in ["Low", "Med", "High"]:
                set_one_hot_fields("vein_visibility", f"VV_{vv_val}", ["VV_Low", "VV_Med", "VV_High"])
                
            # Update Palm Color
            pc_val = features.get("palm_color")
            if pc_val in ["PinkRed", "PinkWhite", "Brown"]:
                set_one_hot_fields("palm_color", f"PC_{pc_val}", ["PC_PinkRed", "PC_PinkWhite", "PC_Brown"])
 
            # Sync updated row to local row object for stage B call
            updated_row = df.iloc[idx]
            
            # Stage B: Predict Prakriti Combination via Unified CV + ML Model
            if progress_callback:
                progress_callback("  [Stage B] Classifying combination Prakriti using Unified CV + ML Model...")
            res = classify_prakriti_with_unified_ml(subj_id, updated_row, wrist_dataset_dir)
            
            predicted_dosha = res.get("predicted_dosha", "Unknown")
            vata = int(res.get("vata_percent", 0))
            pitta = int(res.get("pitta_percent", 0))
            kapha = int(res.get("kapha_percent", 0))
            class_pct = res.get("classification_percent", f"k-{kapha}%, p-{pitta}%, v-{vata}%")
            reasoning = res.get("reasoning", "")
            
            if progress_callback:
                progress_callback(f"    Unified ML Result -> Predicted Prakriti: {predicted_dosha} | Percentages: {class_pct}")
                progress_callback(f"    Diagnosis Rationale: {reasoning}")
                
            # Update Prakriti and Classification Percent columns in df and openpyxl ws
            df.loc[idx, "Prakriti"] = predicted_dosha
            df.loc[idx, "Classification Percent"] = class_pct
            
            if "Prakriti" in hidx:
                ws.cell(row=row_num, column=hidx["Prakriti"], value=predicted_dosha)
            if "Classification Percent" in hidx:
                ws.cell(row=row_num, column=hidx["Classification Percent"], value=class_pct)
            
            # Bifurcate images
            combination = bifurcate_subject_images(
                subj_id, predicted_dosha, vata, pitta, kapha,
                wrist_dataset_dir, output_classified_dir
            )
            if progress_callback:
                progress_callback(f"  [Bifurcation] Copied images to combination folder: {combination}/")
                
            # Update Supabase (Update if exists, otherwise Insert)
            try:
                probs = {
                    "Vata": vata / 100.0,
                    "Pitta": pitta / 100.0,
                    "Kapha": kapha / 100.0
                }
                # Confidence is primary dosha percentage divided by 100
                conf = max(vata, pitta, kapha) / 100.0
                db_res = update_wrist_data_prakriti(subj_id, predicted_dosha, probs, conf)
                if db_res["success"]:
                    if progress_callback:
                        progress_callback(f"  [Supabase] Updated record successfully in the cloud.")
                else:
                    # Fallback: Insert a new record
                    from supabasehelper import insert_wrist_data
                    gender_code = "Female" if "Female" in str(subj_id) or "female" in str(subj_id) or str(row.get("Sex", "")).strip().upper() == "F" else "Male"
                    ins_res = insert_wrist_data(subj_id, gender_code, probs, predicted_dosha, conf)
                    if ins_res["success"]:
                        if progress_callback:
                            progress_callback(f"  [Supabase] Created new record successfully in the cloud.")
                    else:
                        if progress_callback:
                            progress_callback(f"  [Supabase Warning] Cloud synchronization failed: {ins_res['error']}")
            except Exception as db_err:
                if progress_callback:
                    progress_callback(f"  [Supabase Error] Database operation failed: {db_err}")
                    
            # Save progress after each successful subject
            try:
                restore_all_images_in_sheet(ws, wrist_dataset_dir)
                wb.save(xlsx_path)
            except Exception as save_err:
                if progress_callback:
                    progress_callback(f"  [Spreadsheet Warning] Failed to auto-save progress: {save_err}")
                    
            results.append({
                "id": subj_id,
                "name": name,
                "predicted_dosha": predicted_dosha,
                "percentages": class_pct,
                "status": "Success",
                "error": ""
            })
            
        except Exception as e:
            err_msg = str(e)
            if hasattr(e, 'read'):
                try:
                    err_msg += f" | Body: {e.read().decode('utf-8')}"
                except Exception:
                    pass
            if progress_callback:
                progress_callback(f"  [ERROR] Classification failed for subject {subj_id}: {err_msg}")
            results.append({
                "id": subj_id,
                "name": name,
                "predicted_dosha": "Unknown",
                "percentages": "",
                "status": "Failed",
                "error": err_msg
            })
            
        # Pause to avoid rate limits between subjects
        time.sleep(2.0)
            
    # Save spreadsheet using openpyxl to keep all embedded images intact
    try:
        if progress_callback:
            progress_callback("\n[Spreadsheet] Restoring image drawings to prevent openpyxl deletion...")
        restore_all_images_in_sheet(ws, wrist_dataset_dir)
        wb.save(xlsx_path)
        if progress_callback:
            progress_callback(f"\n[Spreadsheet] Successfully saved and updated Excel dataset sheet: {xlsx_path.name}")
    except PermissionError:
        if progress_callback:
            progress_callback(f"\n[Spreadsheet Error] Permission denied when saving to Excel sheet: '{xlsx_path}'. Please make sure it is closed in Microsoft Excel.")
        raise PermissionError(f"Permission denied when saving to Excel sheet: '{xlsx_path}'. Close the file and click classify again.")
    except Exception as save_err:
        if progress_callback:
            progress_callback(f"\n[Spreadsheet Error] Failed to save Excel spreadsheet: {save_err}")
            
    return results
