import os
import re
import json
import urllib.request
import time

# Default fallback values
UNKNOWN = "Unknown"

# Standardized mappings
SKIN_TYPES = {
    "normal": "Smooth",
    "dry": "Dry",
    "rough": "Rough",
    "smooth": "Smooth",
    "oily": "Oily"
}

VEIN_TYPES = {
    "low": "Low",
    "deep": "Low",
    "invisible": "Low",
    "no": "Low",
    "not visible": "Low",
    "not": "Low",
    "none": "Low",
    "nil": "Low",
    "absent": "Low",
    "med": "Med",
    "medium": "Med",
    "mod": "Med",
    "moderate": "Med",
    "slightly visible": "Med",
    "slight": "Med",
    "somewhat": "Med",
    "high": "High",
    "prominent": "High",
    "visible": "High",
    "clearly visible": "High",
    "clear": "High",
    "yes": "High"
}

MOLE_TYPES = {
    "yes": "Yes",
    "y": "Yes",
    "1": "Yes",
    "present": "Yes",
    "no": "No",
    "n": "No",
    "0": "No",
    "none": "No",
    "absent": "No",
    "nil": "No"
}


def clean_text(text):
    """Remove special punctuation while preserving alphanumeric and whitespace."""
    return re.sub(r'[^\w\s-]', ' ', text).strip()


EXTRACTION_PROMPT = (
    "You are an expert clinical parser. Extract patient details and clinical observations from the following unstructured WhatsApp message text.\n\n"
    "CRITICAL Value Mappings (CRUCIAL: NEVER return 'Unknown' when the information is present):\n"
    "1. Vein Visibility:\n"
    "   - 'No', 'Not Visible', 'Invisible', 'None', 'Nil', 'Low' -> ALWAYS map to 'Low'\n"
    "   - 'Slightly visible', 'Medium', 'Med', 'Moderate' -> ALWAYS map to 'Med'\n"
    "   - 'Clearly visible', 'Prominent', 'Visible', 'High', 'Yes' -> ALWAYS map to 'High'\n"
    "2. Mole Visibility:\n"
    "   - 'No', 'No mole', 'None', '0', 'Nil' -> ALWAYS map to 'No'\n"
    "   - 'Yes', '1', 'Present' -> ALWAYS map to 'Yes'\n"
    "3. Skin Type / Texture:\n"
    "   - 'Normal', 'Smooth' -> ALWAYS map to 'Smooth'\n"
    "   - 'Dry' -> ALWAYS map to 'Dry'\n"
    "   - 'Rough' -> ALWAYS map to 'Rough'\n"
    "   - 'Oily' -> ALWAYS map to 'Oily'\n\n"
    "Note on user input order in WhatsApp:\n"
    "Messages typically contain patient details in this exact order:\n"
    "Line 1: Name\n"
    "Line 2: Age (e.g. 22)\n"
    "Line 3: Gender ('Male', 'Female', 'M', 'F')\n"
    "Line 4: Vein Visibility (e.g. 'No', 'Not Visible', 'clearly visible') -> MUST map to 'Low', 'Med', or 'High'\n"
    "Line 5: Mole Visibility (e.g. 'No', 'Yes') -> MUST map to 'No' or 'Yes'\n"
    "Line 6: Skin Type (e.g. 'Normal', 'Dry') -> MUST map to 'Smooth', 'Dry', etc.\n\n"
    "Return a strict JSON object with these exact keys:\n"
    "- \"name\": Full name string (or null if missing)\n"
    "- \"age\": Integer age (or null if missing)\n"
    "- \"gender\": \"male\", \"female\" (or null if missing)\n"
    "- \"vein_visibility\": \"Low\", \"Med\", \"High\" (or null if missing)\n"
    "- \"mole\": \"Yes\", \"No\" (or null if missing)\n"
    "- \"skin_texture\": \"Rough\", \"Smooth\", \"Dry\", \"Oily\" (or null if missing)\n"
    "- \"skin_color\": \"Fair\", \"Medium\", \"Dark\" (or null if missing)\n"
    "- \"palm_color\": \"PinkRed\", \"PinkWhite\", \"Brown\" (or null if missing)\n"
    "- \"prakriti\": Ground-truth dosha label if explicitly written (or null if missing)\n\n"
    "Few-Shot Examples:\n"
    "Example 1 (Unlabelled sequential lines):\n"
    "Input: \"Dhruvi devalia\\n22\\nFemale\\nNo\\nNo\\nNormal\"\n"
    "JSON: {\"name\": \"Dhruvi devalia\", \"age\": 22, \"gender\": \"female\", \"vein_visibility\": \"Low\", \"mole\": \"No\", \"skin_texture\": \"Smooth\"}\n\n"
    "Example 2 (Semicolons and labels):\n"
    "Input: \"Name: Abin Anto\\nAge: 21\\nGender: Male\\nVein Visibility; Not Visible\\nMole: Yes\\nSkin type: Normal\"\n"
    "JSON: {\"name\": \"Abin Anto\", \"age\": 21, \"gender\": \"male\", \"vein_visibility\": \"Low\", \"mole\": \"Yes\", \"skin_texture\": \"Smooth\"}\n\n"
    "Example 3 (Clearly visible vein):\n"
    "Input: \"Ayush Nair\\n19\\nMale\\nClearly Visible\\nNo mole\\nNormal\"\n"
    "JSON: {\"name\": \"Ayush Nair\", \"age\": 19, \"gender\": \"male\", \"vein_visibility\": \"High\", \"mole\": \"No\", \"skin_texture\": \"Smooth\"}\n\n"
    "Example 4 (Slightly visible vein and dry skin):\n"
    "Input: \"Meet Gangwal\\n22\\nMale\\nslightly visible\\nYes\\nDry\"\n"
    "JSON: {\"name\": \"Meet Gangwal\", \"age\": 22, \"gender\": \"male\", \"vein_visibility\": \"Med\", \"mole\": \"Yes\", \"skin_texture\": \"Dry\"}\n"
)


def run_gemini_extraction(text, api_key):
    """
    Queries Google Gemini 3.5 Flash Lite API to parse clinical message text.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={api_key}"
    prompt = f"{EXTRACTION_PROMPT}\nText to parse: \"{text}\""
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.0
        }
    }
    
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content = res_data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(content.strip())
            return format_extracted_dict(parsed)
    except Exception as e:
        print(f"  [Extractor] Gemini extraction error: {e}")
        return None


def run_groq_extraction(text, api_key):
    """
    Calls Groq Chat Completion API as a fast LLM extraction engine.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    prompt = f"{EXTRACTION_PROMPT}\nText to parse: \"{text}\""
    
    payload = {
        "model": "qwen/qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 250
    }
    
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content = res_data["choices"][0]["message"]["content"]
            
            # Clean thinking tags
            content_clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            s_idx = content_clean.find('{')
            e_idx = content_clean.rfind('}')
            if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
                json_str = content_clean[s_idx:e_idx+1]
            else:
                json_str = content_clean
                
            parsed = json.loads(json_str)
            return format_extracted_dict(parsed)
    except Exception as e:
        print(f"  [Extractor] Groq extraction error: {e}")
        return None


def format_extracted_dict(parsed):
    """Standardizes parsed dictionary into expected pipeline schema."""
    skin_tex = parsed.get("skin_texture") or UNKNOWN
    if str(skin_tex).lower() in ("normal", "smooth"):
        skin_tex = "Smooth"
    elif str(skin_tex).lower() == "dry":
        skin_tex = "Dry"

    mole_val = parsed.get("mole")
    if mole_val is not None:
        mole_str = str(mole_val).strip().lower()
        mole_clean = MOLE_TYPES.get(mole_str, "No" if mole_str in ("0", "no", "false") else "Yes")
    else:
        mole_clean = UNKNOWN

    vein_val = parsed.get("vein_visibility")
    if vein_val is not None:
        vein_str = str(vein_val).strip().lower()
        if vein_str in VEIN_TYPES:
            vein_clean = VEIN_TYPES[vein_str]
        else:
            vein_clean = str(vein_val).capitalize()
            for vk in sorted(VEIN_TYPES.keys(), key=len, reverse=True):
                if vk in vein_str:
                    vein_clean = VEIN_TYPES[vk]
                    break
    else:
        vein_clean = UNKNOWN

    age_val = parsed.get("age")
    try:
        age_int = int(age_val) if age_val not in (None, "", UNKNOWN) else UNKNOWN
    except (ValueError, TypeError):
        age_int = UNKNOWN

    gender_val = parsed.get("gender") or UNKNOWN
    if str(gender_val).lower() in ("m", "male"):
        gender_clean = "male"
    elif str(gender_val).lower() in ("f", "female"):
        gender_clean = "female"
    else:
        gender_clean = UNKNOWN

    return {
        "name": parsed.get("name") or UNKNOWN,
        "age": age_int,
        "gender": gender_clean,
        "skin_type": skin_tex,
        "skin_texture": skin_tex,
        "skin_color": parsed.get("skin_color") or UNKNOWN,
        "vein_visibility": vein_clean,
        "mole": mole_clean,
        "palm_color": parsed.get("palm_color") or UNKNOWN,
        "hair_thickness": parsed.get("hair_thickness") or UNKNOWN,
        "hair_color": parsed.get("hair_color") or UNKNOWN,
        "prakriti": parsed.get("prakriti") or UNKNOWN
    }


def run_heuristic_extraction(text):
    """
    Robust deterministic rule-based and positional parser.
    Handles explicit labels and the exact user ordering:
      [Name] [Age] [Gender] [Vein Visibility] [Mole Visibility] [Skin Type (Normal/Dry)]
    e.g.:
      - 'Anjali 24 F Low No Normal'
      - 'Rahul 28 M High Yes Dry'
      - 'Vein: Low, Mole: No, Skin: Normal, Name: Pooja, Age: 23, Gender: F'
    """
    lines = text.replace(',', ' ').splitlines()
    flat_text = " ".join(lines)
    
    age = UNKNOWN
    gender = UNKNOWN
    vein = UNKNOWN
    mole = UNKNOWN
    skin = UNKNOWN
    name = UNKNOWN

    # 1. Regex check for labeled fields
    m_age = re.search(r'\bage[:\s\-]*(\d+)\b', flat_text, re.IGNORECASE)
    if m_age:
        age = int(m_age.group(1))

    m_gen = re.search(r'\b(?:gender|sex)[:\s\-]*([a-zA-Z]+)\b', flat_text, re.IGNORECASE)
    if m_gen:
        g_tok = m_gen.group(1).lower()
        if g_tok in ('f', 'female'): gender = "female"
        elif g_tok in ('m', 'male'): gender = "male"

    m_vein = re.search(r'\bveins?(?:\s*visibility)?[:;\s\-]+([a-zA-Z\s]+?)(?:mole|skin|name|age|gender|\n|$)', flat_text, re.IGNORECASE)
    if m_vein:
        v_tok = m_vein.group(1).strip().lower()
        if v_tok in VEIN_TYPES:
            vein = VEIN_TYPES[v_tok]
        else:
            for vk in sorted(VEIN_TYPES.keys(), key=len, reverse=True):
                if vk in v_tok:
                    vein = VEIN_TYPES[vk]
                    break

    m_mole = re.search(r'\bmoles?(?:\s*visibility)?[:;\s\-]+([a-zA-Z0-9\s]+?)(?:vein|skin|name|age|gender|\n|$)', flat_text, re.IGNORECASE)
    if m_mole:
        m_tok = m_mole.group(1).strip().lower()
        if "no" in m_tok or "nil" in m_tok or "none" in m_tok or "0" in m_tok:
            mole = "No"
        elif "yes" in m_tok or "1" in m_tok or "present" in m_tok:
            mole = "Yes"
        elif m_tok in MOLE_TYPES:
            mole = MOLE_TYPES[m_tok]

    m_skin = re.search(r'\bskin(?:\s*type|\s*texture)?[:;\s\-]+([a-zA-Z\s]+?)(?:vein|mole|name|age|gender|\n|$)', flat_text, re.IGNORECASE)
    if m_skin:
        s_tok = m_skin.group(1).strip().lower()
        if s_tok in SKIN_TYPES:
            skin = SKIN_TYPES[s_tok]
        else:
            for sk, sv in SKIN_TYPES.items():
                if sk in s_tok:
                    skin = sv
                    break

    m_name = re.search(r'\bname[:\s\-]*([a-zA-Z\s]+?)(?:age|gender|sex|vein|mole|skin|\n|$)', flat_text, re.IGNORECASE)
    if m_name:
        name = m_name.group(1).strip()

    # 2. Positional parsing fallback
    tokens = clean_text(flat_text).split()
    used_indices = set()

    # Find Age and Gender token indices
    age_idx = -1
    gen_idx = -1

    for idx, tok in enumerate(tokens):
        # Shorthand 25F or 22M
        m = re.match(r'^(\d+)([fFmM])$', tok)
        if m:
            age = int(m.group(1))
            gender = "female" if m.group(2).lower() == 'f' else "male"
            age_idx = idx
            gen_idx = idx
            used_indices.add(idx)
            break

    if age == UNKNOWN:
        for idx, tok in enumerate(tokens):
            if tok.isdigit() and 1 <= int(tok) <= 120:
                age = int(tok)
                age_idx = idx
                used_indices.add(idx)
                break

    if gender == UNKNOWN:
        for idx, tok in enumerate(tokens):
            if idx in used_indices: continue
            tok_l = tok.lower()
            if tok_l in ('f', 'female'):
                gender = "female"
                gen_idx = idx
                used_indices.add(idx)
                break
            elif tok_l in ('m', 'male'):
                gender = "male"
                gen_idx = idx
                used_indices.add(idx)
                break

    # Check trailing tokens in exact order [Vein Visibility] [Mole Visibility] [Skin Type]
    n_tok = len(tokens)
    for idx in range(n_tok - 1, -1, -1):
        if idx in used_indices: continue
        tok_l = tokens[idx].lower()
        if skin == UNKNOWN and tok_l in SKIN_TYPES:
            skin = SKIN_TYPES[tok_l]
            used_indices.add(idx)
            # Preceding token is likely Mole Visibility
            if idx - 1 >= 0 and idx - 1 not in used_indices and mole == UNKNOWN:
                m_tok = tokens[idx - 1].lower()
                if m_tok in MOLE_TYPES:
                    mole = MOLE_TYPES[m_tok]
                    used_indices.add(idx - 1)
                    # Preceding token is likely Vein Visibility
                    if idx - 2 >= 0 and idx - 2 not in used_indices and vein == UNKNOWN:
                        two_words = f"{tokens[idx - 3]} {tokens[idx - 2]}".lower() if idx - 3 >= 0 and idx - 3 not in used_indices else ""
                        if two_words in VEIN_TYPES:
                            vein = VEIN_TYPES[two_words]
                            used_indices.add(idx - 2)
                            used_indices.add(idx - 3)
                        else:
                            v_tok = tokens[idx - 2].lower()
                            if v_tok in VEIN_TYPES:
                                vein = VEIN_TYPES[v_tok]
                                used_indices.add(idx - 2)
            break

    # General fallback for any remaining standalone vein/mole/skin (check 2-word phrases first)
    for idx in range(len(tokens) - 1):
        if idx in used_indices or idx + 1 in used_indices: continue
        bigram = f"{tokens[idx]} {tokens[idx+1]}".lower()
        if vein == UNKNOWN and bigram in VEIN_TYPES:
            vein = VEIN_TYPES[bigram]
            used_indices.add(idx)
            used_indices.add(idx + 1)
        elif mole == UNKNOWN and bigram in MOLE_TYPES:
            mole = MOLE_TYPES[bigram]
            used_indices.add(idx)
            used_indices.add(idx + 1)

    for idx, tok in enumerate(tokens):
        if idx in used_indices: continue
        tok_l = tok.lower()
        if vein == UNKNOWN and tok_l in VEIN_TYPES:
            vein = VEIN_TYPES[tok_l]
            used_indices.add(idx)
        elif mole == UNKNOWN and tok_l in MOLE_TYPES:
            mole = MOLE_TYPES[tok_l]
            used_indices.add(idx)
        elif skin == UNKNOWN and tok_l in SKIN_TYPES:
            skin = SKIN_TYPES[tok_l]
            used_indices.add(idx)

    # Extract name from unused non-metadata tokens
    if name == UNKNOWN:
        earliest_meta_idx = min([i for i in [age_idx, gen_idx] if i != -1] or [len(tokens)])
        if earliest_meta_idx > 0:
            name_cand_tokens = [tokens[i] for i in range(earliest_meta_idx) if i not in used_indices]
        else:
            # Age/gender was at the start, name is between age/gender and vein/mole/skin
            name_cand_tokens = [t for i, t in enumerate(tokens) if i not in used_indices]

        # Filter out label keywords
        clean_name_tokens = [
            t for t in name_cand_tokens
            if t.lower() not in ('name', 'age', 'gender', 'sex', 'patient', 'sub', 'subject')
        ]
        if clean_name_tokens:
            name = " ".join(clean_name_tokens).strip()

    return {
        "name": name,
        "age": age,
        "gender": gender,
        "skin_type": skin if skin != UNKNOWN else "Smooth",
        "skin_texture": skin if skin != UNKNOWN else "Smooth",
        "skin_color": UNKNOWN,
        "vein_visibility": vein if vein != UNKNOWN else "Low",
        "mole": mole if mole != UNKNOWN else "No",
        "palm_color": UNKNOWN,
        "hair_thickness": UNKNOWN,
        "hair_color": UNKNOWN,
        "prakriti": UNKNOWN
    }


def extract_details(text):
    """
    Primary API entry point for details extraction.
    Tries Gemini API if GEMINI_API_KEY is found, then Groq API if GROQ_API_KEY is found,
    and falls back to deterministic heuristic parsing.
    """
    if not text or not text.strip():
        return format_extracted_dict({})
        
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        try:
            from supabasehelper import get_gemini_api_key
            gemini_key = get_gemini_api_key()
        except Exception:
            pass

    if gemini_key:
        res = run_gemini_extraction(text, gemini_key)
        if res:
            return res

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        try:
            from supabasehelper import get_groq_api_key
            groq_key = get_groq_api_key()
        except Exception:
            pass

    if groq_key:
        res = run_groq_extraction(text, groq_key)
        if res:
            return res

    return run_heuristic_extraction(text)
