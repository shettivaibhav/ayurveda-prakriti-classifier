import os
import json
import base64
import urllib.request
from pathlib import Path

# Categories we expect to assign
CATEGORIES = [
    "palm_open_front",
    "palm_open_back",
    "palm_closed_front",
    "palm_closed_back"
]

# Keywords to detect views from filenames (case-insensitive)
KEYWORD_MAPPINGS = {
    "palm_open_front": ["open_front", "openfront", "frontopen", "front_open", "open_palm"],
    "palm_open_back": ["open_back", "openback", "backopen", "back_open", "open_skin", "back_palm"],
    "palm_closed_front": ["close_front", "closed_front", "closefront", "closedfront", "close_palm"],
    "palm_closed_back": ["close_back", "closed_back", "closeback", "closedback", "close_skin", "fist"]
}

def run_gemini_view_classification(image_paths, api_key):
    """
    Sends base64 image data to the Gemini Multimodal API to classify the views.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={api_key}"
    
    parts = []
    # Add text instructions
    prompt = (
        "You are a computer vision classifier. Classify each of the attached images into one of the following 4 categories:\n"
        "1. 'palm_open_front' (open palm, inner front view)\n"
        "2. 'palm_open_back' (back of the open hand, outer skin view)\n"
        "3. 'palm_closed_front' (closed fist, inner/palm-side view)\n"
        "4. 'palm_closed_back' (closed fist, outer/back-side view)\n\n"
        "Return a JSON object mapping the exact filenames to their predicted category. "
        "Each filename must map to exactly one category from the list above. No other text."
    )
    parts.append({"text": prompt})
    
    # Encode and append each image (compressed to 512x512 to prevent timeouts and rate-limit spikes)
    from PIL import Image as PILImage
    import io
    for img_path in image_paths:
        p = Path(img_path)
        if not p.exists():
            continue
        try:
            with PILImage.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((512, 512))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=75)
                img_data = base64.b64encode(buf.getvalue()).decode("utf-8")
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": img_data
                }
            })
            parts.append({"text": f"Above image filename is '{p.name}'"})
        except Exception as e:
            print(f"  [ViewClassifier] Error loading image {p.name}: {e}")
            
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
            
            # Print performance metrics
            elapsed = time.time() - start_time
            print(f"  [Gemini Stats] Views Classified in {elapsed:.3f}s")
            
            content = res_data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(content.strip())
            
            # Dynamic rate pacing to stay under Gemini 15 RPM (4.5s target)
            sleep_time = max(0.1, 4.5 - elapsed)
            time.sleep(sleep_time)
            
            return parsed
    except Exception as e:
        print(f"  [ViewClassifier] Gemini multimodal check failed: {e}. Falling back to name matching.")
        # Dynamic rate pacing on failure
        elapsed = time.time() - start_time
        sleep_time = max(0.1, 4.5 - elapsed)
        time.sleep(sleep_time)
        return None
 
def run_local_view_classification(image_paths):
    """
    Local fallback classifier: matches by filename keywords first,
    then assigns remaining images chronologically.
    """
    assigned = {}
    remaining_categories = list(CATEGORIES)
    unassigned_images = []
    
    # 1. Match based on filename keywords
    for img_path in image_paths:
        filename = Path(img_path).name.lower()
        matched = False
        for category, keywords in KEYWORD_MAPPINGS.items():
            if category in remaining_categories:
                if any(kw in filename for kw in keywords):
                    assigned[img_path] = category
                    remaining_categories.remove(category)
                    matched = True
                    break
        if not matched:
            unassigned_images.append(img_path)
            
    # 2. Chronologically assign remaining images to remaining categories
    # Sort unassigned by filename to keep it deterministic
    unassigned_images = sorted(unassigned_images)
    for img_path in unassigned_images:
        if remaining_categories:
            cat = remaining_categories.pop(0)
            assigned[img_path] = cat
        else:
            # More images than expected categories, assign None
            assigned[img_path] = None
            
    return assigned
 
def classify_hand_views(image_paths):
    """
    Main entry point for view classification.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            from supabasehelper import get_gemini_api_key
            api_key = get_gemini_api_key()
        except Exception:
            pass
            
    if api_key:
        res = run_gemini_view_classification(image_paths, api_key)
        if res:
            # Build full image path mapping from filenames returned by LLM
            mapping = {}
            for path in image_paths:
                p = Path(path)
                category = res.get(p.name)
                if category in CATEGORIES:
                    mapping[str(p)] = category
                else:
                    # Fallback to local match for this file if missing/malformed in LLM response
                    mapping[str(p)] = None
            
            # Ensure no duplicates or unassigned categories exist
            unassigned_paths = [p for p, c in mapping.items() if c is None]
            used_categories = [c for c in mapping.values() if c is not None]
            remaining_categories = [c for c in CATEGORIES if c not in used_categories]
            for path in unassigned_paths:
                if remaining_categories:
                    mapping[path] = remaining_categories.pop(0)
            return mapping
            
    return run_local_view_classification(image_paths)
