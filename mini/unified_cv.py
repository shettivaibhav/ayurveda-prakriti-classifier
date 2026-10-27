"""Unified Computer Vision Feature Extractor for Ayurvedic Prakriti Classification.

Extracts consistent anatomical Region-of-Interest (ROI) colour, texture, and statistical
biomarkers across all 4 hand/wrist views:
  1. palm_open_front:   Central palm region (above carpal crease)
  2. palm_open_back:    Dorsal hand skin (between knuckles and wrist)
  3. palm_closed_front: Flexed wrist/palm region
  4. palm_closed_back:  Dorsal wrist/knuckle region

Extracts:
  A. 70-D Continuous Anatomical Vector:
     - 4 views x 17 features = 68 CV features (RGB, LAB, Chroma, Redness moments)
     - Age (1) + Sex_M (1) = 2 Demographic features
  B. 7 Physical Clinical Diagnostic Features (defined in clinical master dataset):
     1. Skin Color: Fair / Medium / Dark
     2. Skin Texture: Rough / Smooth / Dry / Oily
     3. Skin Hair Thickness: Thick / Thin / Med
     4. Skin Hair Color: Brown / Black / None
     5. Mole on Skin: No (0) / Yes (1)
     6. Vein Visibility: Low / Med / High
     7. Palm Color: PinkRed / PinkWhite / Brown
  C. 19-D One-Hot Clinical Vector matching palm_dataset_26 June 26.xlsx
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from scipy.ndimage import laplace, sobel
from skimage.color import rgb2lab


VIEW_KEYS = [
    "palm_open_front",
    "palm_open_back",
    "palm_closed_front",
    "palm_closed_back"
]

# Anatomical patch centering parameters (cy_frac, half_size_frac)
PATCH_SPECS = {
    "palm_open_front": (0.40, 0.16),    # Upper central palm
    "palm_open_back": (0.50, 0.16),     # Dorsal hand skin centre
    "palm_closed_front": (0.45, 0.16),  # Flexed wrist/palm
    "palm_closed_back": (0.50, 0.16)    # Dorsal wrist centre
}

STAT_NAMES = [
    "R_mean", "G_mean", "B_mean",
    "R_std", "G_std", "B_std",
    "L_mean", "a_mean", "b_mean",
    "L_std", "a_std", "b_std",
    "cr_mean", "cg_mean", "cb_mean",
    "redness_rg", "redness_ab"
]

FEATURE_NAMES = [
    f"{view}_{stat}" for view in VIEW_KEYS for stat in STAT_NAMES
] + ["Age", "Sex_M"]

# 19 binary clinical column names matching the master Excel sheet
CLINICAL_19_COLS = [
    "SC_Fair", "SC_Medium", "SC_Dark",
    "ST_Rough", "ST_Smooth", "ST_Dry", "ST_Oily",
    "SH_Thick", "SH_Thin", "SH_Med",
    "SHC_Brown", "SHC_Black",
    "Mole on skin",
    "VV_Low", "VV_Med", "VV_High",
    "PC_PinkRed", "PC_PinkWhite", "PC_Brown"
]


def validate_image_quality(
    pil_img: Image.Image,
    min_size: int = 64,
    min_laplacian_var: float = 15.0
) -> Dict[str, Union[bool, float, List[str]]]:
    """
    Validates image resolution, lighting/brightness, and sharpness.
    Returns:
        {
            'valid': bool,
            'blur_score': float,
            'mean_intensity': float,
            'issues': list[str]
        }
    """
    issues = []
    w, h = pil_img.size
    
    if w < min_size or h < min_size:
        issues.append(f"Resolution too low ({w}x{h} < {min_size}x{min_size})")

    # Convert to grayscale numpy float array
    gray = np.asarray(pil_img.convert("L")).astype(float)
    mean_intensity = float(gray.mean())

    if mean_intensity < 15.0:
        issues.append(f"Image underexposed / too dark (mean: {mean_intensity:.1f})")
    elif mean_intensity > 240.0:
        issues.append(f"Image overexposed / too bright (mean: {mean_intensity:.1f})")

    # Sharpness via variance of discrete Laplacian
    lap = laplace(gray)
    blur_score = float(lap.var())

    if blur_score < min_laplacian_var:
        issues.append(f"Image may be blurry (Laplacian variance: {blur_score:.1f} < {min_laplacian_var})")

    return {
        "valid": len(issues) == 0,
        "blur_score": blur_score,
        "mean_intensity": mean_intensity,
        "issues": issues
    }


def normalize_illumination(arr_rgb: np.ndarray) -> np.ndarray:
    """
    Applies adaptive Gray-World illumination normalization to correct for
    uncalibrated phone lighting casts (yellow indoor incandescent vs cool outdoor daylight).
    Input: arr_rgb with values in [0, 1] of shape (H, W, 3) or (N, 3).
    """
    flat = arr_rgb.reshape(-1, 3)
    if flat.shape[0] == 0:
        return arr_rgb

    # Compute mean intensity per channel
    mean_ch = np.mean(flat, axis=0) + 1e-6
    gray_target = np.mean(mean_ch)

    # Scale gains constrained to [0.75, 1.35] to avoid over-amplifying noise
    gains = np.clip(gray_target / mean_ch, 0.75, 1.35)
    normalized = np.clip(flat * gains, 0.0, 1.0)
    return normalized.reshape(arr_rgb.shape)


def extract_view_roi_patch(
    pil_img: Image.Image,
    cy_frac: float = 0.50,
    half_frac: float = 0.16,
    max_patch_dim: int = 256
) -> np.ndarray:
    """
    Crops the anatomical patch ROI from an image and returns pixel array of shape (N, 3) in [0, 1].
    Efficiently crops using PIL before array conversion, avoiding massive full-image allocations.
    """
    w, h = pil_img.size
    cx = w // 2
    cy = int(h * cy_frac)
    hw = max(4, int(w * half_frac))
    hh = max(4, int(h * half_frac))

    y0 = max(0, cy - hh)
    y1 = min(h, cy + hh)
    x0 = max(0, cx - hw)
    x1 = min(w, cx + hw)

    # Crop on PIL image directly (fast and memory-light)
    patch_img = pil_img.crop((x0, y0, x1, y1)).convert("RGB")
    pw, ph = patch_img.size
    if pw > max_patch_dim or ph > max_patch_dim:
        patch_img = patch_img.resize((min(pw, max_patch_dim), min(ph, max_patch_dim)), Image.BILINEAR)

    im = np.asarray(patch_img).astype(float)
    patch = im.reshape(-1, 3) / 255.0
    return patch


def compute_patch_features(patch: np.ndarray) -> np.ndarray:
    """
    Computes 17 colour, texture, and statistical descriptors from an (N, 3) normalized RGB patch:
      - RGB means (3) & standard deviations (3)
      - CIELAB L*, a*, b* means (3) & standard deviations (3)
      - Normalized chroma means r, g, b (3)
      - Redness metrics: (R - G) and a*/(b* + eps) (2)
    """
    if patch.shape[0] == 0:
        return np.zeros(17, dtype=float)

    # RGB statistics
    rgb_mean = patch.mean(axis=0)
    rgb_std = patch.std(axis=0)

    # CIELAB conversion and statistics
    lab_pixels = rgb2lab(patch.reshape(-1, 1, 3)).reshape(-1, 3)
    lab_mean = lab_pixels.mean(axis=0)
    lab_std = lab_pixels.std(axis=0)

    # Normalized chroma
    sums = patch.sum(axis=1, keepdims=True) + 1e-6
    chroma = patch / sums
    chroma_mean = chroma.mean(axis=0)

    # Redness indicators
    redness_rg = float((patch[:, 0] - patch[:, 1]).mean())
    # a*/b* ratio (positive red vs yellow)
    b_denom = np.where(np.abs(lab_pixels[:, 2]) < 1e-3, 1e-3, lab_pixels[:, 2])
    redness_ab = float(np.clip(lab_pixels[:, 1] / b_denom, -10.0, 10.0).mean())

    feats = np.concatenate([
        rgb_mean,
        rgb_std,
        lab_mean,
        lab_std,
        chroma_mean,
        [redness_rg, redness_ab]
    ])
    return feats.astype(float)


def extract_view_features(pil_img: Image.Image, view_key: str) -> np.ndarray:
    """
    Extracts 17 features from a single view's PIL Image using its anatomical patch spec.
    """
    cy_frac, half_frac = PATCH_SPECS.get(view_key, (0.50, 0.16))
    patch = extract_view_roi_patch(pil_img, cy_frac=cy_frac, half_frac=half_frac)
    return compute_patch_features(patch)


def extract_participant_features(
    view_images: Dict[str, Optional[Image.Image]],
    age: Union[int, float],
    sex_m: Union[int, float, str]
) -> np.ndarray:
    """
    Aggregates features from all 4 views plus demographics into a 70-D participant vector.
    """
    # Parse sex flag
    if isinstance(sex_m, str):
        is_male = 1.0 if sex_m.strip().upper().startswith("M") else 0.0
    else:
        is_male = float(sex_m)

    age_val = float(age) if age not in (None, "Unknown", "") else 35.0

    view_feats_list = []
    valid_feats = []

    for vk in VIEW_KEYS:
        img = view_images.get(vk)
        if img is not None:
            try:
                feats = extract_view_features(img, vk)
                view_feats_list.append(feats)
                valid_feats.append(feats)
                continue
            except Exception:
                pass
        view_feats_list.append(None)

    # Impute any missing views with the mean of available views if available
    if len(valid_feats) > 0:
        fallback = np.mean(valid_feats, axis=0)
    else:
        fallback = np.zeros(17, dtype=float)

    imputed_view_feats = [
        f if f is not None else fallback for f in view_feats_list
    ]

    all_feats = np.concatenate(imputed_view_feats + [[age_val, is_male]])
    return all_feats


def extract_cv_clinical_markers(view_images: Dict[str, Optional[Image.Image]]) -> dict:
    """
    Extracts all 7 physical clinical biomarkers from wrist photographs using robust Computer Vision:
      1. skin_color: Fair / Medium / Dark
      2. skin_texture: Rough / Smooth / Dry / Oily
      3. skin_hair_thickness: Thick / Thin / Med
      4. skin_hair_color: Brown / Black / None
      5. mole: Yes / No
      6. vein_visibility: Low / Med / High
      7. palm_color: PinkRed / PinkWhite / Brown

    Also produces the 19-dimensional one-hot binary vector matching the master Excel format.
    """
    sc_img = view_images.get("palm_open_back") or view_images.get("palm_closed_back")
    pc_img = view_images.get("palm_open_front") or view_images.get("palm_closed_front")
    wrist_close_back = view_images.get("palm_closed_back")

    # Defaults / Priors
    skin_color = "Medium"
    skin_texture = "Smooth"
    hair_thickness = "Thin"
    hair_color = "Brown"
    mole = "No"
    vein_visibility = "Low"
    palm_color = "PinkWhite"

    confidences = {}

    # -------------------------------------------------------------
    # 1. Skin Color, Veins, Texture, Moles & Hair from Dorsal Wrist
    # -------------------------------------------------------------
    if sc_img is not None:
        try:
            w, h = sc_img.size
            cx, cy = w // 2, int(h * 0.50)
            hw, hh = max(8, int(w * 0.22)), max(8, int(h * 0.22))
            crop_sc = sc_img.crop((max(0, cx - hw), max(0, cy - hh), min(w, cx + hw), min(h, cy + hh))).convert("RGB")
            
            arr_sc = np.asarray(crop_sc).astype(float) / 255.0
            norm_sc = normalize_illumination(arr_sc)

            # CIELAB conversion
            lab_sc = rgb2lab(norm_sc)
            L_sc = float(lab_sc[:, :, 0].mean())
            a_sc = float(lab_sc[:, :, 1].mean())
            b_sc = float(lab_sc[:, :, 2].mean())

            # Individual Typology Angle (ITA) = arctan((L* - 50) / b*) * (180 / pi)
            # The dermatological gold standard for objective skin classification
            ita = np.arctan2(L_sc - 50.0, max(1.0, b_sc)) * (180.0 / np.pi)

            # 1. Skin Color Classification
            if ita >= 28.0 or L_sc >= 62.0:
                skin_color = "Fair"
                confidences["skin_color"] = min(0.95, 0.70 + (L_sc - 62.0) * 0.02)
            elif ita <= 10.0 or L_sc <= 48.0:
                skin_color = "Dark"
                confidences["skin_color"] = min(0.95, 0.70 + (48.0 - L_sc) * 0.02)
            else:
                skin_color = "Medium"
                confidences["skin_color"] = 0.85

            # 2. Vein Visibility (venous absorption contrast in Green vs Red & dorsal gradients)
            g_chan = norm_sc[:, :, 1]
            r_chan = norm_sc[:, :, 0]
            vein_diff = np.abs(r_chan - g_chan)
            vein_contrast = float(vein_diff.std())

            # Also check flexed wrist view if present for prominent superficial veins
            if wrist_close_back is not None:
                try:
                    w_cb, h_cb = wrist_close_back.size
                    crop_cb = wrist_close_back.crop((int(w_cb * 0.3), int(h_cb * 0.3), int(w_cb * 0.7), int(h_cb * 0.7))).convert("RGB")
                    arr_cb = np.asarray(crop_cb).astype(float) / 255.0
                    vein_contrast = max(vein_contrast, float(np.abs(arr_cb[:, :, 0] - arr_cb[:, :, 1]).std()))
                except Exception:
                    pass

            if vein_contrast >= 0.048:
                vein_visibility = "High"
                confidences["vein_visibility"] = 0.88
            elif vein_contrast >= 0.034:
                vein_visibility = "Med"
                confidences["vein_visibility"] = 0.78
            else:
                vein_visibility = "Low"
                confidences["vein_visibility"] = 0.85

            # 3. Skin Texture & Type
            # Evaluates high-frequency roughness energy and specular gloss highlights
            gray_sc = np.dot(norm_sc[..., :3], [0.2989, 0.5870, 0.1140])
            lap_energy = float(np.abs(laplace(gray_sc)).mean())
            
            # Specular highlight fraction (saturated bright regions indicating oily/glossy surface)
            specular_mask = (norm_sc[:, :, 0] > 0.88) & (norm_sc[:, :, 1] > 0.86) & (norm_sc[:, :, 2] > 0.82)
            specular_ratio = float(np.mean(specular_mask))

            if specular_ratio > 0.045:
                skin_texture = "Oily"
                confidences["skin_texture"] = 0.82
            elif lap_energy >= 0.038:
                skin_texture = "Rough"
                confidences["skin_texture"] = 0.80
            elif lap_energy >= 0.024:
                skin_texture = "Dry"
                confidences["skin_texture"] = 0.78
            else:
                skin_texture = "Smooth"
                confidences["skin_texture"] = 0.86

            # 4. Mole Detection (localized high-contrast dark pigment spots)
            L_map = lab_sc[:, :, 0]
            L_local_diff = L_map.mean() - L_map
            dark_spots = (L_local_diff > 20.0) & (lab_sc[:, :, 0] < 42.0)
            spot_count = int(np.sum(dark_spots))
            
            # True moles typically occupy 12 to 380 contiguous or localized pixels at this scale
            if 12 <= spot_count <= 420:
                mole = "Yes"
                confidences["mole"] = 0.84
            else:
                mole = "No"
                confidences["mole"] = 0.90

            # 5. Skin Hair Thickness & Color on Dorsal Hand/Forearm
            # Directional Sobel filters detect hair shafts on dorsal skin
            dx = sobel(gray_sc, axis=0)
            dy = sobel(gray_sc, axis=1)
            grad_mag = np.hypot(dx, dy)
            hair_ridge_ratio = float(np.mean(grad_mag > 0.18))

            if hair_ridge_ratio >= 0.045:
                hair_thickness = "Thick"
                confidences["hair_thickness"] = 0.80
            elif hair_ridge_ratio >= 0.018:
                hair_thickness = "Med"
                confidences["hair_thickness"] = 0.75
            else:
                hair_thickness = "Thin"
                confidences["hair_thickness"] = 0.82

            # Hair color from detected fine edge pixels
            if hair_ridge_ratio >= 0.012:
                hair_mask = grad_mag > 0.16
                hair_L = float(L_map[hair_mask].mean()) if np.any(hair_mask) else 35.0
                if hair_L < 32.0:
                    hair_color = "Black"
                else:
                    hair_color = "Brown"
                confidences["hair_color"] = 0.78
            else:
                hair_color = "Brown"
                confidences["hair_color"] = 0.70

        except Exception as e:
            pass

    # -------------------------------------------------------------
    # 2. Palm Color from Open Palm View
    # -------------------------------------------------------------
    if pc_img is not None:
        try:
            w, h = pc_img.size
            cx, cy = w // 2, int(h * 0.40)
            hw, hh = max(8, int(w * 0.18)), max(8, int(h * 0.18))
            crop_pc = pc_img.crop((max(0, cx - hw), max(0, cy - hh), min(w, cx + hw), min(h, cy + hh))).convert("RGB")
            
            arr_pc = np.asarray(crop_pc).astype(float) / 255.0
            norm_pc = normalize_illumination(arr_pc)

            lab_pc = rgb2lab(norm_pc)
            L_pc = float(lab_pc[:, :, 0].mean())
            a_pc = float(lab_pc[:, :, 1].mean())
            b_pc = float(lab_pc[:, :, 2].mean())

            # Erythema vs Melanin ratio
            if a_pc >= 13.5 or (b_pc > 0 and (a_pc / b_pc) >= 0.95):
                palm_color = "PinkRed"
                confidences["palm_color"] = 0.88
            elif L_pc >= 58.0:
                palm_color = "PinkWhite"
                confidences["palm_color"] = 0.85
            else:
                palm_color = "Brown"
                confidences["palm_color"] = 0.82
        except Exception:
            pass

    # Build the 19-dimensional one-hot binary vector
    binary_dict = encode_clinical_features({
        "skin_color": skin_color,
        "skin_texture": skin_texture,
        "hair_thickness": hair_thickness,
        "hair_color": hair_color,
        "mole": mole,
        "vein_visibility": vein_visibility,
        "palm_color": palm_color
    })

    return {
        "skin_color": skin_color,
        "skin_texture": skin_texture,
        "skin_type": skin_texture,  # alias for backward compatibility
        "hair_thickness": hair_thickness,
        "hair_color": hair_color,
        "mole": mole,
        "vein_visibility": vein_visibility,
        "palm_color": palm_color,
        "confidences": confidences,
        "binary_19d": binary_dict
    }


def encode_clinical_features(markers: dict) -> Dict[str, float]:
    """
    Encodes the 7 clinical diagnostic features into the exact 19 one-hot binary
    features defined in the clinical master dataset (palm_dataset_26 June 26.xlsx).
    """
    sc = str(markers.get("skin_color", "Medium")).lower()
    st = str(markers.get("skin_texture") or markers.get("skin_type", "Smooth")).lower()
    sh = str(markers.get("hair_thickness", "Thin")).lower()
    shc = str(markers.get("hair_color", "Brown")).lower()
    mole = str(markers.get("mole", "No")).lower()
    vv = str(markers.get("vein_visibility", "Low")).lower()
    pc = str(markers.get("palm_color", "PinkWhite")).lower()

    vec = {col: 0.0 for col in CLINICAL_19_COLS}

    # 1. Skin Color
    if "fair" in sc:
        vec["SC_Fair"] = 1.0
    elif "dark" in sc:
        vec["SC_Dark"] = 1.0
    else:
        vec["SC_Medium"] = 1.0

    # 2. Skin Texture
    if "rough" in st:
        vec["ST_Rough"] = 1.0
    elif "dry" in st:
        vec["ST_Dry"] = 1.0
    elif "oily" in st:
        vec["ST_Oily"] = 1.0
    else:
        vec["ST_Smooth"] = 1.0

    # 3. Hair Thickness
    if "thick" in sh:
        vec["SH_Thick"] = 1.0
    elif "med" in sh:
        vec["SH_Med"] = 1.0
    else:
        vec["SH_Thin"] = 1.0

    # 4. Hair Color
    if "black" in shc:
        vec["SHC_Black"] = 1.0
    else:
        vec["SHC_Brown"] = 1.0

    # 5. Mole on skin
    if mole in ["yes", "1", "true", "present"]:
        vec["Mole on skin"] = 1.0
    else:
        vec["Mole on skin"] = 0.0

    # 6. Vein Visibility
    if "high" in vv:
        vec["VV_High"] = 1.0
    elif "med" in vv:
        vec["VV_Med"] = 1.0
    else:
        vec["VV_Low"] = 1.0

    # 7. Palm Color
    if "pinkred" in pc or "red" in pc:
        vec["PC_PinkRed"] = 1.0
    elif "brown" in pc:
        vec["PC_Brown"] = 1.0
    else:
        vec["PC_PinkWhite"] = 1.0

    return vec


def apply_clinical_modulation(
    raw_probs: Dict[str, float],
    vein: str = "Auto",
    mole: str = "Auto",
    skin: str = "Auto",
    skin_color: str = "Auto",
    palm_color: str = "Auto",
    hair_thickness: str = "Auto",
    hair_color: str = "Auto"
) -> Dict[str, float]:
    """
    Applies classical Ayurvedic likelihood modulation to CV-extracted probabilities.
    When set to 'Auto' or 'Unknown', likelihood multiplier is 1.0 (pure CV).
    When clinical markers are specified (from practitioner, chat, or CV), modulates V/P/K and renormalizes.
    """
    multipliers = {"V": 1.0, "P": 1.0, "K": 1.0}
    vein_s = str(vein) if vein else ""
    mole_s = str(mole) if mole else ""
    skin_s = str(skin) if skin else ""
    sc_s = str(skin_color) if skin_color else ""
    pc_s = str(palm_color) if palm_color else ""
    ht_s = str(hair_thickness) if hair_thickness else ""

    # 1. Vein Visibility (Sira Darshana)
    if "High" in vein_s:       # Prominent / Superficial (Vata hallmark)
        multipliers["V"] *= 1.45
        multipliers["K"] *= 0.70
    elif "Low" in vein_s:      # Deep / Concealed (Kapha hallmark)
        multipliers["K"] *= 1.45
        multipliers["V"] *= 0.65
    elif "Medium" in vein_s or "Med" in vein_s:
        multipliers["P"] *= 1.15

    # 2. Mole on Hand / Wrist (Tilaka)
    if mole_s in ["Yes", "1"]:
        multipliers["P"] *= 1.30
        multipliers["V"] *= 1.15
        multipliers["K"] *= 0.85
    elif mole_s in ["No", "0"]:
        multipliers["K"] *= 1.10
        multipliers["P"] *= 0.95

    # 3. Skin Type / Texture (Sparsha)
    if "Dry" in skin_s:        # Ruksha (Vata hallmark)
        multipliers["V"] *= 1.60
        multipliers["K"] *= 0.60
        multipliers["P"] *= 0.90
    elif "Rough" in skin_s:    # Khara (Vata hallmark)
        multipliers["V"] *= 1.50
        multipliers["K"] *= 0.65
        multipliers["P"] *= 0.90
    elif "Smooth" in skin_s or "Normal" in skin_s:  # Slakshna / Snigdha (Kapha / Pitta)
        multipliers["K"] *= 1.25
        multipliers["P"] *= 1.10
        multipliers["V"] *= 0.80
    elif "Oily" in skin_s:     # Sneha (Kapha / Pitta)
        multipliers["K"] *= 1.35
        multipliers["P"] *= 1.30
        multipliers["V"] *= 0.60

    # 4. Skin Color (Varna)
    if "Fair" in sc_s and "Auto" not in sc_s:
        multipliers["P"] *= 1.15
    elif "Dark" in sc_s and "Auto" not in sc_s:
        multipliers["V"] *= 1.20

    # 5. Palm Color (Kara Tala Varna)
    if ("PinkRed" in pc_s or "Red" in pc_s) and "Auto" not in pc_s:
        multipliers["P"] *= 1.25
    elif "Brown" in pc_s and "Auto" not in pc_s:
        multipliers["V"] *= 1.20
    elif "PinkWhite" in pc_s and "Auto" not in pc_s:
        multipliers["K"] *= 1.15

    # 6. Hair Thickness (Kesha)
    if "Thick" in ht_s and "Auto" not in ht_s:
        multipliers["K"] *= 1.20
    elif "Thin" in ht_s and "Auto" not in ht_s:
        multipliers["V"] *= 1.20

    # Calculate modulated unnormalized values
    modulated = {cls: raw_probs.get(cls, 0.0) * multipliers.get(cls, 1.0) for cls in raw_probs}
    total = sum(modulated.values())
    if total > 0:
        return {cls: val / total for cls, val in modulated.items()}
    return raw_probs
