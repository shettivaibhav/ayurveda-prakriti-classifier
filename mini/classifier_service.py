"""Model and clinical-marker services used by the React API."""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import random

import joblib

from palm_color import extract_features


VIEW_ORDER = [
    "palm_open_front",
    "palm_open_back",
    "palm_closed_front",
    "palm_closed_back",
]

DOSHA_DISPLAY = {
    "K": ("Kapha", "🌊"),
    "P": ("Pitta", "🔥"),
    "V": ("Vata", "🌬️"),
    "K-P": ("Kapha-Pitta", "🌊🔥"),
    "K-V": ("Kapha-Vata", "🌊🌬️"),
    "P-K": ("Pitta-Kapha", "🔥🌊"),
    "P-V": ("Pitta-Vata", "🔥🌬️"),
    "V-K": ("Vata-Kapha", "🌬️🌊"),
    "V-P": ("Vata-Pitta", "🌬️🔥"),
}


def generate_user_id():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{timestamp}_{random.randint(1000, 9999)}"


@lru_cache(maxsize=1)
def load_model():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "unified_prakriti_pipeline.joblib"
    if not model_path.exists():
        model_path = base_dir / "model_color.joblib"
    return joblib.load(model_path)


def clear_model_cache():
    load_model.cache_clear()


def display_name_and_emoji(class_label):
    return DOSHA_DISPLAY.get(class_label, (class_label, "⚪"))


def predict_dosha(images_by_angle, age, sex_m, bundle=None):
    """Extract features from the four views and return model probabilities."""
    if bundle is None:
        bundle = load_model()

    if "pipeline" in bundle:
        from unified_cv import extract_participant_features

        features = extract_participant_features(images_by_angle, age, sex_m).reshape(1, -1)
        probabilities = bundle["pipeline"].predict_proba(features)[0]
    else:
        features = extract_features(
            images_by_angle[bundle.get("skin_view", "palm_open_back")],
            images_by_angle[bundle.get("palm_view", "palm_open_front")],
            age,
            sex_m,
        ).reshape(1, -1)
        probabilities = bundle["model"].predict_proba(features)[0]

    return {
        label: float(probability)
        for label, probability in zip(bundle["classes"], probabilities)
    }


def analyze_images(images_by_angle, age, gender, observations=None):
    """Run the same CV marker extraction and probability modulation as the app."""
    from unified_cv import apply_clinical_modulation, extract_cv_clinical_markers

    observations = observations or {}
    image_markers = extract_cv_clinical_markers(images_by_angle)
    raw_probabilities = predict_dosha(
        images_by_angle,
        int(age),
        1 if str(gender).lower() == "male" else 0,
    )

    auto = "Auto (from images)"
    vein = observations.get("vein_visibility", auto)
    mole = observations.get("mole", auto)
    skin = observations.get("skin_texture", auto)
    skin_color = observations.get("skin_color", auto)
    palm_color = observations.get("palm_color", auto)
    hair_thickness = observations.get("hair_thickness", auto)

    effective = {
        "Vein Visibility": (
            image_markers.get("vein_visibility", "Low") if vein == auto else vein,
            "Computer Vision" if vein == auto else "Practitioner",
        ),
        "Mole on Skin": (
            image_markers.get("mole", "No") if mole == auto else mole,
            "Computer Vision" if mole == auto else "Practitioner",
        ),
        "Skin Texture": (
            image_markers.get("skin_texture", "Smooth") if skin == auto else skin,
            "Computer Vision" if skin == auto else "Practitioner",
        ),
        "Skin Color": (
            image_markers.get("skin_color", "Medium") if skin_color == auto else skin_color,
            "Computer Vision" if skin_color == auto else "Practitioner",
        ),
        "Palm Color": (
            image_markers.get("palm_color", "PinkWhite") if palm_color == auto else palm_color,
            "Computer Vision" if palm_color == auto else "Practitioner",
        ),
        "Hair Thickness": (
            image_markers.get("hair_thickness", "Thin") if hair_thickness == auto else hair_thickness,
            "Computer Vision" if hair_thickness == auto else "Practitioner",
        ),
        "Hair Color": (image_markers.get("hair_color", "Brown"), "Computer Vision"),
    }

    is_practitioner_mode = any(value != auto for value in observations.values())
    final_probabilities = apply_clinical_modulation(
        raw_probabilities,
        vein=effective["Vein Visibility"][0],
        mole=effective["Mole on Skin"][0],
        skin=effective["Skin Texture"][0],
        skin_color=effective["Skin Color"][0],
        palm_color=effective["Palm Color"][0],
        hair_thickness=effective["Hair Thickness"][0],
        hair_color=effective["Hair Color"][0],
    )

    ranked = sorted(final_probabilities.items(), key=lambda item: item[1], reverse=True)
    top_class, confidence = ranked[0]
    name, emoji = display_name_and_emoji(top_class)

    return {
        "probabilities": final_probabilities,
        "raw_probabilities": raw_probabilities,
        "dominant_class": top_class,
        "dominant_name": name,
        "dominant_emoji": emoji,
        "confidence": confidence,
        "secondary_class": ranked[1][0] if len(ranked) > 1 else None,
        "secondary_confidence": ranked[1][1] if len(ranked) > 1 else None,
        "effective_markers": effective,
        "image_markers": image_markers,
        "practitioner_mode": is_practitioner_mode,
        "model_name": load_model().get("model_name", "Unified ML"),
    }
