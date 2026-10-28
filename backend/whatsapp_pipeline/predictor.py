import joblib
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MINI_PATH = PROJECT_ROOT / "mini"

import sys
if str(MINI_PATH) not in sys.path:
    sys.path.insert(0, str(MINI_PATH))

from unified_cv import (
    extract_participant_features, validate_image_quality, apply_clinical_modulation
)


class PrakritiPredictor:
    """
    Unified Computer Vision + Classical ML Predictor for Ayurvedic Prakriti classification.
    Uses 4-view anatomical ROI extraction and the winning trained ML pipeline.
    """
    def __init__(self, model_path=None, confidence_threshold=0.0):
        self.confidence_threshold = confidence_threshold
        if model_path is None:
            model_path = MINI_PATH / "unified_prakriti_pipeline.joblib"
            if not model_path.exists():
                model_path = MINI_PATH / "model_color.joblib"
        
        self.model_path = Path(model_path)
        if self.model_path.exists():
            self.bundle = joblib.load(self.model_path)
        else:
            self.bundle = None

    def predict_subject(
        self,
        age,
        gender,
        palm_open_front_path=None,
        palm_open_back_path=None,
        palm_closed_front_path=None,
        palm_closed_back_path=None,
        details=None
    ):
        """
        Runs image quality validation, extracts 4-view CV features, and executes
        the unified ML pipeline.
        
        Returns:
            dict: {
                'vata_prob': float,
                'pitta_prob': float,
                'kapha_prob': float,
                'predicted_dosha': str,
                'confidence': float,
                'cv_features': dict,
                'needs_review': bool,
                'review_reasons': [str]
            }
        """
        result = {
            "vata_prob": 0.0,
            "pitta_prob": 0.0,
            "kapha_prob": 0.0,
            "predicted_dosha": "Unknown",
            "confidence": 0.0,
            "cv_features": {},
            "needs_review": False,
            "review_reasons": []
        }

        # 1. Check for missing demographic details
        if age in (None, "Unknown", "") or gender in (None, "Unknown", ""):
            result["needs_review"] = True
            result["review_reasons"].append("Missing Age or Gender details")

        # Load available images and validate quality
        paths = {
            "palm_open_front": palm_open_front_path,
            "palm_open_back": palm_open_back_path,
            "palm_closed_front": palm_closed_front_path,
            "palm_closed_back": palm_closed_back_path
        }
        
        images = {}
        for vk, p in paths.items():
            if p and Path(p).exists():
                try:
                    img = Image.open(p).convert("RGB")
                    images[vk] = img
                    # Quality check
                    q_res = validate_image_quality(img)
                    if not q_res["valid"]:
                        result["review_reasons"].extend([f"{vk}: {iss}" for iss in q_res["issues"]])
                except Exception as e:
                    images[vk] = None
                    result["review_reasons"].append(f"Corrupt image {vk}: {e}")
            else:
                images[vk] = None

        # Check critical views
        if images.get("palm_open_front") is None:
            result["needs_review"] = True
            result["review_reasons"].append("Missing required palm_open_front image")
            
        if images.get("palm_open_back") is None:
            result["needs_review"] = True
            result["review_reasons"].append("Missing required palm_open_back image")

        # 2. Run prediction if model is loaded
        if self.bundle is not None:
            try:
                is_male = 1.0 if str(gender).strip().upper().startswith("M") else 0.0
                age_val = float(age) if age not in (None, "Unknown", "") else 35.0

                if "pipeline" in self.bundle:
                    # 70-D unified pipeline
                    feats = extract_participant_features(images, age=age_val, sex_m=is_male).reshape(1, -1)
                    probs_arr = self.bundle["pipeline"].predict_proba(feats)[0]
                    classes = self.bundle["classes"]
                    prob_dict = {cls: float(p) for cls, p in zip(classes, probs_arr)}
                else:
                    # Legacy fallback
                    from palm_color import extract_features
                    sc_img = images.get("palm_open_back")
                    pc_img = images.get("palm_open_front")
                    if sc_img and pc_img:
                        feats = extract_features(sc_img, pc_img, age_val, is_male).reshape(1, -1)
                        probs_arr = self.bundle["model"].predict_proba(feats)[0]
                        classes = self.bundle["classes"]
                        prob_dict = {cls: float(p) for cls, p in zip(classes, probs_arr)}
                    else:
                        prob_dict = {"K": 0.33, "P": 0.33, "V": 0.33}

                # Apply clinical modulation using combined clinical markers (from text or CV)
                if details:
                    prob_dict = apply_clinical_modulation(
                        prob_dict,
                        vein=details.get("vein_visibility", "Auto"),
                        mole=details.get("mole", "Auto"),
                        skin=details.get("skin_texture") or details.get("skin_type", "Auto"),
                        skin_color=details.get("skin_color", "Auto"),
                        palm_color=details.get("palm_color", "Auto"),
                        hair_thickness=details.get("hair_thickness", "Auto"),
                        hair_color=details.get("hair_color", "Auto")
                    )

                result["kapha_prob"] = prob_dict.get("K", 0.0)
                result["pitta_prob"] = prob_dict.get("P", 0.0)
                result["vata_prob"] = prob_dict.get("V", 0.0)

                ranked = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                top_class = ranked[0][0]
                sec_class = ranked[1][0] if len(ranked) > 1 else None
                confidence = prob_dict[top_class]

                # Shared / Dual Prakriti (e.g. K-P, K-V, P-V) if secondary dosha is significant (>= 25%)
                if sec_class and ranked[1][1] >= 0.25:
                    result["predicted_dosha"] = f"{top_class}-{sec_class}"
                else:
                    result["predicted_dosha"] = top_class

                result["confidence"] = confidence
                result["dominant_dosha"] = top_class
                result["secondary_dosha"] = sec_class if (sec_class and ranked[1][1] >= 0.25) else None

                if confidence < self.confidence_threshold:
                    result["needs_review"] = True
                    result["review_reasons"].append(f"Low confidence ({confidence:.1%} < {self.confidence_threshold:.0%})")

            except Exception as e:
                result["needs_review"] = True
                result["review_reasons"].append(f"Prediction error: {e}")

        # Override if ground truth is already provided in text
        if details and details.get("prakriti") and details.get("prakriti") != "Unknown":
            result["predicted_dosha"] = details["prakriti"]
            result["confidence"] = 1.0

        return result
