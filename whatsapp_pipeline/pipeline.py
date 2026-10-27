import os
import time
from pathlib import Path
from .parser import parse_chat_log, group_by_subject
from .extractor import extract_details
from .view_classifier import classify_hand_views
from .predictor import PrakritiPredictor
from .exporter import generate_excel_report

class WhatsAppAIClassifierPipeline:
    def __init__(self, confidence_threshold=0.0):
        self.predictor = PrakritiPredictor(confidence_threshold=confidence_threshold)
        
    def process_chat_export(self, chat_txt_path, media_dir_path, output_excel_path, progress_callback=None):
        """
        Runs the end-to-end agentic pipeline to parse the chat, extract details,
        classify images, predict Prakriti, and export the results to Excel.
        """
        chat_txt_path = Path(chat_txt_path)
        media_dir_path = Path(media_dir_path)
        
        if not chat_txt_path.exists():
            raise FileNotFoundError(f"WhatsApp chat log file not found at: {chat_txt_path}")
        if not media_dir_path.exists():
            raise FileNotFoundError(f"WhatsApp media directory not found at: {media_dir_path}")
            
        if progress_callback:
            progress_callback(2, "Parsing WhatsApp chat messages...")

        print(f"Starting WhatsApp Chat Parsing: {chat_txt_path.name}")
        messages = parse_chat_log(chat_txt_path)
        print(f"  Parsed {len(messages)} messages chronologically.")
        
        subjects = group_by_subject(messages)
        print(f"  Identified {len(subjects)} image-grouped subjects from chronological batches.")
        
        if progress_callback:
            progress_callback(5, f"Identified {len(subjects)} subjects. Beginning analysis...")
        
        results = []
        total_subjects = max(1, len(subjects))
        for idx, subj in enumerate(subjects):
            if idx > 0:
                time.sleep(0.5)
            sender = subj["sender"]
            images = subj["images"]
            text = subj["associated_text"]
            
            pct = int(5 + (idx / total_subjects) * 88)
            if progress_callback:
                progress_callback(pct, f"[{idx+1}/{total_subjects}] Processing Subject {idx+1}...")

            # 1. Extract Details from unstructured text
            details = extract_details(text)
            subj_label = details.get("name") if details.get("name") and details["name"] != "Unknown" else f"Subject {idx+1}"
            if progress_callback:
                progress_callback(pct, f"[{idx+1}/{total_subjects}] Analyzing {subj_label}...")

            print(f"  Extracted Details -> Name: {details['name']}, Age: {details['age']}, Gender: {details['gender']}, Skin: {details['skin_type']}")
            
            # 2. Resolve image paths inside the media directory
            full_image_paths = []
            for img in images:
                # Try exact filename match first
                img_path = media_dir_path / img
                if img_path.exists():
                    full_image_paths.append(str(img_path))
                else:
                    # Fallback to case-insensitive match
                    found = False
                    for child in media_dir_path.iterdir():
                        if child.name.lower() == img.lower():
                            full_image_paths.append(str(child))
                            found = True
                            break
                    if not found:
                        print(f"  [Warning] Attachment file '{img}' not found in media directory.")
                        
            # 3. Classify Image Views
            view_mapping = {}
            if len(full_image_paths) > 0:
                view_mapping = classify_hand_views(full_image_paths)
                print(f"  Classified Views -> { {Path(k).name: v for k, v in view_mapping.items()} }")
            else:
                print("  [Warning] No matching image files found for this subject.")
                
            # Restructure image views for predictor mapping
            mapped_views = {v: k for k, v in view_mapping.items() if v is not None}
            
            # 4. Extract CV clinical markers from available wrist images
            loaded_images = {}
            for vk, img_p in mapped_views.items():
                if img_p and Path(img_p).exists():
                    try:
                        from PIL import Image as PILImage
                        loaded_images[vk] = PILImage.open(img_p).convert("RGB")
                    except Exception:
                        loaded_images[vk] = None
            
            from unified_cv import extract_cv_clinical_markers
            cv_markers = extract_cv_clinical_markers(loaded_images)

            # Rule: If user has given info in chat text, use it; otherwise fill using CV
            combined_markers = dict(details)
            for k in ["vein_visibility", "mole", "skin_texture", "skin_type", "skin_color", "palm_color", "hair_thickness", "hair_color"]:
                val = details.get(k)
                if val and str(val).strip() not in ("Unknown", "None", ""):
                    combined_markers[k] = val
                else:
                    combined_markers[k] = cv_markers.get(k, "Unknown")

            # 5. Predict Prakriti with unified CV + ML
            open_front_path = mapped_views.get("palm_open_front")
            open_back_path = mapped_views.get("palm_open_back")
            closed_front_path = mapped_views.get("palm_closed_front")
            closed_back_path = mapped_views.get("palm_closed_back")
            
            prediction = self.predictor.predict_subject(
                age=details["age"],
                gender=details["gender"],
                palm_open_front_path=open_front_path,
                palm_open_back_path=open_back_path,
                palm_closed_front_path=closed_front_path,
                palm_closed_back_path=closed_back_path,
                details=combined_markers
            )
            
            if prediction["needs_review"]:
                print(f"  [REVIEW] Flagged for review: {', '.join(prediction['review_reasons'])}")
            else:
                print(f"  [RESULT] Extracted Prakriti: {prediction['predicted_dosha']}")
                
            # Compile final subject summary
            subject_result = {
                "sender": sender,
                "name": details["name"],
                "age": details["age"],
                "gender": details["gender"],
                "skin_type": combined_markers.get("skin_type", "Unknown"),
                "skin_color": combined_markers.get("skin_color", "Unknown"),
                "skin_texture": combined_markers.get("skin_texture", "Unknown"),
                "vein_visibility": combined_markers.get("vein_visibility", "Unknown"),
                "palm_color": combined_markers.get("palm_color", "Unknown"),
                "mole": combined_markers.get("mole", "Unknown"),
                "hair_thickness": combined_markers.get("hair_thickness", "None"),
                "hair_color": combined_markers.get("hair_color", "None"),
                "image_views": mapped_views,
                **prediction
            }
            results.append(subject_result)
            
        # 5. Export results to formatted Excel
        if progress_callback:
            progress_callback(96, "Generating formatted Excel batch report...")

        print(f"\nGenerating Excel report: {output_excel_path}")
        report_path = generate_excel_report(results, output_excel_path)
        print(f"Excel report created successfully at: {report_path.resolve()}")
        
        if progress_callback:
            progress_callback(100, f"Completed! Successfully analyzed all {len(results)} subjects.")

        return results
