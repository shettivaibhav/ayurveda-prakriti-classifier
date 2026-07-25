"""
Supabase helper functions for uploading wrist images and data
"""
import io
from PIL import Image
from supabase import create_client
from datetime import datetime
import hashlib
import streamlit as st


# Supabase configuration
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]

# Initialize Supabase client
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)


def compress_image(image, max_size_kb=200, quality=85):
    """
    Compress PIL Image to target size
    
    Args:
        image: PIL Image object
        max_size_kb: Maximum size in KB (default 200KB)
        quality: Initial JPEG quality (default 85)
    
    Returns:
        bytes: Compressed image bytes
    """
    # Convert to RGB if necessary
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    
    # Start with initial quality
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=quality, optimize=True)
    
    # Reduce quality if still too large
    while output.tell() > max_size_kb * 1024 and quality > 20:
        output = io.BytesIO()
        quality -= 5
        image.save(output, format='JPEG', quality=quality, optimize=True)
    
    return output.getvalue()


def upload_image_to_storage(image, user_id, gender, angle_key):
    """
    Upload a single image to Supabase storage
    
    Args:
        image: PIL Image object
        user_id: User ID (e.g., "1001")
        gender: Gender ("male" or "female")
        angle_key: Angle key (e.g., "palm_open_front")
    
    Returns:
        dict: {success: bool, url: str or None, error: str or None}
    """
    try:
        # Compress image
        img_bytes = compress_image(image, max_size_kb=200, quality=85)
        
        # Create storage path: {id}_{gender}/{angle}.jpg
        folder_name = f"{user_id}_{gender}"
        filename = f"{angle_key}.jpg"
        storage_path = f"{folder_name}/{filename}"
        
        # Upload to storage bucket
        response = supabase_client.storage.from_("wrist_images").upload(
            storage_path,
            img_bytes,
            {"content-type": "image/jpeg", "upsert": "true"}
        )
        
        # Check for errors
        if hasattr(response, 'error') and response.error:
            return {
                "success": False,
                "url": None,
                "error": f"Storage error: {response.error.message}"
            }
        
        # Get public URL
        public_url = supabase_client.storage.from_("wrist_images").get_public_url(storage_path)
        
        return {
            "success": True,
            "url": public_url,
            "error": None,
            "size_kb": len(img_bytes) / 1024
        }
        
    except Exception as e:
        return {
            "success": False,
            "url": None,
            "error": str(e)
        }


def upload_all_images(images_dict, user_id, gender):
    """
    Upload all 4 wrist images to Supabase storage
    
    Args:
        images_dict: Dict with keys like "palm_open_front" and PIL Image values
        user_id: User ID
        gender: Gender
    
    Returns:
        dict: {success: bool, urls: dict, errors: list}
    """
    results = {
        "success": True,
        "urls": {},
        "errors": []
    }
    
    for angle_key, image in images_dict.items():
        result = upload_image_to_storage(image, user_id, gender, angle_key)
        
        if result["success"]:
            results["urls"][angle_key] = result["url"]
        else:
            results["success"] = False
            results["errors"].append(f"{angle_key}: {result['error']}")
    
    return results


def insert_wrist_data(user_id, gender, probabilities, final_dosha, confidence):
    """
    Insert wrist analysis data into Supabase table
    
    Args:
        user_id: User ID
        gender: Gender
        probabilities: Dict with "Vata", "Pitta", "Kapha" probabilities
        final_dosha: Final predicted dosha string (e.g., "Vata", "Vata-Pitta")
        confidence: Confidence score (0-1)
    
    Returns:
        dict: {success: bool, record_id: int or None, error: str or None}
    """
    try:
        # Prepare data for insertion
        data = {
            "user_id": str(user_id),
            "gender": gender,
            "angle": "combined_4_views",  # Indicates this is aggregated result
            "vata_prob": round(probabilities.get("Vata", 0.0), 4),
            "pitta_prob": round(probabilities.get("Pitta", 0.0), 4),
            "kapha_prob": round(probabilities.get("Kapha", 0.0), 4),
            "predicted_dosha": final_dosha,
            "confidence": round(confidence, 4),
            "image_data": None,  # Not storing base64, using storage bucket instead
            "image_size": None   # Could calculate total size if needed
        }
        
        # Insert into table
        response = supabase_client.table("wrist_uploads").insert(data).execute()
        
        # Check for successful insert
        if response.data and len(response.data) > 0:
            record_id = response.data[0].get("id")
            return {
                "success": True,
                "record_id": record_id,
                "error": None
            }
        else:
            return {
                "success": False,
                "record_id": None,
                "error": "No data returned from insert"
            }
            
    except Exception as e:
        return {
            "success": False,
            "record_id": None,
            "error": str(e)
        }


def upload_complete_analysis(user_id, gender, images_dict, probabilities, final_dosha, confidence):
    """
    Complete upload workflow: upload images and insert data
    
    Args:
        user_id: User ID (e.g., "1001")
        gender: Gender ("male" or "female")
        images_dict: Dict with angle keys and PIL Image values
        probabilities: Dict with "Vata", "Pitta", "Kapha" probabilities
        final_dosha: Final predicted dosha
        confidence: Confidence score
    
    Returns:
        dict: {success: bool, record_id: int, image_urls: dict, errors: list}
    """
    results = {
        "success": False,
        "record_id": None,
        "image_urls": {},
        "errors": []
    }
    
    # Step 1: Upload all images
    upload_result = upload_all_images(images_dict, user_id, gender)
    
    if not upload_result["success"]:
        results["errors"].extend(upload_result["errors"])
        results["errors"].append("Image upload failed")
        return results
    
    results["image_urls"] = upload_result["urls"]
    
    # Step 2: Insert data into table
    insert_result = insert_wrist_data(user_id, gender, probabilities, final_dosha, confidence)
    
    if not insert_result["success"]:
        results["errors"].append(f"Database insert failed: {insert_result['error']}")
        return results
    
    results["record_id"] = insert_result["record_id"]
    results["success"] = True
    
    return results
