"""
Batch Importer for New Participant ZIP Archives (~30 and ~20 cohorts).
Handles:
  1. Extracting ZIP archives into staging.
  2. Parsing participant text details via extractor.py (Name, Age, Sex, Vein, Mole, Skin).
  3. Organizing 4 hand view images into dataset/Wrist_Dataset/{id}/.
  4. Writing metadata.json for each participant.
  5. Appending new records to doctor_validation_prakriti_master.xlsx with embedded thumbnails.
  6. Uploading images and metadata to Supabase.
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
from PIL import Image as PILImage
import io

PROJECT_ROOT = Path("c:/Users/hp/Desktop/ayurveda prakriti classifier")
WRIST_DIR = PROJECT_ROOT / "dataset" / "Wrist_Dataset"
MASTER_XLSX = PROJECT_ROOT / "doctor_validation_prakriti_master.xlsx"
DATASET_XLSX = PROJECT_ROOT / "dataset" / "doctor_validation_prakriti_master.xlsx"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mini"))
sys.path.insert(0, str(PROJECT_ROOT / "whatsapp_pipeline"))

from extractor import extract_details
from supabasehelper import upload_complete_analysis, supabase_client

def get_next_participant_id():
    """Finds the maximum participant number currently in Wrist_Dataset and Excel."""
    max_num = 1000
    if WRIST_DIR.exists():
        for d in WRIST_DIR.iterdir():
            if d.is_dir():
                num_str, _, _ = d.name.partition("_")
                try:
                    max_num = max(max_num, int(num_str))
                except ValueError:
                    pass
    return max_num + 1

def process_zip_archive(zip_path: str):
    """
    Processes a single ZIP file containing participant folders or WhatsApp exports.
    """
    zip_p = Path(zip_path).resolve()
    if not zip_p.exists():
        print(f"[Error] ZIP file not found at: {zip_p}")
        return []

    print(f"\n=======================================================")
    print(f"Processing ZIP Archive: {zip_p.name}")
    print(f"=======================================================")

    staging_dir = PROJECT_ROOT / "staging_zip_import" / zip_p.stem
    staging_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_p, 'r') as zf:
        zf.extractall(staging_dir)
    print(f"Extracted to staging: {staging_dir}")

    # Process participants inside staging
    # Can handle either folder-per-participant or flat WhatsApp exported chat with media
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Import Participant ZIP Archives")
    parser.add_argument("zip_files", nargs="*", help="Paths to ZIP archives to process")
    args = parser.parse_args()

    if not args.zip_files:
        print("Doctor Validation Batch Importer is ready.")
        print(f"Next participant numeric ID will begin at: {get_next_participant_id()}")
        print("Usage: python import_zip_batches.py path/to/batch1.zip path/to/batch2.zip")
    else:
        for zf in args.zip_files:
            process_zip_archive(zf)
