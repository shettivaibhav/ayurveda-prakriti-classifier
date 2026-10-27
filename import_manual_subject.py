import os
import sys
import argparse
from pathlib import Path

# Add project root to python path to resolve modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "mini"))

from whatsapp_pipeline.exporter import append_to_source_dataset

def main():
    parser = argparse.ArgumentParser(description="Ayurvedic Prakriti Classifier - Manual Subject Importer")
    parser.add_argument("--name", help="Subject Name")
    parser.add_argument("--age", type=int, help="Subject Age")
    parser.add_argument("--gender", choices=["male", "female"], help="Subject Gender")
    parser.add_argument("--open-front", help="Path to Palm Open – Front View (Palm)")
    parser.add_argument("--open-back", help="Path to Palm Open – Back View (Skin/Veins)")
    parser.add_argument("--closed-front", help="Path to Palm Closed – Front View (Fist)")
    parser.add_argument("--closed-back", help="Path to Palm Closed – Back View (Outer fist)")
    
    args = parser.parse_args()

    # Check if any arguments are passed; if not, fall back to interactive mode
    if any(val is not None for val in (args.name, args.age, args.gender, args.open_front, args.open_back, args.closed_front, args.closed_back)):
        # Validate CLI arguments
        if not args.name or not args.age or not args.gender or not args.open_front or not args.open_back or not args.closed_front or not args.closed_back:
            print("[Error] If using command-line arguments, you must provide all of: --name, --age, --gender, --open-front, --open-back, --closed-front, --closed-back")
            return 1
        name = args.name.strip()
        age = args.age
        gender = args.gender.strip().lower()
        image_views = {
            "palm_open_front": str(Path(args.open_front).resolve()),
            "palm_open_back": str(Path(args.open_back).resolve()),
            "palm_closed_front": str(Path(args.closed_front).resolve()),
            "palm_closed_back": str(Path(args.closed_back).resolve())
        }
        for view_key, path_str in image_views.items():
            if not Path(path_str).exists():
                print(f"[Error] File does not exist at '{path_str}'.")
                return 1
    else:
        print("=" * 70)
        print("           Ayurvedic Prakriti Classifier - Manual Subject Importer")
        print("=" * 70)
        print("Use this script to manually import a flagged subject by specifying their details")
        print("and the correct image paths for each view. This will bypass classification errors.")
        print("-" * 70)

        # Gather subject details
        name = input("Subject Name [e.g. Sakshi k]: ").strip()
        if not name:
            print("Error: Name is required.")
            return 1
            
        age_str = input("Age [e.g. 22]: ").strip()
        try:
            age = int(age_str)
        except ValueError:
            print("Error: Age must be an integer.")
            return 1
            
        gender = input("Gender (male/female) [e.g. female]: ").strip().lower()
        if gender not in ("male", "female"):
            print("Error: Gender must be 'male' or 'female'.")
            return 1

        print("\nPlease enter the paths to the 4 hand images. You can drag and drop the files")
        print("into the terminal or type their absolute/relative paths:")
        
        views = {
            "palm_open_front": "Palm Open – Front View (Palm)",
            "palm_open_back": "Palm Open – Back View (Skin/Veins)",
            "palm_closed_front": "Palm Closed – Front View (Fist)",
            "palm_closed_back": "Palm Closed – Back View (Outer fist)"
        }
        
        image_views = {}
        for view_key, view_desc in views.items():
            while True:
                path_str = input(f"  {view_desc}: ").strip().strip('"').strip("'")
                if not path_str:
                    print("    Error: Path cannot be empty.")
                    continue
                path = Path(path_str)
                if not path.exists():
                    print(f"    Error: File does not exist at '{path_str}'. Please check the path and try again.")
                    continue
                image_views[view_key] = str(path.resolve())
                break

    # Construct the subject result structure
    subject = {
        "sender": "manual_import",
        "name": name,
        "age": age,
        "gender": gender,
        "skin_type": "Unknown",
        "skin_color": "Medium",
        "skin_texture": "Smooth",
        "vein_visibility": "Low",
        "palm_color": "PinkWhite",
        "mole": "No",
        "hair_thickness": "None",
        "hair_color": "None",
        "image_views": image_views,
        "needs_review": False,  # Bypassing the review flag
        "review_reasons": []
    }

    # 2. Append to dataset
    xlsx_path = Path("dataset/palm_dataset_26 June 26.xlsx")
    wrist_dataset_dir = Path("dataset/Wrist_Dataset")

    try:
        appended_ids = append_to_source_dataset(
            [subject],
            xlsx_path=xlsx_path,
            wrist_dataset_dir=wrist_dataset_dir
        )
        if appended_ids:
            print("\n" + "=" * 70)
            print("                       IMPORT SUCCESSFUL")
            print("=" * 70)
            print(f"Subject '{name}' has been successfully imported as ID '{appended_ids[0]}'.")
            print("The images were copied, metadata.json was created, and the Excel sheet has been updated.")
            print("=" * 70)
        else:
            print("\n[Error] The subject could not be appended. Please verify that the Excel sheet is closed.")
    except Exception as e:
        print(f"\n[FATAL ERROR] Import failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    sys.exit(main())
