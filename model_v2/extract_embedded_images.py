"""One-time recovery script: extracts wrist photos that were pasted directly
into cells of palm_dataset_26 June 26.xlsx (sheet 'Sheet') rather than saved
as files. pandas/read_excel only reads cell values, so these embedded images
are invisible to data.py's normal join and those subjects were being dropped
as "missing photos" even though the photos exist inside the spreadsheet.

Only writes folders/files that don't already exist in Wrist_Dataset/ - never
overwrites the 80 subjects that already have full-resolution photos on disk.
"""

import io
from pathlib import Path

import openpyxl
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = PROJECT_ROOT / "palm_dataset_26 June 26.xlsx"
WRIST_DATASET_DIR = PROJECT_ROOT / "Wrist_Dataset"

COL_TO_VIEW = {1: "open_front", 2: "open_back", 3: "closed_front", 4: "closed_back"}
VIEW_TO_FILENAME = {
    "open_front": "palm_open_front.jpg",
    "open_back": "palm_open_back.jpg",
    "closed_front": "palm_closed_front.jpg",
    "closed_back": "palm_closed_back.jpg",
}


def normalize_subject_id(sid: str) -> str:
    """'1083_Female' -> '1083_female', matching the existing folder convention."""
    num, _, gender = sid.partition("_")
    return f"{num}_{gender.lower()}"


def main():
    wb = openpyxl.load_workbook(XLSX_PATH)
    ws = wb["Sheet"]

    images_by_subject = {}
    for img in ws._images:
        anchor = img.anchor
        excel_row = anchor._from.row + 1
        col = anchor._from.col
        sid = ws.cell(row=excel_row, column=1).value
        view = COL_TO_VIEW.get(col)
        if sid is None or view is None:
            continue
        # keep first image if a cell somehow has duplicates pasted in
        images_by_subject.setdefault(sid, {}).setdefault(view, img)

    print(f"Found embedded images for {len(images_by_subject)} subjects total.")

    recovered, skipped_existing, partial = [], [], []
    for sid, views in images_by_subject.items():
        norm_id = normalize_subject_id(sid)
        subject_dir = WRIST_DATASET_DIR / norm_id

        if subject_dir.exists():
            skipped_existing.append(norm_id)
            continue

        subject_dir.mkdir(parents=True, exist_ok=False)
        for view, img in views.items():
            data = img._data()
            pil = Image.open(io.BytesIO(data)).convert("RGB")
            out_path = subject_dir / VIEW_TO_FILENAME[view]
            pil.save(out_path, format="JPEG", quality=95)

        if len(views) == 4:
            recovered.append(norm_id)
        else:
            partial.append((norm_id, sorted(views.keys())))

    print(f"\nRecovered {len(recovered)} subjects with all 4 views: {recovered}")
    print(f"Recovered {len(partial)} subjects with partial views (excluded from multi-view training): {partial}")
    print(f"Skipped {len(skipped_existing)} subjects that already had photos on disk.")


if __name__ == "__main__":
    main()
