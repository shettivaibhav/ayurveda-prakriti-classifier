"""Builds the canonical subject-level dataset for model_v2.

Joins the practitioner-labeled xlsx (7-class dual-dosha Prakriti + tabular
features) to the 4 wrist-angle image paths in Wrist_Dataset/, keyed by id.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = PROJECT_ROOT / "palm_dataset_26 June 26.xlsx"
WRIST_DATASET_DIR = PROJECT_ROOT / "Wrist_Dataset"

VIEW_ORDER = ["palm_open_front", "palm_open_back", "palm_closed_front", "palm_closed_back"]

# Filenames are inconsistent across subjects: 70 use "palm_closed_*",
# 10 use "palm_close_*" (missing the 'd'). Try both.
VIEW_FILENAME_CANDIDATES = {
    "palm_open_front": ["palm_open_front.jpg"],
    "palm_open_back": ["palm_open_back.jpg"],
    "palm_closed_front": ["palm_closed_front.jpg", "palm_close_front.jpg"],
    "palm_closed_back": ["palm_closed_back.jpg", "palm_close_back.jpg"],
}

TABULAR_FEATURE_COLS = [
    "SC_Fair", "SC_Medium", "SC_Dark",
    "ST_Rough", "ST_Smooth", "ST_Dry", "ST_Oily",
    "SH_Thick", "SH_Thin", "SH_Med",
    "SHC_Brown", "SHC_Black",
    "Mole on skin",
    "VV_Low", "VV_Med", "VV_High",
    "PC_PinkRed", "PC_PinkWhite", "PC_Brown",
]

LABEL_COL = "Prakriti"


def load_labels() -> pd.DataFrame:
    """Loads the canonical label sheet. Ignores 'Sheet1' (stale draft)."""
    df = pd.read_excel(XLSX_PATH, sheet_name="Sheet")
    df = df.drop(columns=["open_front", "open_back", "closed_front", "closed_back"])
    return df


def _resolve_subject_dir(sid: str) -> Path:
    """Some xlsx ids use '_Male'/'_Female' while on-disk folders are lowercase; try both."""
    exact = WRIST_DATASET_DIR / sid
    if exact.is_dir():
        return exact
    num, _, gender = sid.partition("_")
    normalized = WRIST_DATASET_DIR / f"{num}_{gender.lower()}"
    return normalized if normalized.is_dir() else exact


def _resolve_view_path(subject_dir: Path, view: str) -> Path:
    """Returns the first existing candidate filename for this view, or the primary candidate if none exist."""
    candidates = VIEW_FILENAME_CANDIDATES[view]
    for name in candidates:
        p = subject_dir / name
        if p.is_file():
            return p
    return subject_dir / candidates[0]


def attach_image_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Adds one column per view with the resolved image path; drops subjects missing any view."""
    df = df.copy()
    subject_dirs = df["id"].apply(_resolve_subject_dir)
    for view in VIEW_ORDER:
        df[view] = subject_dirs.apply(lambda d, v=view: _resolve_view_path(d, v))

    exists_mask = df[VIEW_ORDER].map(lambda p: p.is_file()).all(axis=1)
    missing = df.loc[~exists_mask, "id"].tolist()
    if missing:
        print(f"Dropping {len(missing)} subjects with missing image files: {missing}")
    return df.loc[exists_mask].reset_index(drop=True)


def build_dataset() -> pd.DataFrame:
    df = load_labels()
    df = attach_image_paths(df)
    df[VIEW_ORDER] = df[VIEW_ORDER].map(str)
    return df


def main():
    df = build_dataset()
    print(f"Subjects: {len(df)}")
    print("\nPrakriti class distribution:")
    print(df[LABEL_COL].value_counts())
    print(f"\nMissing values:\n{df.isna().sum()[df.isna().sum() > 0]}")
    out_path = Path(__file__).resolve().parent / "results" / "dataset_manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote manifest to {out_path}")


if __name__ == "__main__":
    main()
