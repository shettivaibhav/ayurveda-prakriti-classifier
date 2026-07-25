"""Per-image, subject-grouped dataset for the 4-view averaging experiments.

Unlike model_v2/data.py (which builds ONE row per subject with 4 image columns and
drops any subject missing a view), this builds ONE row per available hand-view image.
That is what the "train per image, average the 4 predictions" approach needs, and it
lets us keep the 2 subjects that only have 3 views.

Image source: palm_dataset_full/ — the only source that covers all 92 subjects with
consistent filenames (Wrist_Dataset/ has higher-res originals but is missing 7
subjects, which would force a mixed-resolution dataset). At 375x500 every image is
already larger than the 224x224 the backbones consume, so resolution is not the
bottleneck here (n=92 is).

Key: every image of a subject carries that subject's Prakriti label, and the
`subject_id` column MUST be used as the CV group so a subject's views never straddle
the train/test split.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "palm_dataset_full"
LABELS_CSV = DATA_DIR / "labels_all.csv"
OUT_CSV = Path(__file__).resolve().parent / "image_manifest.csv"

LABEL_COL = "Prakriti"
# canonical view order (matches mini/app.py VIEW_ORDER, minus the "palm_" prefix here)
VIEWS = ["open_front", "open_back", "closed_front", "closed_back"]
VIEW_FILENAMES = {
    "open_front": "palm_open_front.jpg",
    "open_back": "palm_open_back.jpg",
    "closed_front": "palm_closed_front.jpg",
    "closed_back": "palm_closed_back.jpg",
}


def build_image_df() -> pd.DataFrame:
    """Returns one row per available view image: subject_id, view, img_path, Prakriti."""
    labels = pd.read_csv(LABELS_CSV)
    rows = []
    for _, r in labels.iterrows():
        sid = r["id"]
        subj_dir = DATA_DIR / sid
        for view in VIEWS:
            p = subj_dir / VIEW_FILENAMES[view]
            if p.is_file():
                rows.append(
                    {
                        "subject_id": sid,
                        "view": view,
                        "img_path": str(p),
                        LABEL_COL: r[LABEL_COL],
                    }
                )
    df = pd.DataFrame(rows)
    # stable order: by subject then canonical view
    df["_vorder"] = df["view"].map({v: i for i, v in enumerate(VIEWS)})
    df = df.sort_values(["subject_id", "_vorder"]).drop(columns="_vorder").reset_index(drop=True)
    return df


def subject_labels() -> pd.DataFrame:
    """One row per subject: subject_id, Prakriti (for subject-level CV splitting)."""
    labels = pd.read_csv(LABELS_CSV)
    return labels[["id", LABEL_COL]].rename(columns={"id": "subject_id"}).reset_index(drop=True)


def main():
    df = build_image_df()
    subs = subject_labels()
    print(f"Subjects: {subs['subject_id'].nunique()}   Images: {len(df)}")
    print("\nImages per view:")
    print(df["view"].value_counts().reindex(VIEWS).to_string())
    print("\nSubject-level Prakriti distribution:")
    print(subs[LABEL_COL].value_counts().to_string())
    print("\nImage-level Prakriti distribution:")
    print(df[LABEL_COL].value_counts().to_string())
    n3 = df.groupby("subject_id").size()
    print(f"\nSubjects with <4 views: {(n3 < 4).sum()} -> {list(n3[n3 < 4].index)}")
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote manifest -> {OUT_CSV}")


if __name__ == "__main__":
    main()
