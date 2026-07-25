"""Build the unified modelling manifest from the freshly-extracted dataset.

Reads palm_dataset_full/ (labels_all.csv + per-subject image folders) and emits
model_v2/results_full/manifest.csv with, per subject:

  * numeric one-hot tabular features (the 24 practitioner annotations)
  * three label framings:
        - y7   : the raw 7-class Prakriti (K, K-P, K-V, P-K, P-V, V-K, V-P)
        - ydom : 3-class dominant dosha (first letter of Prakriti)
        - has_K / has_P / has_V : multi-label dosha presence
  * absolute paths to the 4 view images (blank if a view is missing)
"""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "palm_dataset_full"
OUT_DIR = PROJECT_ROOT / "model_v2" / "results_full"

# decoded value -> one-hot columns (keeps SH "None" as an all-zero row, which is
# a real state: no hair on the hand, not a missing annotation)
ONEHOT = {
    "SC": ["Fair", "Medium", "Dark"],
    "ST": ["Rough", "Smooth", "Dry", "Oily"],
    "SH": ["Thick", "Thin", "Med"],
    "SHC": ["Brown", "Black"],
    "VV": ["Low", "Med", "High"],
    "PC": ["PinkRed", "PinkWhite", "Brown"],
}
VIEW_FILES = {
    "open_front": "palm_open_front.jpg",
    "open_back": "palm_open_back.jpg",
    "closed_front": "palm_closed_front.jpg",
    "closed_back": "palm_closed_back.jpg",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(DATA_DIR / "labels_all.csv")))

    feat_cols = [f"{g}_{v}" for g, vs in ONEHOT.items() for v in vs]
    out_cols = (
        ["id", "Age", "Sex_M", "Mole"]
        + feat_cols
        + ["y7", "ydom", "has_K", "has_P", "has_V", "n_views"]
        + [f"img_{v}" for v in VIEW_FILES]
    )

    out_rows = []
    for r in rows:
        rid = r["id"]
        prak = r["Prakriti"].strip()
        rec = {
            "id": rid,
            "Age": r["Age"],
            "Sex_M": 1 if str(r["Sex"]).strip().upper() == "M" else 0,
            "Mole": int(r["Mole"]) if str(r["Mole"]).strip() in ("0", "1") else 0,
        }
        for g, vs in ONEHOT.items():
            val = r[g]
            present = set(val.split("/")) if val and val != "None" else set()
            for v in vs:
                rec[f"{g}_{v}"] = 1 if v in present else 0
        rec["y7"] = prak
        rec["ydom"] = prak[0]  # first letter = dominant dosha
        rec["has_K"] = 1 if "K" in prak else 0
        rec["has_P"] = 1 if "P" in prak else 0
        rec["has_V"] = 1 if "V" in prak else 0
        rec["n_views"] = r["n_views"]
        subj_dir = DATA_DIR / rid
        for v, fn in VIEW_FILES.items():
            p = subj_dir / fn
            rec[f"img_{v}"] = str(p) if p.exists() else ""
        out_rows.append(rec)

    with open(OUT_DIR / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(out_rows)

    # quick summary
    import collections
    y7 = collections.Counter(r["y7"] for r in out_rows)
    ydom = collections.Counter(r["ydom"] for r in out_rows)
    print(f"manifest rows: {len(out_rows)}  ->  {OUT_DIR/'manifest.csv'}")
    print(f"feature columns ({len(feat_cols)}): {feat_cols}")
    print(f"\n7-class distribution: {dict(y7.most_common())}")
    print(f"3-class dominant:     {dict(ydom.most_common())}")
    for d in ("K", "P", "V"):
        n = sum(r[f"has_{d}"] for r in out_rows)
        print(f"  has_{d}: {n}/{len(out_rows)} ({n/len(out_rows)*100:.0f}%)")


if __name__ == "__main__":
    main()
