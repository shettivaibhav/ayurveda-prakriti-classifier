"""End-to-end check of the archived colour model:
load model_color.joblib, feed real subjects' back+front PIL images + age/sex through
palm_color.extract_features, run predict_proba. Confirms it works and reports
training-set agreement (sanity, not generalisation)."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MINI = ROOT / "mini"
DATA = ROOT / "palm_dataset_full"
sys.path.insert(0, str(MINI))
from palm_color import extract_features  # noqa: E402


def main():
    bundle = joblib.load(MINI / "model_color.joblib")
    print("bundle keys:", list(bundle.keys()))
    print("classes:", bundle["classes"], " target:", bundle["target"])
    print("skin_view:", bundle["skin_view"], " palm_view:", bundle["palm_view"])

    labels = pd.read_csv(DATA / "labels_all.csv")
    gender_of = {r["id"]: r["id"].split("_")[1] for _, r in labels.iterrows()}
    dom_of = {r["id"]: r["Prakriti"][0] for _, r in labels.iterrows()}

    n, correct, bad = 0, 0, 0
    shown = 0
    for _, r in labels.iterrows():
        sid = r["id"]
        back = DATA / sid / "palm_open_back.jpg"
        front = DATA / sid / "palm_open_front.jpg"
        if not (back.is_file() and front.is_file()):
            continue
        sex_m = 1 if gender_of[sid] == "male" else 0
        feats = extract_features(Image.open(back), Image.open(front), r["Age"], sex_m).reshape(1, -1)
        proba = bundle["model"].predict_proba(feats)[0]
        probs = {c: float(p) for c, p in zip(bundle["classes"], proba)}
        if abs(sum(probs.values()) - 1.0) > 1e-6:
            bad += 1
        pred = max(probs, key=probs.get)
        correct += (pred == dom_of[sid])
        n += 1
        if shown < 3:
            print(f"  {sid}: true={r['Prakriti']} (dom {dom_of[sid]}) -> pred {pred}  {probs}")
            shown += 1

    print(f"\nsubjects scored: {n}   invalid distributions: {bad}")
    print(f"training-set dominant-dosha agreement: {correct}/{n} = {correct/n:.1%} "
          f"(sanity only; honest CV number is 67%)")


if __name__ == "__main__":
    main()
