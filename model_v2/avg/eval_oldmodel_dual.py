"""Test the original CNN model + its ORIGINAL dual-dosha logic against the sheet.

Old app rule (PROJECT_ARCHITECTURE.md, threshold 0.15) applied to the 4-view-averaged
Vata/Pitta/Kapha probabilities:
    if top1 - top3 < 0.15 -> tri  'VPK'
    elif top1 - top2 < 0.15 -> dual 'top1-top2'   (dominant-secondary)
    else -> single 'top1'
Scored (exact match) against the sheet's Prakriti labels (K, K-P, K-V, P-K, P-V, V-K, V-P).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "mini" / "model.pth"
DATA = ROOT / "palm_dataset_full"
FILES = ["palm_open_front.jpg", "palm_open_back.jpg", "palm_closed_front.jpg", "palm_closed_back.jpg"]
LETTER = {"Vata": "V", "Pitta": "P", "Kapha": "K"}
THRESH = 0.15


class WristClassifier(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        inf = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(inf, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n))

    def forward(self, x):
        return self.backbone(x)


def old_dual_label(prob, idx_to_class):
    letters = [LETTER[idx_to_class[i]] for i in range(len(prob))]
    order = np.argsort(prob)[::-1]
    t1, t2, t3 = order[0], order[1], order[2]
    if prob[t1] - prob[t3] < THRESH:
        return "VPK"
    if prob[t1] - prob[t2] < THRESH:
        return f"{letters[t1]}-{letters[t2]}"
    return letters[t1]


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = WristClassifier(ck["num_classes"]); model.load_state_dict(ck["model_state_dict"]); model.eval()
    idx_to_class = ck["idx_to_class"]
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(ck["normalization"]["mean"], ck["normalization"]["std"])])

    labels = pd.read_csv(DATA / "labels_all.csv")
    truth = {r["id"]: r["Prakriti"] for _, r in labels.iterrows()}

    rows = []
    with torch.no_grad():
        for sid, true_lbl in truth.items():
            probs = []
            for f in FILES:
                p = DATA / sid / f
                if p.is_file():
                    probs.append(torch.softmax(model(tf(Image.open(p).convert("RGB")).unsqueeze(0)), 1))
            avg = torch.stack(probs).mean(0)[0].numpy()
            rows.append((sid, true_lbl, old_dual_label(avg, idx_to_class)))

    dfp = pd.DataFrame(rows, columns=["id", "sheet", "pred"])
    exact = (dfp["sheet"] == dfp["pred"]).mean()
    dom_ok = (dfp["sheet"].str[0] == dfp["pred"].str[0]).mean()   # dominant letter matches

    print("=== OLD model + ORIGINAL dual-dosha logic  vs  the professional's sheet ===")
    print(f"  EXACT dual-label match : {exact:.1%}  ({(dfp['sheet']==dfp['pred']).sum()}/{len(dfp)})")
    print(f"  (for reference) dominant-dosha match : {dom_ok:.1%}")
    print(f"  sheet majority label K-V = {labels['Prakriti'].value_counts().max()/len(labels):.1%}")
    print("\n  what the old model outputs:")
    print(dfp["pred"].value_counts().to_string())
    print("\n  the sheet's actual labels:")
    print(dfp["sheet"].value_counts().to_string())
    ptype = dfp["pred"].apply(lambda s: "tri" if s == "VPK" else ("dual" if "-" in s else "single"))
    print("\n  old-model output types:", dict(ptype.value_counts()))


if __name__ == "__main__":
    main()
