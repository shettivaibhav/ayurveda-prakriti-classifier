"""Test the original CNN model (mini/model.pth, trained on the project's own labels)
against the Ayurveda professional's sheet labels.

Runs the old single-view MobileNet on each of a subject's 4 views and AVERAGES the
softmaxes (its original design), predicts Vata/Pitta/Kapha, and compares to the
sheet's DOMINANT dosha (labels_all.csv, from palm_dataset_26 June 26.xlsx).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "mini" / "model.pth"
DATA = ROOT / "palm_dataset_full"
FILES = ["palm_open_front.jpg", "palm_open_back.jpg", "palm_closed_front.jpg", "palm_closed_back.jpg"]
MAP = {"Vata": "V", "Pitta": "P", "Kapha": "K"}


class WristClassifier(nn.Module):  # matches mini/inference.py
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        inf = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(inf, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.backbone(x)


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"Old model's reported score on the project's own data: "
          f"val_acc={ck.get('best_val_acc')}, val_f1={ck.get('best_val_f1')}")
    model = WristClassifier(ck["num_classes"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    idx_to_class = ck["idx_to_class"]
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(ck["normalization"]["mean"], ck["normalization"]["std"])])

    labels = pd.read_csv(DATA / "labels_all.csv")
    dom = {r["id"]: r["Prakriti"][0] for _, r in labels.iterrows()}   # sheet dominant dosha

    y_true, y_pred = [], []
    with torch.no_grad():
        for sid, true_dom in dom.items():
            probs = []
            for f in FILES:
                p = DATA / sid / f
                if p.is_file():
                    x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
                    probs.append(torch.softmax(model(x), 1))
            avg = torch.stack(probs).mean(0)[0]
            pred_name = idx_to_class[int(avg.argmax())]
            y_true.append(true_dom)
            y_pred.append(MAP[pred_name])

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = (y_true == y_pred).mean()
    bal = balanced_accuracy_score(y_true, y_pred)
    print(f"\n=== OLD model vs the PROFESSIONAL's sheet (dominant dosha, all 92 subjects) ===")
    print(f"  accuracy         : {acc:.1%}  ({(y_true==y_pred).sum()}/{len(y_true)})")
    print(f"  balanced accuracy: {bal:.1%}   (majority baseline = {pd.Series(y_true).value_counts().max()/len(y_true):.1%})")

    lab = ["K", "P", "V"]
    cm = confusion_matrix(y_true, y_pred, labels=lab)
    print("\n  confusion (rows=sheet truth, cols=old-model prediction):")
    print("        pred:  " + "   ".join(lab))
    for i, l in enumerate(lab):
        print(f"    true {l}:   " + "   ".join(f"{cm[i][j]:3d}" for j in range(3)))
    print("\n  sheet distribution:", dict(pd.Series(y_true).value_counts()))
    print("  old-model preds   :", dict(pd.Series(y_pred).value_counts()))


if __name__ == "__main__":
    main()
