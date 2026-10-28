"""End-to-end robustness check of the DEPLOYED averaging inference on all 92 subjects,
using mini/model_v2.pth. Confirms every subject (including the two 3-view subjects)
yields a valid averaged distribution, and that missing views are handled. Reported
accuracy is on TRAINING data (the final model saw all 92) — it is a plumbing check,
NOT a generalization number (that is the CV balanced-acc ~0.24)."""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "mini" / "model_v2.pth"
DATA = ROOT / "palm_dataset_full"
FILES = {"palm_open_front": "palm_open_front.jpg", "palm_open_back": "palm_open_back.jpg",
         "palm_closed_front": "palm_closed_front.jpg", "palm_closed_back": "palm_closed_back.jpg"}


class WristClassifier(nn.Module):  # architecture used by the archived checkpoint
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        in_f = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f, 256), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.backbone(x))


def main():
    import csv
    ckpt = torch.load(CKPT, map_location="cpu")
    model = WristClassifier(ckpt["num_classes"]); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(ckpt["normalization"]["mean"], ckpt["normalization"]["std"])])
    idx_to_class = ckpt["idx_to_class"]

    labels = {}
    with open(DATA / "labels_all.csv") as f:
        for r in csv.DictReader(f):
            labels[r["id"]] = r["Prakriti"]

    n, correct, bad_sum, three_view = 0, 0, 0, []
    for sid, true in labels.items():
        present = [v for v in ckpt["view_order"] if (DATA / sid / FILES[v]).is_file()]
        if len(present) < 4:
            three_view.append((sid, len(present)))
        probs = []
        with torch.no_grad():
            for v in present:
                img = Image.open(DATA / sid / FILES[v]).convert("RGB")
                probs.append(torch.softmax(model(tf(img).unsqueeze(0)), dim=1))
        avg = torch.stack(probs).mean(0)[0]
        if abs(float(avg.sum()) - 1.0) > 1e-4:
            bad_sum += 1
        pred = idx_to_class[int(avg.argmax())]
        correct += (pred == true)
        n += 1

    print(f"subjects scored            : {n}/92")
    print(f"invalid prob distributions : {bad_sum}  (must be 0)")
    print(f"3-view subjects handled    : {three_view}")
    print(f"training-set accuracy      : {correct}/{n} = {correct/n:.3f}  (plumbing only, not generalization)")


if __name__ == "__main__":
    main()
