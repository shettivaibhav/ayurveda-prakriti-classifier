"""Headless check for the archived averaged MobileNetV2 checkpoint."""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "mini" / "model_v2.pth"


class WristClassifier(nn.Module):  # architecture used by the archived checkpoint
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        in_f = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(in_f, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.backbone(x))


def main():
    ckpt = torch.load(CKPT, map_location="cpu")
    print("checkpoint keys:", sorted(k for k in ckpt if k != "model_state_dict"))
    print("num_classes:", ckpt["num_classes"], " idx_to_class:", ckpt["idx_to_class"])
    print("inference_mode:", ckpt.get("inference_mode"), " view_order:", ckpt["view_order"])

    model = WristClassifier(num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])  # strict=True -> keys must match app.py
    model.eval()
    print("\nstate_dict loaded strictly OK (arch matches app.py)")

    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(ckpt["normalization"]["mean"], ckpt["normalization"]["std"])])

    subj = ROOT / "palm_dataset_full" / "1001_male"
    files = {"palm_open_front": "palm_open_front.jpg", "palm_open_back": "palm_open_back.jpg",
             "palm_closed_front": "palm_closed_front.jpg", "palm_closed_back": "palm_closed_back.jpg"}
    probs = []
    with torch.no_grad():
        for v in ckpt["view_order"]:
            img = Image.open(subj / files[v]).convert("RGB")
            probs.append(torch.softmax(model(tf(img).unsqueeze(0)), dim=1))
    avg = torch.stack(probs).mean(0)[0]

    idx_to_class = ckpt["idx_to_class"]
    dist = {idx_to_class[i]: float(avg[i]) for i in range(len(idx_to_class))}
    top = max(dist, key=dist.get)
    print(f"\nSubject 1001_male (true label K-V):")
    print(f"  prob sum = {avg.sum():.4f}  (should be ~1.0)")
    print(f"  predicted = {top}  ({dist[top]:.1%})")
    for c, p in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {c:5s} {p:.1%}")


if __name__ == "__main__":
    main()
