"""Extract frozen CNN embeddings for every palm image.

Uses ImageNet-pretrained MobileNetV2 (same backbone as the deployed mini/ model)
as a fixed feature extractor -> 1280-d vector per image. This is the right way
to use deep learning on ~90 subjects: train nothing, just reuse the pretrained
representation, then let a small classifier decide. Embeddings are cached to
model_v2/results_full/embeddings.npz so the experiments can iterate quickly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "model_v2" / "results_full"
VIEWS = ["open_front", "open_back", "closed_front", "closed_back"]

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    df = pd.read_csv(OUT / "manifest.csv")
    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    net.classifier = torch.nn.Identity()   # -> 1280-d pooled features
    net.eval()
    torch.set_grad_enabled(False)

    emb = {v: np.zeros((len(df), 1280), dtype=np.float32) for v in VIEWS}
    mask = {v: np.zeros(len(df), dtype=bool) for v in VIEWS}
    for i, row in df.iterrows():
        for v in VIEWS:
            p = row[f"img_{v}"]
            if isinstance(p, str) and p and Path(p).exists():
                x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
                emb[v][i] = net(x).squeeze(0).numpy()
                mask[v][i] = True
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(df)} subjects embedded")

    np.savez_compressed(
        OUT / "embeddings.npz",
        ids=df["id"].values,
        **{f"emb_{v}": emb[v] for v in VIEWS},
        **{f"mask_{v}": mask[v] for v in VIEWS},
    )
    present = {v: int(mask[v].sum()) for v in VIEWS}
    print(f"Saved embeddings for {len(df)} subjects -> {OUT/'embeddings.npz'}")
    print(f"views present: {present}")


if __name__ == "__main__":
    main()
