"""Extract frozen (ImageNet / self-supervised) embeddings for every hand image,
from several backbones, so the classical-ML averaging experiments can compare
MobileNetV2 against stronger feature extractors without any fine-tuning.

CPU-only environment => frozen features are the right lever: run each image
through a pretrained backbone once, cache the pooled vector. A small classifier
then decides. One .npz per backbone, row-aligned to build_image_dataset.py's
manifest (same order), holding: emb (N, dim), subject_id (N,), view (N,), label (N,).

Backbones (each wrapped in try/except so a failed download just skips it):
    mobilenet_v2   1280-d   (weights already cached offline)
    resnet50       2048-d
    efficientnet_b0 1280-d
    convnext_tiny   768-d
    dinov2_vits14   384-d   (self-supervised ViT via torch.hub; strong for transfer)
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from build_image_dataset import build_image_df

OUT_DIR = Path(__file__).resolve().parent / "features"
OUT_DIR.mkdir(exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# One consistent transform for every backbone (all consume 224x224, ImageNet norm).
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---- backbone factories: return (module, out_dim); module maps (B,3,224,224) -> (B,dim)
def _mobilenet_v2():
    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    net.classifier = torch.nn.Identity()
    return net, 1280


def _resnet50():
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    return net, 2048


def _efficientnet_b0():
    net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    net.classifier = torch.nn.Identity()
    return net, 1280


def _convnext_tiny():
    net = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    # classifier = [LayerNorm2d, Flatten, Linear]; drop the Linear, keep norm+flatten -> 768
    net.classifier[2] = torch.nn.Identity()
    return net, 768


def _dinov2_vits14():
    net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    return net, 384


BACKBONES = {
    "mobilenet_v2": _mobilenet_v2,
    "resnet50": _resnet50,
    "efficientnet_b0": _efficientnet_b0,
    "convnext_tiny": _convnext_tiny,
    "dinov2_vits14": _dinov2_vits14,
}


@torch.no_grad()
def extract(name, factory, df):
    out_path = OUT_DIR / f"{name}.npz"
    if out_path.exists():
        print(f"[{name}] cached -> {out_path.name}")
        return True
    try:
        net, dim = factory()
    except Exception as e:  # download / hub failure -> skip this backbone
        print(f"[{name}] SKIPPED ({type(e).__name__}: {e})")
        return False
    net.eval()

    emb = np.zeros((len(df), dim), dtype=np.float32)
    for i, p in enumerate(df["img_path"]):
        x = TF(Image.open(p).convert("RGB")).unsqueeze(0)
        emb[i] = net(x).squeeze(0).cpu().numpy()
        if (i + 1) % 60 == 0:
            print(f"[{name}] {i + 1}/{len(df)}")
    np.savez_compressed(
        out_path,
        emb=emb,
        subject_id=df["subject_id"].to_numpy(),
        view=df["view"].to_numpy(),
        label=df["Prakriti"].to_numpy(),
    )
    print(f"[{name}] saved {emb.shape} -> {out_path.name}")
    return True


def main():
    df = build_image_df()
    print(f"Images: {len(df)}\n")
    ok = []
    for name, factory in BACKBONES.items():
        if extract(name, factory, df):
            ok.append(name)
    print(f"\nBackbones available: {ok}")


if __name__ == "__main__":
    main()
