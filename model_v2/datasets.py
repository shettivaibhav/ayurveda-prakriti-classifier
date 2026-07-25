"""PyTorch Dataset classes for model_v2.

Adapted from WristDataset in mini/training_pipeline.ipynb, extended to
support 4-view subjects and 7-class Prakriti labels.
"""

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from data import LABEL_COL, VIEW_ORDER

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_class_mappings(labels: pd.Series) -> tuple[dict, dict]:
    classes = sorted(labels.unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


class WristSingleViewDataset(Dataset):
    """One image per subject (palm_open_back) - baseline/control."""

    def __init__(self, dataframe, class_to_idx, transform=None, view="palm_open_back"):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.view = view
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.loc[idx, self.view]
        label = self.class_to_idx[self.data.loc[idx, LABEL_COL]]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


class WristMultiViewDataset(Dataset):
    """All 4 wrist-angle images per subject, stacked as (num_views, C, H, W)."""

    def __init__(self, dataframe, class_to_idx, transform=None, view_order=VIEW_ORDER):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.view_order = view_order
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        label = self.class_to_idx[row[LABEL_COL]]

        views = []
        for view in self.view_order:
            image = Image.open(row[view]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            views.append(image)

        return torch.stack(views, dim=0), label
