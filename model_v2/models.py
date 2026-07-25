"""Classifier architectures for model_v2.

WristClassifier is the single-view baseline (same architecture as
mini/training_pipeline.ipynb / mini/inference.py, kept as a control).

WristMultiViewClassifier applies a shared-weight MobileNetV2 backbone to all
4 wrist angles and fuses the embeddings before classification.
"""

import random

import torch
import torch.nn as nn
import torchvision.models as tv_models


def _mobilenet_v2_backbone(pretrained: bool) -> tuple[nn.Module, int]:
    weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    backbone = tv_models.mobilenet_v2(weights=weights)
    in_features = backbone.classifier[1].in_features  # 1280
    backbone.classifier = nn.Identity()  # feature extractor only
    return backbone, in_features


class WristClassifier(nn.Module):
    """Single-view MobileNetV2 classifier (baseline/control)."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone, in_features = _mobilenet_v2_backbone(pretrained)
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class WristMultiViewClassifier(nn.Module):
    """Shared-backbone MobileNetV2 over 4 wrist views, fused before classification.

    fusion_mode:
        "concat" - concatenate all 4 view embeddings (primary design; preserves
                   view-specific signal).
        "mean"   - average the 4 view embeddings (ablation comparison).

    view_dropout_p: probability of zeroing out 1-2 of the 4 view embeddings
    during training (regularization for a 4-view model trained on ~80 subjects,
    and robustness to app.py's incremental upload flow where not all views may
    be equally informative).
    """

    def __init__(
        self,
        num_classes: int,
        num_views: int = 4,
        fusion_mode: str = "concat",
        pretrained: bool = True,
        view_dropout_p: float = 0.3,
    ):
        super().__init__()
        if fusion_mode not in ("concat", "mean"):
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        self.num_views = num_views
        self.fusion_mode = fusion_mode
        self.view_dropout_p = view_dropout_p

        self.backbone, in_features = _mobilenet_v2_backbone(pretrained)
        self.embedding_dim = in_features

        fused_dim = in_features * num_views if fusion_mode == "concat" else in_features
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )

    def _apply_view_dropout(self, embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
        if not self.training or self.view_dropout_p <= 0:
            return embeddings
        if random.random() >= self.view_dropout_p:
            return embeddings
        n_drop = random.choice([1, 2])
        drop_idx = set(random.sample(range(self.num_views), k=min(n_drop, self.num_views - 1)))
        return [
            torch.zeros_like(e) if i in drop_idx else e
            for i, e in enumerate(embeddings)
        ]

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """views: (batch, num_views, C, H, W)"""
        batch, num_views = views.shape[0], views.shape[1]
        assert num_views == self.num_views, f"expected {self.num_views} views, got {num_views}"

        embeddings = [self.backbone(views[:, v]) for v in range(num_views)]
        embeddings = self._apply_view_dropout(embeddings)

        if self.fusion_mode == "concat":
            fused = torch.cat(embeddings, dim=1)
        else:
            fused = torch.stack(embeddings, dim=0).mean(dim=0)

        return self.head(fused)
