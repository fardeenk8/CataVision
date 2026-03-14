"""
EfficientNetB0 transfer learning for binary cataract classification.
Replace classifier head with single linear layer -> 1 output (sigmoid).
"""
import torch
import torch.nn as nn
from torch import Tensor

try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
except ImportError:
    efficientnet_b0 = None
    EfficientNet_B0_Weights = None


def get_efficientnet_b0(num_classes: int = 1, pretrained: bool = True, freeze_backbone: bool = False):
    """
    EfficientNetB0 with binary classification head.
    Args:
        num_classes: 1 for binary (sigmoid).
        pretrained: use ImageNet weights.
        freeze_backbone: freeze feature extractor.
    """
    if efficientnet_b0 is None:
        raise ImportError("torchvision is required for EfficientNetB0")
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = efficientnet_b0(weights=weights)
    # In current torchvision, avgpool + flatten are in _forward_impl; classifier receives (B, C).
    # Classifier is Dropout + Linear. Replace only the head.
    in_features = None
    dropout_p = 0.2
    for m in backbone.classifier.modules():
        if isinstance(m, nn.Linear):
            in_features = m.in_features
            break
        if isinstance(m, nn.Dropout):
            dropout_p = m.p
    if in_features is None:
        in_features = 1280
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    if freeze_backbone:
        for param in backbone.features.parameters():
            param.requires_grad = False
    return backbone


class EfficientNetCataract(nn.Module):
    """Wrapper so we can expose .features for Grad-CAM."""

    def __init__(self, num_classes: int = 1, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = get_efficientnet_b0(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)

    @property
    def features(self):
        return self.backbone.features

    @property
    def classifier(self):
        return self.backbone.classifier

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)
