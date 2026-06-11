import os

import torch
import torch.nn as nn

from brain_tumor_mri_classification.config import Config


class DINOv2Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, hidden_dim: int = 512, dropout: float = 0.4):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)["x_norm_clstoken"]
        return self.classifier(features)


def _load_backbone(cfg: Config, device: torch.device) -> nn.Module:
    local_path = cfg.backbone_cache_path
    if os.path.exists(local_path):
        return torch.load(local_path, map_location=device, weights_only=False)
    backbone = torch.hub.load("facebookresearch/dinov2", cfg.backbone_name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    torch.save(backbone, local_path)
    return backbone


def build_model(cfg: Config, num_classes: int, device: torch.device) -> DINOv2Classifier:
    backbone = _load_backbone(cfg, device)
    for param in backbone.parameters():
        param.requires_grad = False
    model = DINOv2Classifier(backbone, num_classes, cfg.hidden_dim, cfg.dropout)
    return model.to(device)


def load_checkpoint(
    model: DINOv2Classifier, checkpoint_path: str, device: torch.device
) -> DINOv2Classifier:
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model
