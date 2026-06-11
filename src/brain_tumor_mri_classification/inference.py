from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from brain_tumor_mri_classification.config import Config
from brain_tumor_mri_classification.model import DINOv2Classifier, _load_backbone, load_checkpoint


def build_inference_transform(cfg: Config) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )


class Predictor:
    def __init__(
        self,
        model: DINOv2Classifier,
        classes_list: list[str],
        cfg: Config,
        device: torch.device,
    ):
        self.model = model.to(device).eval()
        self.classes_list = classes_list
        self.transform = build_inference_transform(cfg)
        self.device = device

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> tuple[str, float]:
        img = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(img)
        probs = torch.softmax(logits, dim=1)
        prob, idx = torch.max(probs, dim=1)
        return self.classes_list[idx.item()], prob.item()

    @torch.inference_mode()
    def predict_top_k(
        self, image: Image.Image, k: int = 5
    ) -> list[tuple[str, float]]:
        img = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(img)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_probs, top_indices = torch.topk(probs, k)
        return [
            (self.classes_list[idx.item()], prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]


def build_predictor(
    checkpoint_path: str,
    classes_list: list[str],
    cfg: Config,
    device: torch.device,
) -> Predictor:
    backbone = _load_backbone(cfg, device)
    for param in backbone.parameters():
        param.requires_grad = False
    model = DINOv2Classifier(backbone, len(classes_list), cfg.hidden_dim, cfg.dropout)
    load_checkpoint(model, checkpoint_path, device)
    return Predictor(model, classes_list, cfg, device)
