import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from brain_tumor_mri_classification.config import Config
from brain_tumor_mri_classification.model import DINOv2Classifier, _load_backbone, build_model, load_checkpoint

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def export_to_onnx(
    checkpoint_path: str,
    classes_list: list[str],
    cfg: Config,
    output_path: str = "model.onnx",
    device: torch.device = torch.device("cpu"),
    opset_version: int = 17,
):
    backbone = _load_backbone(cfg, device)
    for param in backbone.parameters():
        param.requires_grad = False
    model = DINOv2Classifier(backbone, len(classes_list), cfg.hidden_dim, cfg.dropout)
    load_checkpoint(model, checkpoint_path, device)
    model.to(device).eval()

    dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=opset_version,
        )

    logger.info("ONNX model saved to %s", output_path)
    return str(output_path)
