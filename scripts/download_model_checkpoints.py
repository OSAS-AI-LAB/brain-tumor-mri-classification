import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.brain_tumor_mri_classification.config import Config


def download_backbone(backbone_name: str, cache_path: str):
    if os.path.exists(cache_path):
        print(f"Backbone already cached at: {cache_path}")
        return

    print(f"Downloading DINOv2 backbone: {backbone_name}")
    backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(backbone, cache_path)
    print(f"Backbone saved to: {cache_path}")


def main():
    parser = argparse.ArgumentParser(description="Download DINOv2 backbone model to local cache.")
    parser.add_argument("--model-config", default=None, help="Path to model YAML config")
    parser.add_argument("--input", default=None, help="DINOv2 variant (e.g. dinov2_vitb14)")
    parser.add_argument("--output", default=None, help="Local cache path for backbone")
    args = parser.parse_args()

    if args.model_config and Path(args.model_config).exists():
        cfg = Config.from_yaml(args.model_config)
        backbone_name = args.input or cfg.backbone_name
        cache_path = args.output or cfg.backbone_cache_path
    else:
        backbone_name = args.input or "dinov2_vitb14"
        cache_path = args.output or "ckpts/facebookresearch--dinov2-dinov2_vitb14"

    download_backbone(backbone_name=backbone_name, cache_path=cache_path)


if __name__ == "__main__":
    main()
