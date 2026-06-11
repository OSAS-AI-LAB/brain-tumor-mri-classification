from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    dataset_url: str = "https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-30-classes"
    json_path: str = "data/dataset/brain-tumor-mri-images-30-classes/DATA.json"
    img_dir: str = "data/dataset/brain-tumor-mri-images-30-classes"
    checkpoint_dir: str = "ckpts/dinov2_brain_tumor"
    best_model_path: str = "ckpts/dinov2_brain_tumor/best_dinov2_brain_tumor.pth"

    img_size: int = 518
    batch_size: int = 16
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 512
    dropout: float = 0.4
    val_split: float = 0.2
    seed: int = 42

    num_workers: int = 2
    pin_memory: bool = True

    backbone_name: str = "dinov2_vitb14"
    backbone_cache_path: str = "ckpts/facebookresearch--dinov2-dinov2_vitb14"

    mean: tuple = field(default_factory=lambda: (0.485, 0.456, 0.406))
    std: tuple = field(default_factory=lambda: (0.229, 0.224, 0.225))

    @classmethod
    def from_yaml(cls, *paths: str, overrides: Optional[dict] = None) -> "Config":
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")

        configs_dir = Path(__file__).resolve().parents[2] / "configs"

        resolved = []
        for p in paths:
            path = Path(p)
            if path.exists():
                resolved.append(str(path))
            else:
                alt = configs_dir / p
                if alt.exists():
                    resolved.append(str(alt))

        mapping = {
            "dataset_url": ("data", "dataset_url"),
            "json_path": ("data", "json_path"),
            "img_dir": ("data", "img_dir"),
            "checkpoint_dir": ("checkpoint", "checkpoint_dir"),
            "best_model_path": ("checkpoint", "best_model_path"),
            "backbone_name": ("model", "backbone_name"),
            "backbone_cache_path": ("model", "backbone_cache_path"),
            "hidden_dim": ("model", "hidden_dim"),
            "dropout": ("model", "dropout"),
            "img_size": ("training", "img_size"),
            "batch_size": ("training", "batch_size"),
            "epochs": ("training", "epochs"),
            "lr": ("training", "lr"),
            "weight_decay": ("training", "weight_decay"),
            "val_split": ("training", "val_split"),
            "seed": ("training", "seed"),
            "num_workers": ("training", "num_workers"),
            "pin_memory": ("training", "pin_memory"),
            "mean": ("normalization", "mean"),
            "std": ("normalization", "std"),
        }

        cfg = cls()

        for path in resolved:
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            for field_name, keys in mapping.items():
                value = raw
                for k in keys:
                    if isinstance(value, dict):
                        value = value.get(k)
                    else:
                        value = None
                        break
                if value is not None:
                    setattr(cfg, field_name, value)

        if overrides:
            for k, v in overrides.items():
                if v is not None and hasattr(cfg, k):
                    setattr(cfg, k, v)

        return cfg
