import json
import os
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from brain_tumor_mri_classification.config import Config


class BrainTumorDataset(Dataset):
    def __init__(
        self,
        dataset_info: list[dict],
        class_to_idx: dict[str, int],
        transform: Optional[Callable] = None,
    ):
        self.dataset_info = dataset_info
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset_info)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.dataset_info[idx]
        img = Image.open(item["img_path"]).convert("RGB")
        label = self.class_to_idx[item["class"]]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_dataset_info(
    json_path: str, img_dir: str
) -> tuple[list[dict], list[str], dict[str, int]]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    dataset_info = []
    classes_set: set[str] = set()

    for rel_path, info in data.items():
        safe = rel_path.replace("\\", os.sep).replace("/", os.sep)
        full_path = os.path.join(img_dir, safe)
        if not os.path.exists(full_path):
            continue
        class_name = info["class"]
        classes_set.add(class_name)
        dataset_info.append({"img_path": full_path, "class": class_name})

    classes_list = sorted(classes_set)
    class_to_idx = {c: i for i, c in enumerate(classes_list)}
    return dataset_info, classes_list, class_to_idx


def build_transforms(cfg: Config):
    train_transform = transforms.Compose(
        [
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.4
            ),
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )

    return train_transform, val_transform


def create_dataloaders(
    cfg: Config,
) -> tuple[DataLoader, DataLoader, list[str], dict[str, int]]:
    dataset_info, classes_list, class_to_idx = load_dataset_info(
        cfg.json_path, cfg.img_dir
    )

    train_transform, val_transform = build_transforms(cfg)

    train_info, val_info = train_test_split(
        dataset_info,
        test_size=cfg.val_split,
        random_state=cfg.seed,
        stratify=[d["class"] for d in dataset_info],
    )

    train_dataset = BrainTumorDataset(train_info, class_to_idx, train_transform)
    val_dataset = BrainTumorDataset(val_info, class_to_idx, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, val_loader, classes_list, class_to_idx
