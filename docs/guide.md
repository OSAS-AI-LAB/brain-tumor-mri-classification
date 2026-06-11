# User Guide

## Setup

### Dependencies

```bash
pip install torch torchvision pillow scikit-learn matplotlib numpy gradio pyyaml
```

### Dataset

```bash
python scripts/download_kaggle_dataset.py
```

Extracts into `data/dataset/brain-tumor-mri-images-30-classes/`.

## Configuration

Settings are split across two YAML files in `configs/`:

| File | Purpose |
|------|---------|
| `configs/data.yml` | Data paths |
| `configs/DINOv2_model_configs.yml` | Checkpoint paths, model architecture, training hyperparameters, normalization |

Both files are loaded automatically by the CLI and Gradio app. CLI flags override YAML values when both are provided:

```bash
python apps/cli/main.py train --data-config configs/data.yml --model-config configs/DINOv2_model_configs.yml --lr 5e-4 --epochs 20
```

## Commands

### Training

```bash
python apps/cli/main.py train
```

Custom overrides:

```bash
python apps/cli/main.py train \
    --data-config configs/data.yml \
    --model-config configs/DINOv2_model_configs.yml \
    --epochs 20 \
    --batch-size 32 \
    --lr 5e-4 \
    --ckpt-dir ckpts/my_experiment
```

The model uses a frozen DINOv2 ViT-B/14 backbone with a trainable MLP head. Only the head parameters are updated. The best model (by validation accuracy) is saved to the checkpoint directory.

### Evaluation

```bash
python apps/cli/main.py evaluate \
    --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_brain_tumor.pth
```

Save a confusion matrix:

```bash
python apps/cli/main.py evaluate \
    --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_brain_tumor.pth \
    --cm confusion_matrix.png
```

### Inference

```bash
python apps/cli/main.py inference \
    --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_brain_tumor.pth \
    --image path/to/mri_scan.jpg
```

### Gradio Web App

```bash
python apps/gradio_app/main.py
```

Opens at `http://127.0.0.1:7860`. Use `--port` and `--share` flags to customize.

### Notebook

The original training notebook at `notebooks/training_notebook.ipynb` follows the same pipeline.

## Configuration Reference

Both files are loaded via `Config.from_yaml(*paths)` which merges them left-to-right.

### `configs/data.yml`

| YAML Key | Config Field | Default | Description |
|----------|--------------|---------|-------------|
| `data.json_path` | `json_path` | `data/.../DATA.json` | Dataset metadata JSON |
| `data.img_dir` | `img_dir` | `data/...` | Image root directory |

### `configs/DINOv2_model_configs.yml`

| YAML Key | Config Field | Default | Description |
|----------|--------------|---------|-------------|
| `checkpoint.checkpoint_dir` | `checkpoint_dir` | `ckpts/dinov2_brain_tumor` | Checkpoint save dir |
| `checkpoint.best_model_path` | `best_model_path` | `ckpts/.../best_dinov2_brain_tumor.pth` | Best model path |
| `model.backbone_name` | `backbone_name` | `dinov2_vitb14` | DINOv2 variant |
| `model.backbone_cache_path` | `backbone_cache_path` | `ckpts/facebookresearch--dinov2-dinov2_vitb14` | Backbone cache path |
| `model.hidden_dim` | `hidden_dim` | 512 | MLP hidden layer size |
| `model.dropout` | `dropout` | 0.4 | Dropout rate |
| `training.img_size` | `img_size` | 518 | Resize dimension |
| `training.batch_size` | `batch_size` | 16 | Batch size |
| `training.epochs` | `epochs` | 15 | Training epochs |
| `training.lr` | `lr` | 1e-3 | Learning rate |
| `training.weight_decay` | `weight_decay` | 1e-4 | AdamW weight decay |
| `training.val_split` | `val_split` | 0.2 | Validation fraction |
| `training.seed` | `seed` | 42 | Random seed |
| `training.num_workers` | `num_workers` | 2 | DataLoader workers |
| `training.pin_memory` | `pin_memory` | true | DataLoader pin_memory |
| `normalization.mean` | `mean` | (0.485, 0.456, 0.406) | ImageNet mean |
| `normalization.std` | `std` | (0.229, 0.224, 0.225) | ImageNet std |

## Augmentation

Training transforms:
- Resize to 518×518
- Random horizontal flip (50%)
- Color jitter (brightness ±15%, contrast ±15%) with 40% probability
- Random affine (translate ±5%, scale 0.95–1.05)
- ImageNet normalization

Validation/Inference transforms:
- Resize to 518×518
- ImageNet normalization
