# Configuration

Two YAML files in `configs/` are loaded automatically. CLI flags override YAML values.

```
python apps/cli/main.py train --lr 5e-4 --epochs 20
```

Files are merged left-to-right via `Config.from_yaml(*paths)`.

## `configs/data.yml`

| Key | Field | Default | Description |
|-----|-------|---------|-------------|
| `data.dataset_url` | `dataset_url` | `https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-30-classes` | Kaggle dataset URL |
| `data.json_path` | `json_path` | `data/.../DATA.json` | Dataset metadata JSON |
| `data.img_dir` | `img_dir` | `data/...` | Image root directory |

## `configs/DINOv2_model_configs.yml`

| Key | Field | Default | Description |
|-----|-------|---------|-------------|
| `checkpoint.checkpoint_dir` | `checkpoint_dir` | `ckpts/dinov2_brain_tumor` | Checkpoint save dir |
| `checkpoint.best_model_path` | `best_model_path` | `ckpts/.../best_dinov2_brain_tumor.pth` | Best model path |
| `model.backbone_name` | `backbone_name` | `dinov2_vitb14` | DINOv2 variant |
| `model.backbone_cache_path` | `backbone_cache_path` | `ckpts/...-dinov2_vitb14` | Backbone cache path |
| `model.hidden_dim` | `hidden_dim` | 512 | MLP hidden size |
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
