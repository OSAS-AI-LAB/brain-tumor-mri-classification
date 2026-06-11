# Configuration

Config names are resolved from `configs/` automatically. CLI flags override YAML values.

```
python apps/cli/main.py train --model-config DINOv2_large_model_configs.yml --lr 5e-4
```

## Data Config

`configs/data.yml`:

| Key | Field | Default | Description |
|-----|-------|---------|-------------|
| `data.dataset_url` | `dataset_url` | Kaggle URL | Dataset URL |
| `data.json_path` | `json_path` | `data/.../DATA.json` | Dataset metadata JSON |
| `data.img_dir` | `img_dir` | `data/...` | Image root directory |

## Model Config Variants

| Config File | Backbone | Embed Dim | Hidden Dim | Batch Size |
|-------------|----------|-----------|------------|------------|
| `DINOv2_small_model_configs.yml` | `dinov2_vits14` | 384 | 256 | 32 |
| `DINOv2_vitb14_model_configs.yml` | `dinov2_vitb14` | 768 | 512 | 16 |
| `DINOv2_large_model_configs.yml` | `dinov2_vitl14` | 1024 | 768 | 8 |
| `DINOv2_giant_model_configs.yml` | `dinov2_vitg14` | 1536 | 1024 | 4 |

### Common Fields

| Key | Field | Default | Description |
|-----|-------|---------|-------------|
| `checkpoint.checkpoint_dir` | `checkpoint_dir` | `ckpts/dinov2_brain_tumor` | Checkpoint save dir |
| `checkpoint.best_model_path` | `best_model_path` | `ckpts/.../best_*.pth` | Best model path |
| `model.backbone_name` | `backbone_name` | *per variant* | DINOv2 variant |
| `model.backbone_cache_path` | `backbone_cache_path` | `ckpts/...-*` | Backbone cache path |
| `model.hidden_dim` | `hidden_dim` | *per variant* | MLP hidden size |
| `model.dropout` | `dropout` | 0.4 | Dropout rate |
| `training.img_size` | `img_size` | 518 | Resize dimension |
| `training.batch_size` | `batch_size` | *per variant* | Batch size |
| `training.epochs` | `epochs` | 15 | Training epochs |
| `training.lr` | `lr` | 1e-3 | Learning rate |
| `training.weight_decay` | `weight_decay` | 1e-4 | AdamW weight decay |
| `training.val_split` | `val_split` | 0.2 | Validation fraction |
| `training.seed` | `seed` | 42 | Random seed |
| `training.num_workers` | `num_workers` | 2 | DataLoader workers |
| `training.pin_memory` | `pin_memory` | true | DataLoader pin_memory |
| `normalization.mean` | `mean` | (0.485, 0.456, 0.406) | ImageNet mean |
| `normalization.std` | `std` | (0.229, 0.224, 0.225) | ImageNet std |
