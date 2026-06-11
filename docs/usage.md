# Usage

## CLI Reference

`apps/cli/main.py` — Brain tumor MRI classification CLI with subcommands.

### Global Flags

Each subcommand accepts its own `--data-config` and `--model-config` flags. Config names are resolved from the `configs/` folder automatically. All paths support short names (`data.yml` works without a full path).

### `train`

```bash
# Default (large)
python apps/cli/main.py train

# Custom model variant
python apps/cli/main.py train --model-config DINOv2_large_model_configs.yml

# With overrides
python apps/cli/main.py train --data-config data.yml --model-config DINOv2_large_model_configs.yml --epochs 20 --batch-size 8 --lr 5e-4
```

Uses frozen DINOv2 backbone with trainable MLP head. Best model saved by validation accuracy.

| Flag | Default | Description |
|------|---------|-------------|
| `--data-config` | `data.yml` | Data YAML config (resolved from `configs/`) |
| `--model-config` | `DINOv2_large_model_configs.yml` | Model YAML config |
| `--json` | *from config* | Override JSON path |
| `--img-dir` | *from config* | Override image directory |
| `--ckpt-dir` | *from config* | Override checkpoint save directory |
| `--batch-size` | *from config* | Override batch size |
| `--epochs` | *from config* | Override epochs |
| `--lr` | *from config* | Override learning rate |

Backbone loading order:
1. Check `backbone_cache_path` for local copy
2. Download via `torch.hub.load` and cache locally

### `evaluate`

```bash
python apps/cli/main.py evaluate --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_vitl14_brain_tumor.pth
python apps/cli/main.py evaluate --checkpoint ckpts/.../best.pth --cm confusion.png
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data-config` | `data.yml` | Data YAML config |
| `--checkpoint` | *required* | Path to model checkpoint `.pth` |
| `--json` | *from config* | Override JSON path |
| `--img-dir` | *from config* | Override image directory |
| `--cm` | `None` | Save confusion matrix PNG to path |

Outputs per-class precision, recall, f1-score, and support.

### `inference`

```bash
python apps/cli/main.py inference --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_vitl14_brain_tumor.pth --image path/to/mri.jpg
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data-config` | `data.yml` | Data YAML config |
| `--model-config` | `DINOv2_large_model_configs.yml` | Model YAML config |
| `--checkpoint` | *required* | Path to model checkpoint `.pth` |
| `--image` | *required* | Path to input MRI image |
| `--json` | *from config* | Override JSON path |
| `--img-dir` | *from config* | Override image directory |
| `--classes` | *auto* | Comma-separated class names (inferred from data if omitted) |
| `--img-size` | *from config* | Override image resize dimension |

### `export-onnx`

```bash
python apps/cli/main.py export-onnx --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_vitl14_brain_tumor.pth --output model.onnx
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data-config` | `data.yml` | Data YAML config |
| `--model-config` | `DINOv2_large_model_configs.yml` | Model YAML config |
| `--checkpoint` | *required* | Path to model checkpoint `.pth` |
| `--output` | `model.onnx` | Output ONNX file path |
| `--json` | *from config* | Override JSON path |
| `--img-dir` | *from config* | Override image directory |
| `--classes` | *auto* | Comma-separated class names |

Dynamic batch: input `(N, 3, 518, 518)` → output `(N, num_classes)`.

## Gradio Web App

```bash
python apps/gradio_app/main.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data-config` | `data.yml` | Data config |
| `--model-config` | `DINOv2_large_model_configs.yml` | Model config |
| `--checkpoint` | `ckpts/.../best_dinov2_vitl14_brain_tumor.pth` | Checkpoint |
| `--port` | 7860 | Port |
| `--share` | False | Public link |

## Notebook

`notebooks/training_notebook.ipynb` follows the same pipeline using `facebook/dinov2-large` from Hugging Face.
