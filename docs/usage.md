# Usage

Each CLI subcommand accepts its own `--data-config` and `--model-config` flags. Config names are resolved from the `configs/` folder automatically.

## Training

```bash
# Default (vitb14)
python apps/cli/main.py train

# Custom model variant
python apps/cli/main.py train --model-config DINOv2_large_model_configs.yml

# With overrides
python apps/cli/main.py train --model-config DINOv2_large_model_configs.yml --epochs 20 --batch-size 8 --lr 5e-4
```

Uses frozen DINOv2 backbone with trainable MLP head. Best model saved by validation accuracy.

Backbone loading order:
1. Check `backbone_cache_path` for local copy
2. Download via `torch.hub.load` and cache locally

## Evaluation

```bash
python apps/cli/main.py evaluate --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_vitb14_brain_tumor.pth
python apps/cli/main.py evaluate --checkpoint ckpts/.../best.pth --cm confusion.png
```

Outputs per-class precision, recall, f1-score, and support.

## Inference

```bash
python apps/cli/main.py inference --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_vitb14_brain_tumor.pth --image path/to/mri.jpg
```

## ONNX Export

```bash
python apps/cli/main.py export-onnx --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_vitb14_brain_tumor.pth --output model.onnx
```

Dynamic batch: input `(N, 3, 518, 518)` → output `(N, num_classes)`.

## Gradio Web App

```bash
python apps/gradio_app/main.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data-config` | `data.yml` | Data config |
| `--model-config` | `DINOv2_vitb14_model_configs.yml` | Model config |
| `--checkpoint` | `ckpts/.../best_dinov2_vitb14_brain_tumor.pth` | Checkpoint |
| `--port` | 7860 | Port |
| `--share` | False | Public link |

## Notebook

`notebooks/training_notebook.ipynb` follows the same pipeline.
