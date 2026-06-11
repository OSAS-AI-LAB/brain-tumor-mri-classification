# Usage

## Training

```bash
python apps/cli/main.py train
python apps/cli/main.py train --epochs 20 --batch-size 32 --lr 5e-4 --ckpt-dir ckpts/my_experiment
```

Uses frozen DINOv2 ViT-B/14 backbone with trainable MLP head (Linear → GELU → Dropout → Linear). Best model saved by validation accuracy.

Backbone loading order:
1. Check `backbone_cache_path` for local copy
2. Download via `torch.hub.load` and cache locally

## Evaluation

```bash
python apps/cli/main.py evaluate --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_brain_tumor.pth
python apps/cli/main.py evaluate --checkpoint ckpts/.../best.pth --cm confusion.png
```

Outputs per-class precision, recall, f1-score, and support.

## Inference

```bash
python apps/cli/main.py inference --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_brain_tumor.pth --image path/to/mri.jpg
```

## ONNX Export

```bash
python apps/cli/main.py export-onnx --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_brain_tumor.pth --output model.onnx
```

Dynamic batch: input `(N, 3, 518, 518)` → output `(N, num_classes)`.

## Gradio Web App

```bash
python apps/gradio_app/main.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data-config` | `configs/data.yml` | Data config |
| `--model-config` | `configs/DINOv2_model_configs.yml` | Model config |
| `--checkpoint` | `ckpts/.../best.pth` | Checkpoint |
| `--port` | 7860 | Port |
| `--share` | False | Public link |

## Notebook

`notebooks/training_notebook.ipynb` follows the same pipeline.
