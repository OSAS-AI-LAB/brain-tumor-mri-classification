# MRI Classification

A compact brain tumor MRI classification project using a [30-class Kaggle dataset](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-30-classes) with DINOv2 (ViT-B/14) fine-tuning in PyTorch.

## Project Structure

```
├── configs/
│   ├── data.yml                 # Data paths, checkpoint paths, normalization
│   └── DINOv2_model_configs.yml # Model architecture & training hyperparams
├── src/brain_tumor_mri_classification/  # Core Python package
│   ├── config.py          # Config dataclass + multi-YAML loader
│   ├── dataset.py         # Dataset, transforms, dataloaders
│   ├── model.py           # DINOv2Classifier definition
│   ├── trainer.py         # Training & validation loops
│   ├── evaluate.py        # Evaluation metrics & confusion matrix
│   └── inference.py       # Predictor class for inference
├── apps/
│   ├── cli/main.py        # CLI: train / evaluate / inference
│   └── gradio_app/main.py # Web UI for interactive inference
├── notebooks/
│   └── training_notebook.ipynb  # Original training notebook
├── scripts/
│   └── download_kaggle_dataset.py  # Dataset downloader
├── data/dataset/          # Dataset root (gitignored)
├── ckpts/                 # Model checkpoints (gitignored)
└── docs/                  # Documentation
```

## Quick Start

```bash
pip install torch torchvision pillow scikit-learn matplotlib numpy gradio pyyaml

python scripts/download_kaggle_dataset.py

# Train
python apps/cli/main.py train

# Evaluate
python apps/cli/main.py evaluate \
    --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_brain_tumor.pth --cm confusion.png

# Single image inference
python apps/cli/main.py inference \
    --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_brain_tumor.pth --image path/to/mri.jpg

# Web UI
python apps/gradio_app/main.py
```

## Configuration

Settings are split into two YAML files. Both are loaded automatically from `configs/`.

| File | Contents |
|------|----------|
| `configs/data.yml` | Data paths |
| `configs/DINOv2_model_configs.yml` | Checkpoint paths, model architecture, training hyperparameters, normalization |

CLI flags override YAML values when both are provided.

```bash
python apps/cli/main.py train --data-config configs/data.yml --model-config configs/DINOv2_model_configs.yml --epochs 20
```

See [docs/guide.md](docs/guide.md) for full usage and [docs/api.md](docs/api.md) for API reference.
