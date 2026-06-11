# MRI Classification

Brain tumor MRI classification using a [30-class Kaggle dataset](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-30-classes) with DINOv2 (ViT-B/14) fine-tuning in PyTorch.

```
├── configs/                           # YAML config files
├── src/brain_tumor_mri_classification/  # Core package
├── apps/cli/main.py                   # CLI
├── apps/gradio_app/main.py            # Web UI
├── scripts/
│   ├── download_dataset.py            # Dataset downloader
│   └── download_model_checkpoints.py  # Backbone downloader
├── notebooks/
├── data/dataset/                      # Dataset (gitignored)
├── ckpts/                             # Checkpoints (gitignored)
└── docs/
    ├── setup.md
    ├── configuration.md
    ├── usage.md
    ├── augmentation.md
    └── api.md
```

## Quick Start

```bash
pip install torch torchvision pillow scikit-learn matplotlib numpy gradio pyyaml kaggle

python scripts/download_dataset.py
python scripts/download_model_checkpoints.py
python apps/cli/main.py train
python apps/cli/main.py evaluate --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_brain_tumor.pth --cm confusion.png
python apps/cli/main.py inference --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_brain_tumor.pth --image path/to/mri.jpg
python apps/gradio_app/main.py
python apps/cli/main.py export-onnx --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_brain_tumor.pth --output model.onnx
```

## Documentation

| Document | Contents |
|----------|----------|
| [Setup](docs/setup.md) | Install, dataset, backbone cache |
| [Configuration](docs/configuration.md) | YAML config reference |
| [Usage](docs/usage.md) | CLI, Gradio, ONNX |
| [Augmentation](docs/augmentation.md) | Transform details |
| [API Reference](docs/api.md) | Module API docs |
