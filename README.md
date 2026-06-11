# MRI Classification

Brain tumor MRI classification using a [30-class Kaggle dataset](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-30-classes) with DINOv2 fine-tuning in PyTorch.

## Quick Start

```bash
pip install -r requirements/requirements-dev.txt

python scripts/download_dataset.py
python scripts/download_model_checkpoints.py
python apps/cli/main.py train --model-config DINOv2_large_model_configs.yml
python apps/cli/main.py evaluate --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_vitl14_brain_tumor.pth --cm confusion.png
python apps/cli/main.py inference --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_vitl14_brain_tumor.pth --image path/to/mri.jpg
python apps/cli/main.py export-onnx --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_vitl14_brain_tumor.pth --output model.onnx
python apps/gradio_app/main.py
```

## Documentation

| Document | Contents |
|----------|----------|
| [Setup](docs/setup.md) | Install, dataset, backbone cache |
| [Configuration](docs/configuration.md) | YAML config files & model variants |
| [Usage](docs/usage.md) | CLI commands, Gradio, ONNX |
| [Augmentation](docs/augmentation.md) | Transform details |
| [API Reference](docs/api.md) | Module API docs |
