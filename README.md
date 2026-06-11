# Brain Tumor MRI Classification

**State-of-the-art 30-class brain tumor classification** from MRI scans using **DINOv2** (Vision Transformer) fine-tuning in PyTorch.

Leverages the [Brain Tumor MRI Images - 30 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-30-classes) dataset for fine-grained classification of tumor types and subtypes.

---

## ✨ Features

- **DINOv2 backbone** (ViT-L/14 & smaller variants) with LoRA / full fine-tuning support
- High-accuracy multi-class classification (30 classes)
- Comprehensive data augmentation pipeline tailored for medical imaging
- Easy-to-use CLI + Gradio web interface
- ONNX export for deployment
- Reproducible training with YAML-based configuration
- Confusion matrix & evaluation tools

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone <your-repo-url>
cd mri-classification
pip install -r requirements/requirements-dev.txt

# 2. Download dataset and model checkpoints
python scripts/download_dataset.py
python scripts/download_model_checkpoints.py

# 3. Train
python apps/cli/main.py train --model-config configs/DINOv2_large_model_configs.yml

# 4. Evaluate
python apps/cli/main.py evaluate \
  --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_vitl14_brain_tumor.pth \
  --cm confusion_matrix.png

# 5. Inference
python apps/cli/main.py inference \
  --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_vitl14_brain_tumor.pth \
  --image path/to/your_mri.jpg

# 6. Export to ONNX
python apps/cli/main.py export-onnx \
  --checkpoint ckpts/dinov2_brain_tumor/best_dinov2_vitl14_brain_tumor.pth \
  --output model.onnx

# 7. Launch Gradio demo
python apps/gradio_app/main.py
```

---

## 📖 Documentation

| Document              | Description |
|-----------------------|-------------|
| **[Setup](docs/setup.md)**          | Installation, dataset preparation, and backbone caching |
| **[Configuration](docs/configuration.md)** | YAML configs, model variants, training hyperparameters |
| **[Usage](docs/usage.md)**          | CLI reference, Gradio app, ONNX export & deployment |
| **[Augmentation](docs/augmentation.md)** | Medical-specific transforms and augmentation strategy |
| **[API Reference](docs/api.md)**    | Core module documentation |

---

## Project Structure (high-level)

```
mri-classification/
├── configs/              # Training & model configurations
├── scripts/              # Dataset & checkpoint download
├── apps/
│   ├── cli/              # Command-line interface
│   └── gradio_app/       # Interactive web demo
├── src/                  # Core training, models, and utils
├── ckpts/                # Trained model & backbone caches
├── examples/             # Inference examples
├── docs/                 # Full documentation
└── requirements/         # Dependency files
```

---

**Built with ❤️ for medical imaging research**

Contributions, issues, and feedback are welcome!
