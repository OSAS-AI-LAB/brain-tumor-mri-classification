# User Guide

| Document | Contents |
|----------|----------|
| [Setup](setup.md) | Dependencies, dataset, backbone cache |
| [Configuration](configuration.md) | YAML config files & model variants |
| [Usage](usage.md) | CLI commands, Gradio, ONNX |
| [Augmentation](augmentation.md) | Transform details |
| [API Reference](api.md) | Module API docs |

```
├── configs/
│   ├── data.yml
│   ├── DINOv2_small_model_configs.yml
│   ├── DINOv2_large_model_configs.yml
│   └── DINOv2_giant_model_configs.yml
├── src/brain_tumor_mri_classification/
├── apps/
│   ├── cli/main.py
│   └── gradio_app/main.py
├── scripts/
│   ├── download_dataset.py
│   └── download_model_checkpoints.py
├── requirements/
│   ├── requirements-base.txt
│   └── requirements-dev.txt
├── notebooks/
├── data/dataset/
├── ckpts/
└── docs/
```
