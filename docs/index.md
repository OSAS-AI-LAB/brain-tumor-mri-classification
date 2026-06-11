# User Guide

| Document | Contents |
|----------|----------|
| [Setup](setup.md) | Dependencies, dataset download, backbone cache |
| [Configuration](configuration.md) | YAML config reference |
| [Usage](usage.md) | CLI, Gradio, ONNX export |
| [Augmentation](augmentation.md) | Transform details |
| [API Reference](api.md) | Module API docs |

```
├── configs/
│   ├── data.yml
│   └── DINOv2_model_configs.yml
├── src/brain_tumor_mri_classification/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── trainer.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
├── apps/
│   ├── cli/main.py
│   └── gradio_app/main.py
├── scripts/
│   ├── download_dataset.py
│   └── download_model_checkpoints.py
├── notebooks/
├── data/dataset/
├── ckpts/
└── docs/
    ├── index.md
    ├── setup.md
    ├── configuration.md
    ├── usage.md
    ├── augmentation.md
    └── api.md
```
