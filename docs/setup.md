# Setup

## Dependencies

```bash
pip install torch torchvision pillow scikit-learn matplotlib numpy gradio pyyaml kaggle
```

## Dataset

Authenticate Kaggle CLI first ([docs](https://www.kaggle.com/docs/api)).

```bash
# With config
python scripts/download_dataset.py --data-config configs/data.yml

# Without config (direct args)
python scripts/download_dataset.py --input fernando2rad/brain-tumor-mri-images-30-classes --output data/dataset/brain-tumor-mri-images-30-classes
```

## Backbone Cache

Pre-download the DINOv2 backbone so training/inference don't fetch via torch hub at runtime.

```bash
# With config
python scripts/download_model_checkpoints.py --model-config configs/DINOv2_model_configs.yml

# Without config (direct args)
python scripts/download_model_checkpoints.py --input dinov2_vitb14 --output ckpts/facebookresearch--dinov2-dinov2_vitb14
```

Skips if the cache path already exists.
