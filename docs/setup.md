# Setup

## Dependencies

```bash
pip install -r requirements/requirements-dev.txt
```

## Dataset

Authenticate Kaggle CLI first ([docs](https://www.kaggle.com/docs/api)).

```bash
# With config
python scripts/download_dataset.py --data-config data.yml

# Without config (direct args)
python scripts/download_dataset.py --input fernando2rad/brain-tumor-mri-images-30-classes --output data/dataset/brain-tumor-mri-images-30-classes
```

Config names are resolved from the `configs/` folder automatically, so `data.yml` works without the full path.

## Backbone Cache

Pre-download the DINOv2 backbone to avoid torch hub fetch at runtime.

```bash
# With config (default: large)
python scripts/download_model_checkpoints.py --model-config DINOv2_large_model_configs.yml

# With a different variant
python scripts/download_model_checkpoints.py --model-config DINOv2_small_model_configs.yml

# Without config
python scripts/download_model_checkpoints.py --input dinov2_vitl14 --output ckpts/facebookresearch--dinov2-dinov2_vitl14
```

Skips if the cache path already exists.
