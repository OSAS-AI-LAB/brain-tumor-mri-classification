# API Reference

## `brain_tumor_mri_classification.config`

### `Config`
Dataclass with defaults. Load from YAML via `from_yaml(*paths, overrides=None)`.

Paths are resolved from `configs/` folder automatically.

```python
cfg = Config()
cfg = Config.from_yaml("data.yml", "DINOv2_large_model_configs.yml")
cfg = Config.from_yaml(overrides={"epochs": 20})
```

| Field | Type | Default |
|-------|------|---------|
| `dataset_url` | `str` | Kaggle URL |
| `json_path` | `str` | `data/.../DATA.json` |
| `img_dir` | `str` | `data/...` |
| `checkpoint_dir` | `str` | `ckpts/dinov2_brain_tumor` |
| `best_model_path` | `str` | `ckpts/.../best_dinov2_vitl14_brain_tumor.pth` |
| `backbone_name` | `str` | `dinov2_vitl14` |
| `backbone_cache_path` | `str` | `ckpts/...-dinov2_vitl14` |
| `hidden_dim` | `int` | 768 |
| `dropout` | `float` | 0.4 |
| `img_size` | `int` | 518 |
| `batch_size` | `int` | 8 |
| `epochs` | `int` | 15 |
| `lr` | `float` | 1e-3 |
| `weight_decay` | `float` | 1e-4 |
| `val_split` | `float` | 0.2 |
| `seed` | `int` | 42 |
| `num_workers` | `int` | 2 |
| `pin_memory` | `bool` | True |
| `mean` | `tuple` | (0.485, 0.456, 0.406) |
| `std` | `tuple` | (0.229, 0.224, 0.225) |

---

## `brain_tumor_mri_classification.dataset`

### `BrainTumorDataset`
```python
ds = BrainTumorDataset(dataset_info, class_to_idx, transform=val_transform)
img, label = ds[0]
```

### `load_dataset_info(json_path, img_dir)` → `(dataset_info, classes_list, class_to_idx)`

### `build_transforms(cfg)` → `(train_transform, val_transform)`

### `create_dataloaders(cfg)` → `(train_loader, val_loader, classes_list, class_to_idx)`

---

## `brain_tumor_mri_classification.model`

### `DINOv2Classifier`
```python
model = DINOv2Classifier(backbone, num_classes=30, hidden_dim=768, dropout=0.4)
logits = model(images)
```

### `_load_backbone(cfg, device)`
Loads backbone from `cfg.backbone_cache_path`, or downloads via hub and caches.

### `build_model(cfg, num_classes, device)` → `DINOv2Classifier`
Loads backbone, freezes it, wraps in classifier head.

### `load_checkpoint(model, checkpoint_path, device)`

---

## `brain_tumor_mri_classification.trainer`

### `train_one_epoch(model, loader, criterion, optimizer, device)` → `(loss, acc)`

### `validate(model, loader, criterion, device)` → `(loss, acc)`

### `run_training(model, train_loader, val_loader, cfg, device)` → `best_val_acc`

---

## `brain_tumor_mri_classification.evaluate`

### `evaluate_model(model, loader, classes_list, device)` → `dict`

Returns `report`, `report_str`, `cm`, `y_true`, `y_pred`.

### `plot_confusion_matrix(cm, classes_list, save_path)`

---

## `brain_tumor_mri_classification.inference`

### `Predictor`
```python
p = Predictor(model, classes_list, cfg, device)
label, conf = p.predict(image)
top5 = p.predict_top_k(image, k=5)
```

### `build_predictor(checkpoint_path, classes_list, cfg, device)` → `Predictor`

### `build_inference_transform(cfg)` → `Compose`

---

## `brain_tumor_mri_classification.utils`

### `set_seed(seed)`
Seeds Python, NumPy, PyTorch, CUDA.

### `count_parameters(model)` → `{total, trainable}`

### `export_to_onnx(checkpoint_path, classes_list, cfg, output_path, device, opset_version)`
```python
export_to_onnx(checkpoint_path="ckpts/.../best.pth", classes_list=classes_list, cfg=cfg, output_path="model.onnx")
```
