# API Reference

## `brain_tumor_mri_classification.config`

### `Config`
Dataclass holding all configuration defaults.

```python
from brain_tumor_mri_classification.config import Config

cfg = Config(
    json_path="data/dataset/brain-tumor-mri-images-30-classes/DATA.json",
    img_dir="data/dataset/brain-tumor-mri-images-30-classes",
    img_size=518,
    batch_size=16,
    epochs=15,
    lr=1e-3,
)
```

---

## `brain_tumor_mri_classification.dataset`

### `BrainTumorDataset`
PyTorch `Dataset` for brain tumor MRI images.

```python
from brain_tumor_mri_classification.dataset import BrainTumorDataset

dataset = BrainTumorDataset(dataset_info, class_to_idx, transform=val_transform)
img, label = dataset[0]
```

### `load_dataset_info(json_path, img_dir)`
Loads image paths and class labels from `DATA.json`.

Returns `(dataset_info, classes_list, class_to_idx)`.

| Return | Type | Description |
|--------|------|-------------|
| `dataset_info` | `list[dict]` | `[{"img_path": str, "class": str}, ...]` |
| `classes_list` | `list[str]` | Sorted unique class names |
| `class_to_idx` | `dict[str, int]` | Class name → integer index |

### `build_transforms(cfg)`
Returns `(train_transform, val_transform)` composed `torchvision.transforms`.

### `create_dataloaders(cfg)`
Creates train/val DataLoaders with stratified split.

Returns `(train_loader, val_loader, classes_list, class_to_idx)`.

---

## `brain_tumor_mri_classification.model`

### `DINOv2Classifier`
DINOv2 backbone + MLP classification head.

```python
model = DINOv2Classifier(backbone, num_classes=30, hidden_dim=512, dropout=0.4)
logits = model(images)  # (batch, num_classes)
```

### `build_model(cfg, num_classes, device)`
Loads DINOv2 backbone from torch hub, freezes it, wraps in `DINOv2Classifier`.

### `load_checkpoint(model, checkpoint_path, device)`
Loads saved state dict into model.

---

## `brain_tumor_mri_classification.trainer`

### `train_one_epoch(model, loader, criterion, optimizer, device)`
Runs one training epoch. Returns `(avg_loss, accuracy)`.

### `validate(model, loader, criterion, device)`
Runs validation. Returns `(avg_loss, accuracy)`.

### `run_training(model, train_loader, val_loader, cfg, device)`
Full training loop with LR scheduling and best-model checkpointing. Returns `best_val_acc`.

---

## `brain_tumor_mri_classification.evaluate`

### `evaluate_model(model, loader, classes_list, device)`
Collects predictions and returns metrics dict:

```python
result = evaluate_model(model, val_loader, classes_list, device)
# result["report"]      → dict (sklearn classification_report output_dict=True)
# result["report_str"]  → str  (formatted text report)
# result["cm"]          → np.ndarray (confusion matrix)
# result["y_true"]      → list[int]
# result["y_pred"]      → list[int]
```

### `plot_confusion_matrix(cm, classes_list, save_path)`
Saves a matplotlib confusion matrix heatmap to `save_path`.

---

## `brain_tumor_mri_classification.inference`

### `Predictor`
High-level inference wrapper.

```python
predictor = Predictor(model, classes_list, cfg, device)
label, confidence = predictor.predict(image)         # top-1
top5 = predictor.predict_top_k(image, k=5)           # [(label, prob), ...]
```

### `build_predictor(checkpoint_path, classes_list, cfg, device)`
Loads a checkpoint and returns a ready-to-use `Predictor`.

### `build_inference_transform(cfg)`
Returns the inference-only transform (resize + normalize).
