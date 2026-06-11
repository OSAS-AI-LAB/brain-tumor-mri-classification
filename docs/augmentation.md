# Augmentation

## Training

1. Resize to `img_size` × `img_size` (default 518)
2. Random horizontal flip (p=0.5)
3. ColorJitter(brightness=0.15, contrast=0.15) with p=0.4
4. RandomAffine(translate=0.05, scale=(0.95, 1.05))
5. ToTensor
6. Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

## Validation / Inference

1. Resize to `img_size` × `img_size`
2. ToTensor
3. Normalize (same ImageNet stats)
