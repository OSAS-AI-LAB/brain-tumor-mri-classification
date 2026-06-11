import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    classes_list: list[str],
    device: torch.device,
) -> dict:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    report = classification_report(
        all_labels, all_preds, target_names=classes_list, output_dict=True
    )
    report_str = classification_report(
        all_labels, all_preds, target_names=classes_list
    )
    cm = confusion_matrix(all_labels, all_preds)

    logger.info("\n" + report_str)

    return {"report": report, "report_str": report_str, "cm": cm, "y_true": all_labels, "y_pred": all_preds}


def plot_confusion_matrix(
    cm: np.ndarray,
    classes_list: list[str],
    save_path: str = "confusion_matrix_dinov2.png",
):
    fig, ax = plt.subplots(figsize=(20, 16))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes_list)),
        yticks=np.arange(len(classes_list)),
        xticklabels=classes_list,
        yticklabels=classes_list,
        title="Confusion Matrix - DINOv2",
        ylabel="Real Class",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)
