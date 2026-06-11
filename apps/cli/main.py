import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from brain_tumor_mri_classification.config import Config
from brain_tumor_mri_classification.dataset import create_dataloaders, load_dataset_info
from brain_tumor_mri_classification.model import build_model, load_checkpoint
from brain_tumor_mri_classification.trainer import run_training
from brain_tumor_mri_classification.evaluate import evaluate_model, plot_confusion_matrix
from brain_tumor_mri_classification.inference import build_predictor
from brain_tumor_mri_classification.utils import export_to_onnx


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _load_cfg(data_config: str | None, model_config: str | None, overrides: dict | None = None) -> Config:
    paths = []
    if data_config:
        paths.append(data_config)
    if model_config:
        paths.append(model_config)
    return Config.from_yaml(*paths, overrides=overrides)


def cmd_train(args):
    overrides = {
        "json_path": args.json,
        "img_dir": args.img_dir,
        "checkpoint_dir": args.ckpt_dir,
        "best_model_path": str(Path(args.ckpt_dir) / "best_dinov2_brain_tumor.pth") if args.ckpt_dir else None,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    cfg = _load_cfg(args.data_config, args.model_config, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_loader, val_loader, classes_list, _ = create_dataloaders(cfg)
    logger.info("Classes: %d | Train: %d | Val: %d", len(classes_list), len(train_loader.dataset), len(val_loader.dataset))

    model = build_model(cfg, len(classes_list), device)
    best_acc = run_training(model, train_loader, val_loader, cfg, device)
    logger.info("Training complete. Best Val Acc: %.4f", best_acc)


def cmd_evaluate(args):
    overrides = {"json_path": args.json, "img_dir": args.img_dir}
    cfg = _load_cfg(args.data_config, None, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    _, val_loader, classes_list, _ = create_dataloaders(cfg)
    model = build_model(cfg, len(classes_list), device)
    load_checkpoint(model, args.checkpoint, device)

    result = evaluate_model(model, val_loader, classes_list, device)

    if args.cm:
        plot_confusion_matrix(result["cm"], classes_list, args.cm)


def cmd_inference(args):
    overrides = {"json_path": args.json or None, "img_dir": args.img_dir or None, "img_size": args.img_size}
    cfg = _load_cfg(args.data_config, args.model_config, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    classes_list = args.classes.split(",") if args.classes else []
    if not classes_list:
        _, classes_list, _ = load_dataset_info(cfg.json_path, cfg.img_dir)

    predictor = build_predictor(args.checkpoint, classes_list, cfg, device)

    from PIL import Image
    image = Image.open(args.image).convert("RGB")
    label, confidence = predictor.predict(image)
    print(f"Prediction: {label} (confidence: {confidence:.4f})")


def cmd_export_onnx(args):
    overrides = {"json_path": args.json or None, "img_dir": args.img_dir or None}
    cfg = _load_cfg(args.data_config, args.model_config, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    classes_list = args.classes.split(",") if args.classes else []
    if not classes_list:
        _, classes_list, _ = load_dataset_info(cfg.json_path, cfg.img_dir)

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        classes_list=classes_list,
        cfg=cfg,
        output_path=args.output,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor MRI Classification CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data-config", default="configs/data.yml")
    p_train.add_argument("--model-config", default="configs/DINOv2_large_model_configs.yml")
    p_train.add_argument("--json")
    p_train.add_argument("--img-dir")
    p_train.add_argument("--ckpt-dir")
    p_train.add_argument("--batch-size", type=int)
    p_train.add_argument("--epochs", type=int)
    p_train.add_argument("--lr", type=float)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--data-config", default="configs/data.yml")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--json")
    p_eval.add_argument("--img-dir")
    p_eval.add_argument("--cm", default=None, help="Path to save confusion matrix PNG")
    p_eval.set_defaults(func=cmd_evaluate)

    p_infer = sub.add_parser("inference")
    p_infer.add_argument("--data-config", default="configs/data.yml")
    p_infer.add_argument("--model-config", default="configs/DINOv2_large_model_configs.yml")
    p_infer.add_argument("--checkpoint", required=True)
    p_infer.add_argument("--image", required=True)
    p_infer.add_argument("--json")
    p_infer.add_argument("--img-dir")
    p_infer.add_argument("--classes", default=None, help="Comma-separated class names")
    p_infer.add_argument("--img-size", type=int)
    p_infer.set_defaults(func=cmd_inference)

    p_export = sub.add_parser("export-onnx")
    p_export.add_argument("--data-config", default="configs/data.yml")
    p_export.add_argument("--model-config", default="configs/DINOv2_large_model_configs.yml")
    p_export.add_argument("--checkpoint", required=True)
    p_export.add_argument("--output", default="model.onnx", help="Output ONNX path")
    p_export.add_argument("--json")
    p_export.add_argument("--img-dir")
    p_export.add_argument("--classes", default=None, help="Comma-separated class names")
    p_export.set_defaults(func=cmd_export_onnx)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
