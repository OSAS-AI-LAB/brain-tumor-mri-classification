import argparse
import sys
from pathlib import Path

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from brain_tumor_mri_classification.config import Config
from brain_tumor_mri_classification.dataset import load_dataset_info
from brain_tumor_mri_classification.inference import build_predictor

DEFAULT_DATA_CONFIG = "configs/data.yml"
DEFAULT_MODEL_CONFIG = "configs/DINOv2_model_configs.yml"
DEFAULT_CHECKPOINT = "ckpts/dinov2_brain_tumor/best_dinov2_brain_tumor.pth"


def main():
    parser = argparse.ArgumentParser(description="Gradio Brain Tumor MRI Classifier")
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG, help="Path to data YAML config")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG, help="Path to model YAML config")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to model checkpoint")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    config_paths = [p for p in [args.data_config, args.model_config] if Path(p).exists()]
    cfg = Config.from_yaml(*config_paths) if config_paths else Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, classes_list, _ = load_dataset_info(cfg.json_path, cfg.img_dir)
    predictor = build_predictor(args.checkpoint, classes_list, cfg, device)

    def predict(image):
        label, confidence = predictor.predict(image)
        top_k = predictor.predict_top_k(image, k=5)
        lines = [f"**Prediction**: {label}  (confidence: {confidence:.2%})", "", "**Top-5**:", ""]
        for i, (cls, prob) in enumerate(top_k, 1):
            lines.append(f"{i}. {cls} — {prob:.2%}")
        return "\n".join(lines)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload MRI Scan"),
        outputs=gr.Markdown(label="Result"),
        title="Brain Tumor MRI Classification",
        description=f"Upload a brain MRI image. The model predicts across {len(classes_list)} tumor classes using DINOv2.",
        allow_flagging="never",
    )

    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
