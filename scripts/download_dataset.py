import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import kaggle
except ImportError:
    raise SystemExit("kaggle package not installed.\nInstall with: pip install kaggle")

from src.brain_tumor_mri_classification.config import Config


def download_dataset(dataset: str, output_dir: Path, tmp_dir: Path):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset: {dataset}")
    kaggle.api.dataset_download_files(dataset, path=str(tmp_dir), quiet=False)

    zip_files = sorted(tmp_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError(f"No zip file found in {tmp_dir}")

    zip_path = zip_files[0]
    if output_dir.exists():
        print(f"Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {output_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    print(f"Deleting zip: {zip_path}")
    zip_path.unlink()
    print("Done.")
    print(f"Dataset extracted to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and extract a Kaggle dataset.")
    parser.add_argument("--data-config", default="configs/data.yml", help="Path to data YAML config")
    parser.add_argument("--model-config", default="configs/DINOv2_model_configs.yml", help="Path to model YAML config")
    parser.add_argument("--dataset", default=None, help="Kaggle dataset slug (overrides config)")
    parser.add_argument("--output", default=None, help="Extraction directory (overrides config)")
    parser.add_argument("--tmp", default="tmp", help="Temporary download directory")
    args = parser.parse_args()

    config_paths = [p for p in [args.data_config, args.model_config] if Path(p).exists()]
    cfg = Config.from_yaml(*config_paths) if config_paths else Config()

    dataset = args.dataset or "/".join(cfg.dataset_url.split("/")[-2:])
    output_dir = Path(args.output or cfg.img_dir)
    download_dataset(dataset=dataset, output_dir=output_dir, tmp_dir=Path(args.tmp))


if __name__ == "__main__":
    main()
