import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import kagglehub
except ImportError:
    raise SystemExit(
        "kagglehub package not installed.\n"
        "Install with: pip install kagglehub"
    )

from src.brain_tumor_mri_classification.config import Config


def download_dataset(dataset: str, output_dir: Path, tmp_dir: Path):
    """
    Downloads and extracts a public Kaggle dataset without requiring an API token.
    The dataset is first downloaded and extracted to tmp_dir, then copied to output_dir.
    """
    print(f"Downloading public dataset: {dataset} (No API token required)")
    
    # Ensure tmp_dir exists
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Use output_dir parameter instead of path parameter
    cache_path = Path(kagglehub.dataset_download(dataset, output_dir=str(tmp_dir)))
    
    if output_dir.exists():
        print(f"Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying extracted files from {cache_path} to {output_dir}")
    # Copy all contents from the tmp_dir to the desired output directory
    for item in cache_path.iterdir():
        if item.is_dir():
            shutil.copytree(item, output_dir / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, output_dir / item.name)
            
    print("Done.")
    print(f"Dataset successfully prepared at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract a public Kaggle dataset without an API token."
    )
    parser.add_argument("--data-config", default=None, help="Path to data YAML config")
    parser.add_argument("--input", default=None, help="Kaggle dataset slug (e.g., owner/name)")
    parser.add_argument("--output", default=None, help="Extraction directory")
    parser.add_argument("--tmp", default="tmp", help="Temporary directory for initial download and extraction")
    args = parser.parse_args()

    if args.data_config and Path(args.data_config).exists():
        cfg = Config.from_yaml(args.data_config)
        dataset = args.input or "/".join(cfg.dataset_url.split("/")[-2:])
        output_dir = Path(args.output or cfg.img_dir)
    else:
        dataset = args.input or "fernando2rad/brain-tumor-mri-images-30-classes"
        output_dir = Path(args.output or "data/dataset/brain-tumor-mri-images-30-classes")

    download_dataset(dataset=dataset, output_dir=output_dir, tmp_dir=Path(args.tmp))


if __name__ == "__main__":
    main()