# scripts/download_kaggle_dataset.py

from pathlib import Path
import argparse
import shutil
import zipfile

try:
    import kaggle
except ImportError:
    raise SystemExit(
        "kaggle package not installed.\n"
        "Install with: pip install kaggle"
    )


def download_dataset(dataset: str, output_dir: Path, tmp_dir: Path):
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {dataset}")

    kaggle.api.dataset_download_files(
        dataset,
        path=str(tmp_dir),
        quiet=False,
    )

    zip_files = sorted(
        tmp_dir.glob("*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not zip_files:
        raise FileNotFoundError(
            f"No zip file found in {tmp_dir}"
        )

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
    parser = argparse.ArgumentParser(
        description="Download and extract a Kaggle dataset."
    )

    parser.add_argument(
        "--dataset",
        default="fernando2rad/brain-tumor-mri-images-30-classes",
        help="Kaggle dataset slug, e.g. owner/dataset-name",
    )

    parser.add_argument(
        "--output",
        default="data/dataset/brain-tumor-mri-images-30-classes",
        help="Extraction directory",
    )

    parser.add_argument(
        "--tmp",
        default="tmp",
        help="Temporary download directory (default: tmp)",
    )

    args = parser.parse_args()

    download_dataset(
        dataset=args.dataset,
        output_dir=Path(args.output),
        tmp_dir=Path(args.tmp),
    )


if __name__ == "__main__":
    main()