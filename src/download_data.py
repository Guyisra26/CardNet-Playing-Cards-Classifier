from pathlib import Path
import os
import shutil
import kagglehub

# Kaggle dataset identifier
DATASET_NAME = "gpiosenka/cards-image-datasetclassification"

# Resolve paths relative to repository root (ex1/)
ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT_DIR / "data" / "cards"


def download_and_reset_data() -> None:
    print("--- Starting Data Download Process ---")

    # 1) Delete old dataset folder (start clean)
    if TARGET_DIR.exists():
        print(f"Deleting old directory: {TARGET_DIR}...")
        shutil.rmtree(TARGET_DIR)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # 2) Download dataset to local cache
    print("Downloading dataset from Kaggle (this may take a moment)...")
    cache_path = kagglehub.dataset_download(DATASET_NAME)
    cache_path = Path(cache_path)
    print(f"Download finished to cache: {cache_path}")

    # 3) Copy from cache into project data folder
    print(f"Copying files to: {TARGET_DIR}...")
    shutil.copytree(cache_path, TARGET_DIR, dirs_exist_ok=True)

    print("\n✅ SUCCESS! Dataset is ready.")
    print(f"Files are located at: {TARGET_DIR}")


if __name__ == "__main__":
    download_and_reset_data()
