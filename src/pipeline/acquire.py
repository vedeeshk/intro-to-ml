"""
Acquire the UCI HAR dataset by downloading and extracting the raw files if they are not already present.
This stage prepares the raw dataset directory used by the rest of the pipeline.
"""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve


@dataclass(frozen=True)
class AcquireConfig:
    url: str
    raw_dir: Path
    zip_path: Path
    expected_root: str = "UCI HAR Dataset"


def acquire_uci_har(cfg: AcquireConfig) -> Path:
    """
    Download and extract the UCI HAR dataset if it is not already available.
    Returns the path to the extracted dataset root.
    """
    dataset_root = cfg.raw_dir / cfg.expected_root
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already extracted
    if dataset_root.exists():
        print(f"[acquire] Dataset already present at: {dataset_root}")
        return dataset_root

    # Download if zip not present
    if not cfg.zip_path.exists():
        print(f"[acquire] Downloading dataset to {cfg.zip_path}")
        urlretrieve(cfg.url, cfg.zip_path)
        print("[acquire] Download complete.")

    # Extract zip
    print(f"[acquire] Extracting dataset to {cfg.raw_dir}")
    with zipfile.ZipFile(cfg.zip_path, "r") as zf:
        zf.extractall(cfg.raw_dir)

    # Remove macOS metadata folder if it exists
    macosx_dir = cfg.raw_dir / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)
        print("[acquire] Removed __MACOSX folder")

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Expected dataset folder '{cfg.expected_root}' was not found after extraction."
        )

    print(f"[acquire] Ready: {dataset_root}")
    return dataset_root