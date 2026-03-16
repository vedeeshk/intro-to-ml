from __future__ import annotations

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

    # Skip everything if dataset is already extracted
    if dataset_root.exists():
        print(f"[acquire] Dataset already present at: {dataset_root}")
        return dataset_root

    # Download zip if needed
    if not cfg.zip_path.exists():
        print(f"[acquire] Downloading dataset to {cfg.zip_path}")
        urlretrieve(cfg.url, cfg.zip_path)
        print("[acquire] Download complete.")

    # Extract zip
    print(f"[acquire] Extracting dataset to {cfg.raw_dir}")
    with zipfile.ZipFile(cfg.zip_path, "r") as zf:
        zf.extractall(cfg.raw_dir)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Expected dataset folder '{cfg.expected_root}' was not found after extraction."
        )

    print(f"[acquire] Ready: {dataset_root}")
    return dataset_root