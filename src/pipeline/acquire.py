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


def _is_dataset_present(raw_dir: Path, expected_root: str) -> bool:
    root = raw_dir / expected_root
    # Minimal check: look for a couple of key files/folders
    return (root / "train").exists() and (root / "test").exists() and (root / "activity_labels.txt").exists()


def acquire_uci_har(cfg: AcquireConfig) -> Path:
    """
    Download and extract the UCI HAR dataset into cfg.raw_dir/cfg.expected_root.

    Idempotent:
      - If the dataset folder already exists with expected structure, do nothing.
      - If the zip is already present, skip download and just extract.

    Returns:
      Path to extracted dataset root folder (e.g., data/raw/uci_har/UCI HAR Dataset)
    """
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    if _is_dataset_present(cfg.raw_dir, cfg.expected_root):
        dataset_root = cfg.raw_dir / cfg.expected_root
        print(f"[acquire] Dataset already present at: {dataset_root}")
        return dataset_root

    # Download zip if needed
    if not cfg.zip_path.exists():
        print(f"[acquire] Downloading dataset from:\n  {cfg.url}\n→ {cfg.zip_path}")
        cfg.zip_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(cfg.url, cfg.zip_path)
        print("[acquire] Download complete.")
    else:
        print(f"[acquire] Zip already exists: {cfg.zip_path} (skipping download)")

    # Extract zip
    print(f"[acquire] Extracting to: {cfg.raw_dir}")
    with zipfile.ZipFile(cfg.zip_path, "r") as zf:
        zf.extractall(cfg.raw_dir)

    dataset_root = cfg.raw_dir / cfg.expected_root
    if not dataset_root.exists():
        # Some zips might unpack differently; help the user diagnose
        extracted = [p.name for p in cfg.raw_dir.iterdir()]
        raise FileNotFoundError(
            f"Expected extracted root folder '{cfg.expected_root}' not found in {cfg.raw_dir}.\n"
            f"Found: {extracted}"
        )

    # Optional: remove macOS __MACOSX directory if present
    macosx = cfg.raw_dir / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx, ignore_errors=True)

    print(f"[acquire] Ready: {dataset_root}")
    return dataset_root