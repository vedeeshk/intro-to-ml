from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    dataset_root: Path                 # .../data/raw/uci_har/UCI HAR Dataset
    processed_path: Path               # .../data/processed/uci_har_windows.npz


# UCI HAR inertial signal filenames (6 channels)
SIGNAL_FILES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]


def _load_split(dataset_root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads one split (train or test) as:
      X: (N, T, C) where T=128, C=6
      y: (N,) labels 0..5
      subject: (N,) subject IDs
    """
    split_dir = dataset_root / split
    inertial_dir = split_dir / "Inertial Signals"

    # labels are 1..6 in files; we'll convert to 0..5
    y = np.loadtxt(split_dir / f"y_{split}.txt", dtype=np.int64) - 1
    subject = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=np.int64)

    # load each channel: each file is (N, 128)
    channels = []
    for name in SIGNAL_FILES:
        f = inertial_dir / f"{name}_{split}.txt"
        arr = np.loadtxt(f, dtype=np.float32)  # (N, 128)
        channels.append(arr)

    # stack into (N, 128, 6)
    X = np.stack(channels, axis=-1)  # (N, T, C)
    return X, y, subject


def preprocess_uci_har(cfg: PreprocessConfig) -> Path:
    cfg.processed_path.parent.mkdir(parents=True, exist_ok=True)

    # Load both splits then merge
    X_train, y_train, s_train = _load_split(cfg.dataset_root, "train")
    X_test, y_test, s_test = _load_split(cfg.dataset_root, "test")

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    subject = np.concatenate([s_train, s_test], axis=0)

    np.savez_compressed(
        cfg.processed_path,
        X=X,
        y=y,
        subject=subject,
        channels=np.array(SIGNAL_FILES),
    )

    print(f"[preprocess] Saved: {cfg.processed_path}")
    print(f"[preprocess] X shape: {X.shape} (N, T, C)")
    print(f"[preprocess] y shape: {y.shape}, subjects: {len(np.unique(subject))}")
    return cfg.processed_path