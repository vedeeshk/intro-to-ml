"""
Load the raw UCI HAR sensor files, combine the six inertial signals into multivariate time-series windows,
and save a processed dataset used for training and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    dataset_root: Path
    processed_path: Path


SIGNAL_FILES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]


def load_uci_har_split(dataset_root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one UCI HAR split (train or test).

    Returns:
        X: (N, 128, 6)
        y: (N,) labels in range 0..5
        subject: (N,) subject IDs
    """
    split_dir = dataset_root / split
    inertial_dir = split_dir / "Inertial Signals"

    y = np.loadtxt(split_dir / f"y_{split}.txt", dtype=np.int64) - 1
    subject = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=np.int64)

    channels = []
    for signal_name in SIGNAL_FILES:
        signal_path = inertial_dir / f"{signal_name}_{split}.txt"
        signal = np.loadtxt(signal_path, dtype=np.float32)
        channels.append(signal)

    X = np.stack(channels, axis=-1)
    return X, y, subject


def preprocess_uci_har(cfg: PreprocessConfig) -> Path:
    """
    Load the raw UCI HAR train/test splits, merge them into one dataset,
    and save the processed windows as a compressed NumPy file.
    """
    if not cfg.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {cfg.dataset_root}")

    cfg.processed_path.parent.mkdir(parents=True, exist_ok=True)

    X_train, y_train, subject_train = load_uci_har_split(cfg.dataset_root, "train")
    X_test, y_test, subject_test = load_uci_har_split(cfg.dataset_root, "test")

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    subject = np.concatenate([subject_train, subject_test], axis=0)

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