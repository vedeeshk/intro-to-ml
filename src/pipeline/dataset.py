from __future__ import annotations

"""
Load the processed UCI HAR dataset, apply subject-wise train/validation/test splits,
normalize sensor channels using training statistics, and return PyTorch tensors for model training.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class HarDataConfig:
    processed_path: Path
    split_path: Path
    normalize: bool = True


def load_npz(path: Path) -> dict:
    return dict(np.load(path, allow_pickle=True))


def compute_norm_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and standard deviation over a dataset of shape (N, T, C).

    Returns:
        mean: (1, 1, C)
        std:  (1, 1, C)
    """
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


class UciHarWindowDataset(Dataset):
    """
    PyTorch dataset for processed UCI HAR windows.

    Processed data format:
        X: (N, T, C)
        y: (N,)
        subject: (N,)

    Split file format:
        train_idx, val_idx, test_idx
    """

    def __init__(
        self,
        cfg: HarDataConfig,
        split: SplitName,
        norm_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.cfg = cfg
        self.split = split

        data = load_npz(cfg.processed_path)
        splits = load_npz(cfg.split_path)

        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)

        idx_key = f"{split}_idx"
        if idx_key not in splits:
            raise KeyError(f"Split file missing key '{idx_key}'. Found: {list(splits.keys())}")

        indices = splits[idx_key].astype(np.int64)

        self.X = X[indices]
        self.y = y[indices]

        self.mean = None
        self.std = None

        if cfg.normalize:
            if norm_stats is None:
                if split != "train":
                    raise ValueError("norm_stats must be provided for val/test when normalize=True.")
                mean, std = compute_norm_stats(self.X)
            else:
                mean, std = norm_stats

            self.mean = mean.astype(np.float32)
            self.std = std.astype(np.float32)
            self.X = (self.X - self.mean) / self.std

        # Convert from (N, T, C) to (N, C, T) for Conv1D models
        self.X = np.transpose(self.X, (0, 2, 1))

    def get_norm_stats(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.mean is None or self.std is None:
            return None
        return self.mean, self.std

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[index])  # (C, T)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y


def smoke_test_dataset(processed_path: str, split_path: str) -> None:
    cfg = HarDataConfig(
        processed_path=Path(processed_path),
        split_path=Path(split_path),
        normalize=True,
    )

    train_ds = UciHarWindowDataset(cfg, "train")
    norm_stats = train_ds.get_norm_stats()

    val_ds = UciHarWindowDataset(cfg, "val", norm_stats=norm_stats)
    test_ds = UciHarWindowDataset(cfg, "test", norm_stats=norm_stats)

    x_train, y_train = train_ds[0]
    x_val, y_val = val_ds[0]
    x_test, y_test = test_ds[0]

    print(f"[dataset] train size = {len(train_ds)}")
    print(f"[dataset] val size   = {len(val_ds)}")
    print(f"[dataset] test size  = {len(test_ds)}")
    print(f"[dataset] train sample shape = {tuple(x_train.shape)} (C, T), label = {int(y_train)}")
    print(f"[dataset] val sample shape   = {tuple(x_val.shape)} (C, T), label = {int(y_val)}")
    print(f"[dataset] test sample shape  = {tuple(x_test.shape)} (C, T), label = {int(y_test)}")

    assert x_train.ndim == 2, "Expected sample shape (C, T)"
    assert x_train.shape[0] == 6, "Expected 6 channels"
    assert x_train.shape[1] == 128, "Expected 128 timesteps"
    assert 0 <= int(y_train) <= 5, "Expected label in range 0..5"

    print("[dataset] Dataset smoke test passed.")