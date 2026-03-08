from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class HarDataConfig:
    processed_path: Path
    split_path: Path
    normalize: bool = True


def _load_npz(path: Path) -> dict:
    return dict(np.load(path, allow_pickle=True))


def _compute_norm_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean/std over (N, T, C) using all timesteps.
    Returns:
      mean: (1, 1, C)
      std : (1, 1, C)
    """
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


class UciHarWindowDataset(Dataset):
    """
    Loads UCI HAR windowed inertial signals from .npz and applies subject-wise split indices.

    Saved processed format:
      X: (N, T, C) float32
      y: (N,) int64 labels 0..5
      subject: (N,) int64
    Split format:
      train_idx/val_idx/test_idx arrays of indices

    Returns:
      x: torch.FloatTensor of shape (C, T)  (channels-first for conv1d)
      y: torch.LongTensor scalar
    """

    def __init__(
        self,
        cfg: HarDataConfig,
        split: SplitName,
        norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.cfg = cfg
        self.split = split

        data = _load_npz(cfg.processed_path)
        splits = _load_npz(cfg.split_path)

        X = data["X"].astype(np.float32)  # (N,T,C)
        y = data["y"].astype(np.int64)    # (N,)

        idx_key = f"{split}_idx"
        if idx_key not in splits:
            raise KeyError(f"Split file missing key '{idx_key}'. Found: {list(splits.keys())}")

        idx = splits[idx_key].astype(np.int64)
        self.X = X[idx]
        self.y = y[idx]

        # Normalization: fit on train only, apply to all
        self.mean = None
        self.std = None
        if cfg.normalize:
            if norm_stats is None:
                if split != "train":
                    raise ValueError("norm_stats must be provided for val/test if normalize=True.")
                mean, std = _compute_norm_stats(self.X)
            else:
                mean, std = norm_stats
            self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
            self.X = (self.X - self.mean) / self.std

        # Convert to channels-first for CNNs: (N,T,C) -> (N,C,T)
        self.X = np.transpose(self.X, (0, 2, 1))

    def get_norm_stats(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.mean is None or self.std is None:
            return None
        return self.mean, self.std

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.X[i])  # (C,T)
        y = torch.tensor(self.y[i], dtype=torch.long)
        return x, y