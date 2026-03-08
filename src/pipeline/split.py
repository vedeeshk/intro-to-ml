from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class SplitConfig:
    processed_path: Path
    split_path: Path
    seed: int = 42
    test_frac: float = 0.2
    val_frac: float = 0.2


def make_subject_holdout_split(cfg: SplitConfig) -> Path:
    """
    Subject-wise holdout split.
    Loads processed .npz containing: X, y, subject.
    Splits subjects into train/val/test, then returns indices for each subset.

    Saves:
      train_idx, val_idx, test_idx
      train_subjects, val_subjects, test_subjects
    """
    data = np.load(cfg.processed_path, allow_pickle=True)
    subject = data["subject"].astype(int)

    subjects = np.unique(subject)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = max(1, int(round(cfg.test_frac * n)))
    n_val = max(1, int(round(cfg.val_frac * n)))

    test_subjects = subjects[:n_test]
    val_subjects = subjects[n_test:n_test + n_val]
    train_subjects = subjects[n_test + n_val:]

    if len(train_subjects) < 1:
        raise ValueError("Split leaves no train subjects. Reduce val/test fractions.")

    train_mask = np.isin(subject, train_subjects)
    val_mask = np.isin(subject, val_subjects)
    test_mask = np.isin(subject, test_subjects)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    cfg.split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.split_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )

    print(f"[split] Saved: {cfg.split_path}")
    print(f"[split] Subjects: train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)}")
    print(f"[split] Samples : train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"[split] Test subjects: {test_subjects}")
    return cfg.split_path