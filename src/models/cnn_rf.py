from __future__ import annotations

"""
Utility functions for the CNN-RF hybrid model.

This module provides feature extraction from a trained CNN by taking the
128-dimensional activations from the penultimate layer.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader


def extract_cnn_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 128-dimensional CNN features for all samples in a loader.

    The features are taken from the penultimate layer of the CNN
    (after Flatten -> Linear -> ReLU, before Dropout and final classification).

    Returns:
        features: (N, 128)
        labels: (N,)
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            # CNN feature extractor
            out = model.features(x)

            # Penultimate representation: Flatten -> Linear -> ReLU
            out = model.classifier[0](out)
            out = model.classifier[1](out)
            out = model.classifier[2](out)

            all_features.append(out.cpu().numpy())
            all_labels.append(y.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels