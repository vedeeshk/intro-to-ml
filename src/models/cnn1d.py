from __future__ import annotations

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    Simple 1D CNN for HAR.

    Input shape:
        (batch_size, channels, timesteps) = (B, 6, 128)

    Output shape:
        (batch_size, num_classes) = (B, 6)
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),   # 128 -> 64

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),   # 64 -> 32

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),   # 32 -> 16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 6, 128)
        returns: (B, 6)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x