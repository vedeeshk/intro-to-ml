from __future__ import annotations

"""
Hybrid CNN-LSTM model for human activity recognition.

The CNN extracts local temporal features from raw IMU windows, and the LSTM models
their evolution over time before classification.
"""

import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM for HAR.

    Input:
        (B, C, T) = (batch_size, 6, 128)

    Internal flow:
        1. CNN extracts local temporal features
        2. transpose to sequence format
        3. LSTM models temporal evolution of CNN features
        4. classifier predicts activity class

    Output:
        (B, num_classes) = (B, 6)
    """

    def __init__(
        self,
        in_channels: int = 6,
        cnn_channels: int = 64,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        num_classes: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),   # 128 -> 64

            nn.Conv1d(in_channels=32, out_channels=cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.MaxPool1d(kernel_size=2),   # 64 -> 32
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, 6, 128)

        Returns:
            Logits of shape (B, 6)
        """
        # CNN feature extraction
        x = self.cnn(x)  # (B, cnn_channels, 32)

        # Convert to sequence format for LSTM
        x = x.transpose(1, 2)  # (B, 32, cnn_channels)

        # LSTM forward pass
        _, (hidden, _) = self.lstm(x)

        # Use final hidden state for classification
        last_hidden = hidden[-1]  # (B, lstm_hidden_size)
        logits = self.classifier(last_hidden)
        return logits