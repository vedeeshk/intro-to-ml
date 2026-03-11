from __future__ import annotations

import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM for HAR.

    Input:
        (B, C, T) = (batch, 6, 128)

    Steps:
        1. CNN extracts local temporal features
        2. transpose to sequence format
        3. LSTM models temporal evolution of extracted features
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
        x: (B, 6, 128)
        returns: (B, 6)
        """
        # CNN expects (B, C, T)
        x = self.cnn(x)  # (B, cnn_channels, 32)

        # LSTM expects (B, seq_len, features)
        x = x.transpose(1, 2)  # (B, 32, cnn_channels)

        output, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]  # (B, lstm_hidden_size)
        logits = self.classifier(last_hidden)
        return logits