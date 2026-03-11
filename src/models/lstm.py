from __future__ import annotations

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM for HAR.

    Expected input shape:
        (batch_size, channels, timesteps) = (B, 6, 128)

    Internally we transpose to:
        (batch_size, timesteps, channels) = (B, 128, 6)

    Output:
        (batch_size, num_classes) = (B, 6)
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 6, 128)
        returns: (B, 6)
        """
        # Convert from (B, C, T) to (B, T, C)
        x = x.transpose(1, 2)

        output, (hidden, cell) = self.lstm(x)

        # hidden shape: (num_layers, B, hidden_size)
        last_hidden = hidden[-1]  # (B, hidden_size)

        logits = self.classifier(last_hidden)
        return logits