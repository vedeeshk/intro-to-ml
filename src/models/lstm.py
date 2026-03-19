from __future__ import annotations

"""
LSTM model for human activity recognition.

Processes each window as a sequence of length 128 with 6 features per timestep.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM for HAR.

    Input:
        (B, C, T) = (batch_size, 6, 128)

    Internal representation:
        (B, T, C) = (batch_size, 128, 6)

    Output:
        (B, num_classes) = (B, 6)
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
        Args:
            x: input tensor of shape (B, 6, 128)

        Returns:
            Logits of shape (B, 6)
        """
        # Convert from (B, C, T) to (B, T, C)
        x = x.transpose(1, 2)

        # LSTM forward pass
        _, (hidden, _) = self.lstm(x)

        # Take final layer hidden state
        last_hidden = hidden[-1]  # (B, hidden_size)

        logits = self.classifier(last_hidden)
        return logits