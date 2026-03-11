from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from src.data.dataset import HarDataConfig, UciHarWindowDataset
from src.models.cnn1d import CNN1D
from src.models.lstm import LSTMModel
from src.models.cnn_lstm import CNNLSTMModel


ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def load_model(model_type: str, model_path: Path, device: torch.device):
    if model_type == "cnn":
        model = CNN1D(in_channels=6, num_classes=6)

    elif model_type == "lstm":
        model = LSTMModel(input_size=6, hidden_size=64, num_layers=2, num_classes=6)

    elif model_type == "cnn_lstm":
        model = CNNLSTMModel(
            in_channels=6,
            cnn_channels=64,
            lstm_hidden_size=64,
            num_classes=6,
        )

    else:
        raise ValueError("Unknown model type")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def evaluate_model(model_type: str, model_path: Path, processed_path: Path, split_path: Path):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    data_cfg = HarDataConfig(
        processed_path=processed_path,
        split_path=split_path,
        normalize=True,
    )

    train_ds = UciHarWindowDataset(data_cfg, "train")
    norm_stats = train_ds.get_norm_stats()

    test_ds = UciHarWindowDataset(data_cfg, "test", norm_stats=norm_stats)

    loader = DataLoader(test_ds, batch_size=64)

    model = load_model(model_type, model_path, device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=ACTIVITY_NAMES))