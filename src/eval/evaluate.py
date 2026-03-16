from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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
            lstm_num_layers=1,
            num_classes=6,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str,
    normalize: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    display_cm = cm.astype(float)

    if normalize:
        row_sums = display_cm.sum(axis=1, keepdims=True)
        display_cm = np.divide(display_cm, row_sums, where=row_sums != 0)

    # light blue heatmap
    im = ax.imshow(display_cm, cmap="Blues")

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # choose text color based on cell darkness
    threshold = display_cm.max() / 2.0 if display_cm.max() > 0 else 0.0

    for i in range(display_cm.shape[0]):
        for j in range(display_cm.shape[1]):
            value = display_cm[i, j]
            text = f"{value:.2f}" if normalize else f"{cm[i, j]}"
            color = "white" if value > threshold else "black"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=11,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(
    model_type: str,
    model_path: Path,
    processed_path: Path,
    split_path: Path,
    output_dir: Path,
) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    data_cfg = HarDataConfig(
        processed_path=processed_path,
        split_path=split_path,
        normalize=True,
    )

    # Fit normalization on train split only
    train_ds = UciHarWindowDataset(data_cfg, "train")
    norm_stats = train_ds.get_norm_stats()

    test_ds = UciHarWindowDataset(data_cfg, "test", norm_stats=norm_stats)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

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

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=ACTIVITY_NAMES)

    print(f"\nAccuracy: {acc:.4f}")
    print("\nConfusion Matrix")
    print(cm)
    print("\nClassification Report")
    print(report)

    safe_model_name = model_type.lower().replace("-", "_")

    raw_plot_path = output_dir / f"{safe_model_name}_confusion_matrix.png"
    norm_plot_path = output_dir / f"{safe_model_name}_confusion_matrix_normalized.png"
    report_path = output_dir / f"{safe_model_name}_classification_report.txt"

    plot_confusion_matrix(
        cm=cm,
        class_names=ACTIVITY_NAMES,
        save_path=raw_plot_path,
        title=f"{model_type.upper()} Confusion Matrix",
        normalize=False,
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=ACTIVITY_NAMES,
        save_path=norm_plot_path,
        title=f"{model_type.upper()} Confusion Matrix (Normalized)",
        normalize=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"Model: {model_type}\n")
        f.write(f"Checkpoint: {model_path}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report\n")
        f.write(report)

    print(f"\nSaved raw confusion matrix to: {raw_plot_path}")
    print(f"Saved normalized confusion matrix to: {norm_plot_path}")
    print(f"Saved classification report to: {report_path}")