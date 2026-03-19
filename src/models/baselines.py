from __future__ import annotations

"""
Train and evaluate classical baselines on the same raw UCI HAR windows used by the deep models.

The raw sensor windows are loaded from the processed dataset, normalised using training
statistics only, flattened into fixed-length feature vectors, and then passed to
classical classifiers such as Logistic Regression and Random Forest.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def load_processed_dataset(processed_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the processed raw HAR dataset.

    Returns:
        X: (N, 128, 6)
        y: (N,)
        subject: (N,)
    """
    data = np.load(processed_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    subject = data["subject"].astype(np.int64)
    return X, y, subject


def load_subject_split_indices(split_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(split_path, allow_pickle=True)
    return (
        data["train_idx"].astype(np.int64),
        data["val_idx"].astype(np.int64),
        data["test_idx"].astype(np.int64),
    )


def compute_norm_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean/std over data of shape (N, T, C).
    """
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def normalize_windows(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return (X - mean) / std


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """
    Flatten raw windows from (N, 128, 6) to (N, 768) for classical models.
    """
    return X.reshape(X.shape[0], -1)


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

    im = ax.imshow(display_cm, cmap="Blues")

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

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


def evaluate_baseline_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    output_dir: Path,
):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=ACTIVITY_NAMES)

    print(f"\n===== {model_name} =====")
    print(f"Test Accuracy: {acc:.4f}")
    print("\nConfusion Matrix")
    print(cm)
    print("\nClassification Report")
    print(report)

    safe_model_name = model_name.lower().replace(" ", "_").replace("-", "_")

    raw_plot_path = output_dir / f"{safe_model_name}_confusion_matrix.png"
    norm_plot_path = output_dir / f"{safe_model_name}_confusion_matrix_normalized.png"
    report_path = output_dir / f"{safe_model_name}_classification_report.txt"

    plot_confusion_matrix(
        cm=cm,
        class_names=ACTIVITY_NAMES,
        save_path=raw_plot_path,
        title=f"{model_name} Confusion Matrix",
        normalize=False,
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=ACTIVITY_NAMES,
        save_path=norm_plot_path,
        title=f"{model_name} Confusion Matrix (Normalized)",
        normalize=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report\n")
        f.write(report)

    print(f"\nSaved raw confusion matrix to: {raw_plot_path}")
    print(f"Saved normalized confusion matrix to: {norm_plot_path}")
    print(f"Saved classification report to: {report_path}")

    return acc, cm, report


def prepare_raw_baseline_data(
    processed_path: Path,
    split_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the raw processed dataset, apply the subject-wise split, normalize using
    train-only statistics, and flatten windows for classical ML models.

    Returns:
        X_train_flat, y_train, X_test_flat, y_test
    """
    X, y, _ = load_processed_dataset(processed_path)
    train_idx, _, test_idx = load_subject_split_indices(split_path)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    mean, std = compute_norm_stats(X_train)
    X_train = normalize_windows(X_train, mean, std)
    X_test = normalize_windows(X_test, mean, std)

    X_train_flat = flatten_windows(X_train)
    X_test_flat = flatten_windows(X_test)

    return X_train_flat, y_train, X_test_flat, y_test


def run_logistic_regression(processed_path: Path, split_path: Path, output_dir: Path):
    X_train, y_train, X_test, y_test = prepare_raw_baseline_data(processed_path, split_path)

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )

    return evaluate_baseline_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name="Logistic Regression (Raw)",
        output_dir=output_dir,
    )


def run_random_forest(processed_path: Path, split_path: Path, output_dir: Path):
    X_train, y_train, X_test, y_test = prepare_raw_baseline_data(processed_path, split_path)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    return evaluate_baseline_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name="Random Forest (Raw)",
        output_dir=output_dir,
    )