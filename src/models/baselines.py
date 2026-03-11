from __future__ import annotations

from pathlib import Path
import numpy as np
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


def _load_engineered_split(dataset_root: Path, split: str):
    """
    Load engineered UCI HAR features for one split.

    Returns:
        X: (N, 561)
        y: (N,) labels in 0..5
        subject: (N,)
    """
    split_dir = dataset_root / split

    X = np.loadtxt(split_dir / f"X_{split}.txt", dtype=np.float32)
    y = np.loadtxt(split_dir / f"y_{split}.txt", dtype=np.int64) - 1
    subject = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=np.int64)

    return X, y, subject


def load_engineered_dataset(dataset_root: Path):
    """
    Merge train and test engineered features into one dataset.
    """
    X_train, y_train, s_train = _load_engineered_split(dataset_root, "train")
    X_test, y_test, s_test = _load_engineered_split(dataset_root, "test")

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    subject = np.concatenate([s_train, s_test], axis=0)

    return X, y, subject


def load_subject_split_indices(split_path: Path):
    data = np.load(split_path, allow_pickle=True)
    return (
        data["train_idx"].astype(np.int64),
        data["val_idx"].astype(np.int64),
        data["test_idx"].astype(np.int64),
    )


def evaluate_baseline_model(model, X_train, y_train, X_test, y_test, model_name: str):
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

    return acc, cm, report


def run_logistic_regression(dataset_root: Path, split_path: Path):
    X, y, _ = load_engineered_dataset(dataset_root)
    train_idx, val_idx, test_idx = load_subject_split_indices(split_path)

    # Classical baselines: train on train only, evaluate on test
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    model = LogisticRegression(
        max_iter=2000,
        multi_class="auto",
        solver="lbfgs",
        n_jobs=None,
    )

    return evaluate_baseline_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name="Logistic Regression",
    )


def run_random_forest(dataset_root: Path, split_path: Path):
    X, y, _ = load_engineered_dataset(dataset_root)
    train_idx, val_idx, test_idx = load_subject_split_indices(split_path)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

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
        model_name="Random Forest",
    )