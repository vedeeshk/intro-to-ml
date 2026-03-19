from __future__ import annotations

"""
Train and evaluate deep learning models for UCI HAR.

This module handles:
- loading train/validation/test datasets
- applying train-set normalization through the dataset pipeline
- training CNN, LSTM, and CNN-LSTM models
- running multi-seed experiments
- training a CNN-RF hybrid using a pretrained CNN as a feature extractor
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier

from src.utils.seed import set_seed
from src.pipeline.dataset import HarDataConfig, UciHarWindowDataset
from src.models.cnn1d import CNN1D
from src.models.lstm import LSTMModel
from src.models.cnn_lstm import CNNLSTMModel
from src.models.cnn_rf import extract_cnn_features


@dataclass(frozen=True)
class TrainConfig:
    processed_path: Path
    split_path: Path
    seed: int
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    model_save_path: Path = Path("artifacts/models/cnn1d_best.pt")


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_data_cfg(cfg: TrainConfig) -> HarDataConfig:
    return HarDataConfig(
        processed_path=cfg.processed_path,
        split_path=cfg.split_path,
        normalize=True,
    )


def _build_dataloaders(
    cfg: TrainConfig,
    batch_size: int | None = None,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = _build_data_cfg(cfg)

    train_ds = UciHarWindowDataset(data_cfg, "train")
    norm_stats = train_ds.get_norm_stats()

    val_ds = UciHarWindowDataset(data_cfg, "val", norm_stats=norm_stats)
    test_ds = UciHarWindowDataset(data_cfg, "test", norm_stats=norm_stats)

    bs = batch_size if batch_size is not None else cfg.batch_size

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=shuffle_train)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    return train_loader, val_loader, test_loader


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def _train_neural_model(
    model: nn.Module,
    cfg: TrainConfig,
    model_name: str,
) -> float:
    set_seed(cfg.seed)
    print(f"[train] Seed = {cfg.seed}")

    device = _get_device()
    print(f"[train] Using device: {device}")

    train_loader, val_loader, test_loader = _build_dataloaders(cfg)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_val_acc = 0.0
    cfg.model_save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == y).sum().item()
            running_samples += x.size(0)

        train_loss = running_loss / running_samples
        train_acc = running_correct / running_samples

        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

        print(
            f"[train] Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg.model_save_path)
            print(f"[train] Saved new best model to {cfg.model_save_path}")

    model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))
    test_loss, test_acc = _evaluate(model, test_loader, criterion, device)

    print(f"[train] Best val_acc={best_val_acc:.4f}")
    print(f"[train] Test loss={test_loss:.4f} test_acc={test_acc:.4f}")

    return test_acc


def train_cnn1d(cfg: TrainConfig) -> float:
    model = CNN1D(in_channels=6, num_classes=6)
    return _train_neural_model(model, cfg, model_name="cnn1d")


def train_lstm(cfg: TrainConfig) -> float:
    model = LSTMModel(
        input_size=6,
        hidden_size=64,
        num_layers=2,
        num_classes=6,
    )
    return _train_neural_model(model, cfg, model_name="lstm")


def train_cnn_lstm(cfg: TrainConfig) -> float:
    model = CNNLSTMModel(
        in_channels=6,
        cnn_channels=64,
        lstm_hidden_size=64,
        lstm_num_layers=1,
        num_classes=6,
    )
    return _train_neural_model(model, cfg, model_name="cnn_lstm")


def train_cnn_rf(cfg: TrainConfig) -> float:
    """
    Train a CNN-RF hybrid using a pretrained CNN checkpoint.

    The CNN is not retrained here. Instead:
    1. load the pretrained CNN checkpoint
    2. extract 128-dimensional features from train/val/test sets
    3. train a Random Forest on the train features
    4. report validation and test accuracy
    """
    set_seed(cfg.seed)
    print(f"[train] Seed = {cfg.seed}")

    device = _get_device()
    print(f"[train] Using device: {device}")

    train_loader, val_loader, test_loader = _build_dataloaders(
        cfg,
        batch_size=128,
        shuffle_train=False,
    )

    cnn = CNN1D(in_channels=6, num_classes=6).to(device)
    cnn.load_state_dict(torch.load(cfg.model_save_path, map_location=device))

    print("[cnn_rf] Extracting train features...")
    X_train, y_train = extract_cnn_features(cnn, train_loader, device)

    print("[cnn_rf] Extracting val features...")
    X_val, y_val = extract_cnn_features(cnn, val_loader, device)

    print("[cnn_rf] Extracting test features...")
    X_test, y_test = extract_cnn_features(cnn, test_loader, device)

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=cfg.seed,
        n_jobs=-1,
    )

    print(f"[cnn_rf] Training Random Forest on {X_train.shape[1]} CNN features...")
    rf_model.fit(X_train, y_train)

    val_acc = rf_model.score(X_val, y_val)
    test_acc = rf_model.score(X_test, y_test)

    print(f"[cnn_rf] Validation Accuracy: {val_acc:.4f}")
    print(f"[cnn_rf] Test Accuracy: {test_acc:.4f}")

    return test_acc


def _summarise_seed_results(seeds, results) -> None:
    mean_acc = sum(results) / len(results)
    std_acc = (sum((x - mean_acc) ** 2 for x in results) / len(results)) ** 0.5

    print("\n==============================")
    print("Final Results")
    print("==============================")

    for seed, result in zip(seeds, results):
        print(f"Seed {seed}: test_acc = {result:.4f}")

    print(f"\nMean test_acc = {mean_acc:.4f}")
    print(f"Std  test_acc = {std_acc:.4f}")


def run_multi_seed_experiment(seeds, base_cfg) -> None:
    results = []

    for seed in seeds:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")

        cfg = TrainConfig(
            processed_path=base_cfg.processed_path,
            split_path=base_cfg.split_path,
            seed=seed,
            batch_size=base_cfg.batch_size,
            epochs=base_cfg.epochs,
            learning_rate=base_cfg.learning_rate,
            weight_decay=base_cfg.weight_decay,
            model_save_path=base_cfg.model_save_path.parent / f"cnn1d_seed{seed}.pt",
        )

        test_acc = train_cnn1d(cfg)
        results.append(test_acc)

    _summarise_seed_results(seeds, results)


def run_multi_seed_experiment_lstm(seeds, base_cfg) -> None:
    results = []

    for seed in seeds:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")

        cfg = TrainConfig(
            processed_path=base_cfg.processed_path,
            split_path=base_cfg.split_path,
            seed=seed,
            batch_size=base_cfg.batch_size,
            epochs=base_cfg.epochs,
            learning_rate=base_cfg.learning_rate,
            weight_decay=base_cfg.weight_decay,
            model_save_path=base_cfg.model_save_path.parent / f"lstm_seed{seed}.pt",
        )

        test_acc = train_lstm(cfg)
        results.append(test_acc)

    _summarise_seed_results(seeds, results)


def run_multi_seed_experiment_cnn_lstm(seeds, base_cfg) -> None:
    results = []

    for seed in seeds:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")

        cfg = TrainConfig(
            processed_path=base_cfg.processed_path,
            split_path=base_cfg.split_path,
            seed=seed,
            batch_size=base_cfg.batch_size,
            epochs=base_cfg.epochs,
            learning_rate=base_cfg.learning_rate,
            weight_decay=base_cfg.weight_decay,
            model_save_path=base_cfg.model_save_path.parent / f"cnn_lstm_seed{seed}.pt",
        )

        test_acc = train_cnn_lstm(cfg)
        results.append(test_acc)

    _summarise_seed_results(seeds, results)


def run_multi_seed_experiment_cnn_rf(seeds, base_cfg) -> None:
    results = []

    for seed in seeds:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")

        cnn_checkpoint = base_cfg.model_save_path.parent / f"cnn1d_seed{seed}.pt"

        cfg = TrainConfig(
            processed_path=base_cfg.processed_path,
            split_path=base_cfg.split_path,
            seed=seed,
            batch_size=base_cfg.batch_size,
            epochs=base_cfg.epochs,
            learning_rate=base_cfg.learning_rate,
            weight_decay=base_cfg.weight_decay,
            model_save_path=cnn_checkpoint,
        )

        test_acc = train_cnn_rf(cfg)
        results.append(test_acc)

    _summarise_seed_results(seeds, results)


def smoke_test_lstm(processed_path: str, split_path: str, batch_size: int = 32) -> None:
    data_cfg = HarDataConfig(
        processed_path=Path(processed_path),
        split_path=Path(split_path),
        normalize=True,
    )

    train_ds = UciHarWindowDataset(data_cfg, "train")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    x_batch, y_batch = next(iter(train_loader))

    model = LSTMModel(input_size=6, hidden_size=64, num_layers=2, num_classes=6)
    logits = model(x_batch)

    print(f"[lstm] x_batch shape = {tuple(x_batch.shape)}")
    print(f"[lstm] y_batch shape = {tuple(y_batch.shape)}")
    print(f"[lstm] logits shape  = {tuple(logits.shape)}")

    assert x_batch.ndim == 3, "Expected input shape (B, C, T)"
    assert x_batch.shape[1] == 6, "Expected 6 channels"
    assert x_batch.shape[2] == 128, "Expected 128 timesteps"
    assert logits.shape[0] == x_batch.shape[0], "Output batch mismatch"
    assert logits.shape[1] == 6, "Expected 6 output classes"

    print("[lstm] LSTM smoke test passed.")


def smoke_test_cnn_lstm(processed_path: str, split_path: str, batch_size: int = 32) -> None:
    data_cfg = HarDataConfig(
        processed_path=Path(processed_path),
        split_path=Path(split_path),
        normalize=True,
    )

    train_ds = UciHarWindowDataset(data_cfg, "train")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    x_batch, y_batch = next(iter(train_loader))

    model = CNNLSTMModel(
        in_channels=6,
        cnn_channels=64,
        lstm_hidden_size=64,
        lstm_num_layers=1,
        num_classes=6,
    )

    logits = model(x_batch)

    print(f"[cnn_lstm] x_batch shape = {tuple(x_batch.shape)}")
    print(f"[cnn_lstm] y_batch shape = {tuple(y_batch.shape)}")
    print(f"[cnn_lstm] logits shape  = {tuple(logits.shape)}")

    assert x_batch.ndim == 3, "Expected input shape (B, C, T)"
    assert x_batch.shape[1] == 6, "Expected 6 channels"
    assert x_batch.shape[2] == 128, "Expected 128 timesteps"
    assert logits.shape[0] == x_batch.shape[0], "Output batch mismatch"
    assert logits.shape[1] == 6, "Expected 6 output classes"

    print("[cnn_lstm] CNN-LSTM smoke test passed.")