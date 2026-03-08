from pathlib import Path

from src.data.dataset import HarDataConfig, UciHarWindowDataset


def smoke_test_dataset(processed_path: str, split_path: str) -> None:
    cfg = HarDataConfig(
        processed_path=Path(processed_path),
        split_path=Path(split_path),
        normalize=True,
    )

    # Fit normalization on train only
    train_ds = UciHarWindowDataset(cfg, "train")
    norm_stats = train_ds.get_norm_stats()

    # Reuse train stats for val/test
    val_ds = UciHarWindowDataset(cfg, "val", norm_stats=norm_stats)
    test_ds = UciHarWindowDataset(cfg, "test", norm_stats=norm_stats)

    x_train, y_train = train_ds[0]
    x_val, y_val = val_ds[0]
    x_test, y_test = test_ds[0]

    print(f"[features] train size = {len(train_ds)}")
    print(f"[features] val size   = {len(val_ds)}")
    print(f"[features] test size  = {len(test_ds)}")
    print(f"[features] train sample shape = {tuple(x_train.shape)} (C, T), label = {int(y_train)}")
    print(f"[features] val sample shape   = {tuple(x_val.shape)} (C, T), label = {int(y_val)}")
    print(f"[features] test sample shape  = {tuple(x_test.shape)} (C, T), label = {int(y_test)}")

    # Sanity checks
    assert x_train.ndim == 2, "Expected sample shape (C, T)"
    assert x_train.shape[0] == 6, "Expected 6 channels"
    assert x_train.shape[1] == 128, "Expected 128 timesteps"
    assert 0 <= int(y_train) <= 5, "Expected label in range 0..5"

    print("[features] Dataset smoke test passed.")