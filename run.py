import argparse
from pathlib import Path

import yaml

from src.pipeline.acquire import AcquireConfig, acquire_uci_har
from src.pipeline.preprocess import PreprocessConfig, preprocess_uci_har
from src.pipeline.split import SplitConfig, make_subject_holdout_split
from src.pipeline.dataset import smoke_test_dataset
from src.models.baselines import run_logistic_regression, run_random_forest
from src.train.trainer import (
    TrainConfig,
    run_multi_seed_experiment,
    run_multi_seed_experiment_lstm,
    run_multi_seed_experiment_cnn_lstm,
    run_multi_seed_experiment_cnn_rf,
    smoke_test_lstm,
    smoke_test_cnn_lstm,
)
from src.eval.evaluate import evaluate_model


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        type=str,
        help=(
            "Pipeline stage "
            "(acquire, preprocess, split, features, train, train_lstm, "
            "train_cnn_lstm, cnn_rf, lstm_test, cnn_lstm_test, "
            "evaluate, baseline_lr, baseline_rf)"
        ),
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, help="cnn | lstm | cnn_lstm")
    parser.add_argument("--model_path", type=str, help="path to model checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.stage == "acquire":
        dcfg = cfg["dataset"]
        acfg = AcquireConfig(
            url=dcfg["url"],
            raw_dir=Path(dcfg["raw_dir"]),
            zip_path=Path(dcfg["zip_path"]),
            expected_root=dcfg.get("expected_root", "UCI HAR Dataset"),
        )
        acquire_uci_har(acfg)

    elif args.stage == "preprocess":
        dcfg = cfg["dataset"]
        pcfg = cfg.get("preprocess", {})
        processed_path = pcfg.get("processed_path", "data/processed/uci_har_windows.npz")

        dataset_root = Path(dcfg["raw_dir"]) / dcfg.get("expected_root", "UCI HAR Dataset")

        preprocess_uci_har(
            PreprocessConfig(
                dataset_root=dataset_root,
                processed_path=Path(processed_path),
            )
        )

    elif args.stage == "split":
        scfg = cfg.get("split", {})
        make_subject_holdout_split(
            SplitConfig(
                processed_path=Path(scfg.get("processed_path", "data/processed/uci_har_windows.npz")),
                split_path=Path(scfg.get("split_path", "data/processed/splits_subject_holdout.npz")),
                seed=int(scfg.get("seed", 42)),
                test_frac=float(scfg.get("test_frac", 0.2)),
                val_frac=float(scfg.get("val_frac", 0.2)),
            )
        )

    elif args.stage == "features":
        scfg = cfg.get("split", {})
        smoke_test_dataset(
            processed_path=scfg.get("processed_path", "data/processed/uci_har_windows.npz"),
            split_path=scfg.get("split_path", "data/processed/splits_subject_holdout.npz"),
        )

    elif args.stage == "train":
        tcfg = cfg.get("train", {})
        seeds = tcfg.get("seeds", [42])

        base_cfg = TrainConfig(
            processed_path=Path(tcfg.get("processed_path")),
            split_path=Path(tcfg.get("split_path")),
            seed=seeds[0],
            batch_size=int(tcfg.get("batch_size", 64)),
            epochs=int(tcfg.get("epochs", 10)),
            learning_rate=float(tcfg.get("learning_rate", 5e-4)),
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
            model_save_path=Path(tcfg.get("model_save_path")),
        )

        run_multi_seed_experiment(seeds, base_cfg)

    elif args.stage == "train_lstm":
        tcfg = cfg.get("train", {})
        seeds = tcfg.get("seeds", [42])

        base_cfg = TrainConfig(
            processed_path=Path(tcfg.get("processed_path")),
            split_path=Path(tcfg.get("split_path")),
            seed=seeds[0],
            batch_size=int(tcfg.get("batch_size", 64)),
            epochs=int(tcfg.get("epochs", 10)),
            learning_rate=float(tcfg.get("learning_rate", 5e-4)),
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
            model_save_path=Path("artifacts/models/lstm_best.pt"),
        )

        run_multi_seed_experiment_lstm(seeds, base_cfg)

    elif args.stage == "train_cnn_lstm":
        tcfg = cfg.get("train", {})
        seeds = tcfg.get("seeds", [42])

        base_cfg = TrainConfig(
            processed_path=Path(tcfg.get("processed_path")),
            split_path=Path(tcfg.get("split_path")),
            seed=seeds[0],
            batch_size=int(tcfg.get("batch_size", 64)),
            epochs=int(tcfg.get("epochs", 10)),
            learning_rate=float(tcfg.get("learning_rate", 5e-4)),
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
            model_save_path=Path("artifacts/models/cnn_lstm_best.pt"),
        )

        run_multi_seed_experiment_cnn_lstm(seeds, base_cfg)

    elif args.stage == "cnn_rf":
        tcfg = cfg.get("train", {})
        seeds = tcfg.get("seeds", [42])

        base_cfg = TrainConfig(
            processed_path=Path(tcfg.get("processed_path")),
            split_path=Path(tcfg.get("split_path")),
            seed=seeds[0],
            batch_size=int(tcfg.get("batch_size", 64)),
            epochs=int(tcfg.get("epochs", 10)),
            learning_rate=float(tcfg.get("learning_rate", 5e-4)),
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
            model_save_path=Path(tcfg.get("model_save_path")),
        )

        run_multi_seed_experiment_cnn_rf(seeds, base_cfg)

    elif args.stage == "lstm_test":
        tcfg = cfg.get("train", {})
        smoke_test_lstm(
            processed_path=tcfg.get("processed_path", "data/processed/uci_har_windows.npz"),
            split_path=tcfg.get("split_path", "data/processed/splits_subject_holdout.npz"),
            batch_size=32,
        )

    elif args.stage == "cnn_lstm_test":
        tcfg = cfg.get("train", {})
        smoke_test_cnn_lstm(
            processed_path=tcfg.get("processed_path", "data/processed/uci_har_windows.npz"),
            split_path=tcfg.get("split_path", "data/processed/splits_subject_holdout.npz"),
            batch_size=32,
        )

    elif args.stage == "baseline_lr":
        tcfg = cfg.get("train", {})
        run_logistic_regression(
            processed_path=Path(tcfg.get("processed_path", "data/processed/uci_har_windows.npz")),
            split_path=Path(tcfg.get("split_path", "data/processed/splits_subject_holdout.npz")),
            output_dir=Path("artifacts/plots"),
        )

    elif args.stage == "baseline_rf":
        tcfg = cfg.get("train", {})
        run_random_forest(
            processed_path=Path(tcfg.get("processed_path", "data/processed/uci_har_windows.npz")),
            split_path=Path(tcfg.get("split_path", "data/processed/splits_subject_holdout.npz")),
            output_dir=Path("artifacts/plots"),
        )

    elif args.stage == "evaluate":
        tcfg = cfg.get("train", {})
        evaluate_model(
            model_type=args.model,
            model_path=Path(args.model_path),
            processed_path=Path(tcfg.get("processed_path")),
            split_path=Path(tcfg.get("split_path")),
            output_dir=Path("artifacts/plots"),
        )

    else:
        print(f"Stage '{args.stage}' not implemented yet.")


if __name__ == "__main__":
    main()