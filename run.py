import argparse
from pathlib import Path

import yaml

from src.pipeline.acquire import AcquireConfig, acquire_uci_har
from src.pipeline.preprocess import PreprocessConfig, preprocess_uci_har
from src.pipeline.split import SplitConfig, make_subject_holdout_split
from src.pipeline.features import smoke_test_dataset


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, help="Pipeline stage (acquire, preprocess, split, features, train, eval)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
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
        pcfg = cfg.get("preprocess", {})  # <-- safe default
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
        processed_path = scfg.get("processed_path", "data/processed/uci_har_windows.npz")
        split_path = scfg.get("split_path", "data/processed/splits_subject_holdout.npz")

        smoke_test_dataset(
            processed_path=processed_path,
            split_path=split_path,
        )

    else:
        print(f"Stage '{args.stage}' not implemented yet.")

    


if __name__ == "__main__":
    main()