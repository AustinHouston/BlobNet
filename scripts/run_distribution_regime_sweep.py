from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from blobnet.loss_func import CombinedGaussianLoss
from blobnet.metrics import evaluate_model_localization
from blobnet.networks import build_unet
from blobnet.synthetic import metadata_collate
from scripts.train_unet import resolve_training_device


FAMILIES = ("random", "square", "hexagonal")

REGIMES: dict[str, dict[str, list[float]]] = {
    "compact_9_13_sigma_1p0_2p4": {
        "spacing_range": [9.0, 13.0],
        "sigma_range": [1.0, 2.4],
    },
    "small_7_11_sigma_0p8_2p0": {
        "spacing_range": [7.0, 11.0],
        "sigma_range": [0.8, 2.0],
    },
    "tiny_6_10_sigma_0p6_1p6": {
        "spacing_range": [6.0, 10.0],
        "sigma_range": [0.6, 1.6],
    },
    "baseline_11_15_sigma_1p4_3p0": {
        "spacing_range": [11.0, 15.0],
        "sigma_range": [1.4, 3.0],
    },
    "spacing_wide_8_18_sigma_1p4_3p0": {
        "spacing_range": [8.0, 18.0],
        "sigma_range": [1.4, 3.0],
    },
    "sigma_wide_11_15_sigma_1p0_4p0": {
        "spacing_range": [11.0, 15.0],
        "sigma_range": [1.0, 4.0],
    },
    "moderate_9_17_sigma_1p2_3p5": {
        "spacing_range": [9.0, 17.0],
        "sigma_range": [1.2, 3.5],
    },
    "wide_8_18_sigma_1p0_4p0": {
        "spacing_range": [8.0, 18.0],
        "sigma_range": [1.0, 4.0],
    },
    "very_wide_7_22_sigma_0p8_5p0": {
        "spacing_range": [7.0, 22.0],
        "sigma_range": [0.8, 5.0],
    },
}

MODEL_VARIANTS: dict[str, dict[str, Any]] = {
    "base": {
        "filters": [32, 64, 128, 256],
        "dropout": 0.2,
    },
    "narrow": {
        "filters": [16, 32, 64, 128],
        "dropout": 0.2,
    },
    "wide": {
        "filters": [48, 96, 192, 384],
        "dropout": 0.2,
    },
    "deep": {
        "filters": [32, 64, 128, 256, 512],
        "dropout": 0.2,
    },
    "low_dropout": {
        "filters": [32, 64, 128, 256],
        "dropout": 0.1,
    },
    "high_dropout": {
        "filters": [32, 64, 128, 256],
        "dropout": 0.35,
    },
}


class SavedAtomImageDatasetWithMetadata(Dataset):
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.paths = sorted(self.directory.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No NPZ samples found in {self.directory}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with np.load(self.paths[index], allow_pickle=False) as sample:
            image = torch.from_numpy(sample["image"].astype(np.float32)).unsqueeze(0)
            target = torch.from_numpy(sample["target"].astype(np.float32)).unsqueeze(0)
            metadata = {"coordinates": sample["coordinates"].astype(np.float32)}
        return image, target, metadata


def run_command(command: list[str]) -> None:
    print("\n" + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def build_dataset_config(
    family: str,
    regime_name: str,
    regime: dict[str, list[float]],
    output_root: Path,
    train_samples: int,
    val_samples: int,
    test_samples: int,
) -> Path:
    config = read_yaml(Path("configs") / "dataset_configs" / f"{family}.yaml")
    dataset_dir = output_root / "datasets" / regime_name / family
    config["dataset"]["name"] = f"{family}_{regime_name}"
    config["dataset"]["output_dir"] = str(dataset_dir)
    config["dataset"]["splits"] = {
        "train": int(train_samples),
        "val": int(val_samples),
        "test": int(test_samples),
    }
    parameters = config["parameters"]
    parameters["sigma_range"] = list(regime["sigma_range"])
    if family == "random":
        parameters["min_separation"] = float(np.mean(regime["spacing_range"]))
        parameters["min_separation_range"] = list(regime["spacing_range"])
    else:
        parameters["lattice_spacing_range"] = list(regime["spacing_range"])

    config_path = output_root / "configs" / regime_name / f"{family}_dataset.yaml"
    write_yaml(config_path, config)
    return config_path


def build_model_config(
    family: str,
    regime_name: str,
    dataset_dir: Path,
    output_root: Path,
    model_variant_name: str,
    model_variant: dict[str, Any],
) -> Path:
    config = read_yaml(Path("configs") / "model_configs" / f"{family}_unet.yaml")
    config["dataset"]["path"] = str(dataset_dir)
    model_settings = config["model"]
    model_settings["filters"] = list(model_variant["filters"])
    model_settings["dropout"] = float(model_variant["dropout"])
    config["training"]["output_dir"] = str(output_root / "models" / regime_name / model_variant_name / family)
    config_path = output_root / "configs" / regime_name / model_variant_name / f"{family}_unet.yaml"
    write_yaml(config_path, config)
    return config_path


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    config = read_yaml(config_path)
    model_settings = config["model"]
    model = build_unet(
        input_channels=int(model_settings.get("input_channels", 1)),
        num_classes=int(model_settings.get("output_channels", 1)),
        num_filters=model_settings["filters"],
        dropout=float(model_settings["dropout"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def evaluate_loss(
    model: torch.nn.Module,
    config_path: Path,
    loader: DataLoader,
    device: torch.device,
) -> float:
    config = read_yaml(config_path)
    criterion = CombinedGaussianLoss(**config["loss"])
    losses = []
    model.eval()
    with torch.inference_mode():
        for images, targets, _metadata in loader:
            images = images.to(device)
            targets = targets.to(device)
            losses.append(float(criterion(model(images), targets).item()))
    return float(np.mean(losses))


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep training data spacing/sigma regimes across BlobNet families.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/distribution_regime_sweep"))
    parser.add_argument("--regime", choices=sorted(REGIMES), action="append")
    parser.add_argument("--model-variant", choices=sorted(MODEL_VARIANTS), action="append")
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--dataset-workers", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--localization-samples", type=int, default=0)
    parser.add_argument("--max-peaks", type=int, default=256)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    selected_regimes = args.regime or list(REGIMES)
    selected_model_variants = args.model_variant or ["base"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_training_device(args.device)

    rows: list[dict[str, Any]] = []
    for regime_name in selected_regimes:
        regime = REGIMES[regime_name]
        dataset_configs: dict[str, Path] = {}
        dataset_dirs: dict[str, Path] = {}

        for family in FAMILIES:
            dataset_config = build_dataset_config(
                family,
                regime_name,
                regime,
                args.output_root,
                args.train_samples,
                args.val_samples,
                args.test_samples,
            )
            dataset_configs[family] = dataset_config
            dataset_dirs[family] = args.output_root / "datasets" / regime_name / family

            if not args.skip_generation:
                run_command([
                    sys.executable,
                    "scripts/generate_training_dataset.py",
                    "--config",
                    str(dataset_config),
                    "--num-workers",
                    str(args.dataset_workers),
                    "--overwrite",
                ])

        for model_variant_name in selected_model_variants:
            model_variant = MODEL_VARIANTS[model_variant_name]
            model_configs: dict[str, Path] = {}
            model_dirs: dict[str, Path] = {}
            for family in FAMILIES:
                model_config = build_model_config(
                    family,
                    regime_name,
                    dataset_dirs[family],
                    args.output_root,
                    model_variant_name,
                    model_variant,
                )
                model_configs[family] = model_config
                model_dirs[family] = args.output_root / "models" / regime_name / model_variant_name / family

                if not args.skip_training:
                    run_command([
                        sys.executable,
                        "scripts/train_unet.py",
                        "--config",
                        str(model_config),
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        str(args.batch_size),
                        "--device",
                        args.device,
                        "--num-workers",
                        str(args.num_workers),
                    ])

            for train_family in FAMILIES:
                model_dir = model_dirs[train_family]
                model = load_model(model_configs[train_family], model_dir / "unet_best.pth", device)
                train_metrics = json.loads((model_dir / "training_metrics.json").read_text())
                for test_family in FAMILIES:
                    test_dataset = SavedAtomImageDatasetWithMetadata(dataset_dirs[test_family] / "test")
                    loader = DataLoader(
                        test_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0,
                        collate_fn=metadata_collate,
                    )
                    loss = evaluate_loss(model, model_configs[train_family], loader, device)
                    localization: dict[str, Any]
                    if args.localization_samples > 0:
                        localization_dataset = Subset(
                            test_dataset,
                            range(min(int(args.localization_samples), len(test_dataset))),
                        )
                        localization_loader = DataLoader(
                            localization_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            persistent_workers=args.num_workers > 0,
                            collate_fn=metadata_collate,
                        )
                        localization = evaluate_model_localization(
                            model,
                            localization_loader,
                            device=device,
                            apply_sigmoid=True,
                            threshold_rel=0.35,
                            min_distance=3,
                            peak_window_size=5,
                            match_distance=3.0,
                            max_peaks=int(args.max_peaks),
                        )
                    else:
                        localization = {
                            "precision": np.nan,
                            "recall": np.nan,
                            "f1": np.nan,
                            "mean_error": np.nan,
                            "median_error": np.nan,
                            "rmse": np.nan,
                            "tp": 0,
                            "fp": 0,
                            "fn": 0,
                            "samples": 0,
                        }
                    rows.append({
                        "regime": regime_name,
                        "model_variant": model_variant_name,
                        "filters": json.dumps(model_variant["filters"]),
                        "dropout": model_variant["dropout"],
                        "spacing_min": regime["spacing_range"][0],
                        "spacing_max": regime["spacing_range"][1],
                        "sigma_min": regime["sigma_range"][0],
                        "sigma_max": regime["sigma_range"][1],
                        "train_family": train_family,
                        "test_family": test_family,
                        "train_best_validation_loss": train_metrics["best_validation_loss"],
                        "native_test_loss": train_metrics["test_loss"],
                        "cross_test_loss": loss,
                        "precision": localization["precision"],
                        "recall": localization["recall"],
                        "f1": localization["f1"],
                        "mean_error": localization["mean_error"],
                        "median_error": localization["median_error"],
                        "rmse": localization["rmse"],
                        "tp": localization["tp"],
                        "fp": localization["fp"],
                        "fn": localization["fn"],
                        "test_samples": localization["samples"],
                    })
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    csv_path = args.output_root / "regime_cross_eval.csv"
    json_path = args.output_root / "regime_cross_eval.json"
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved cross-evaluation metrics to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
