from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GombNet.loss_func import CombinedGaussianLoss
from GombNet.metrics import evaluate_heatmap_localization, evaluate_model_localization
from GombNet.networks import build_unet
from GombNet.synthetic import (
    ImageFormationConfig,
    PeriodicLatticeConfig,
    RandomMicroscopeImageConfig,
    SyntheticMicroscopeDataset,
    TightSpacingRandomMicroscopeImageConfig,
    metadata_collate,
)
from GombNet.utils import resolve_torch_device, train_model
from GombNet.visualization import (
    build_prediction_gallery,
    collect_matched_offsets,
    plot_generalization_summary,
    plot_loss_curves,
    plot_offset_cloud,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASE_LABELS = {
    "random_min_distance": "Random, min distance",
    "tight_random_spacing": "Random, tight spacing",
    "square_lattice": "Square lattice",
    "hexagonal_lattice": "Hexagonal lattice",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one U-Net per synthetic dataset type and compare every trained "
            "model against every held-out dataset type."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--min-atoms", type=int, default=550)
    parser.add_argument("--max-atoms", type=int, default=800)
    parser.add_argument("--min-separation", type=float, default=14.5)
    parser.add_argument("--min-separation-range-min", type=float, default=12.5)
    parser.add_argument("--min-separation-range-max", type=float, default=16.7)
    parser.add_argument("--tight-spacing-min", type=float, default=14.0)
    parser.add_argument("--tight-spacing-max", type=float, default=15.2)
    parser.add_argument("--tight-spacing-jitter-min", type=float, default=0.03)
    parser.add_argument("--tight-spacing-jitter-max", type=float, default=0.08)
    parser.add_argument("--tight-min-spacing-fraction", type=float, default=0.86)
    parser.add_argument("--sigma-min", type=float, default=2.8)
    parser.add_argument("--sigma-max", type=float, default=4.3)
    parser.add_argument("--background-min", type=float, default=0.0)
    parser.add_argument("--background-max", type=float, default=0.30)
    parser.add_argument("--inhom-background-min", type=float, default=0.0)
    parser.add_argument("--inhom-background-max", type=float, default=0.0)
    parser.add_argument("--low-freq-noise-min", type=float, default=0.08)
    parser.add_argument("--low-freq-noise-max", type=float, default=0.30)
    parser.add_argument("--read-noise-min", type=float, default=0.03)
    parser.add_argument("--read-noise-max", type=float, default=0.14)
    parser.add_argument("--total-counts-min", type=float, default=50.0)
    parser.add_argument("--total-counts-max", type=float, default=25000.0)
    parser.add_argument("--counts-per-pixel-min", type=float, default=None)
    parser.add_argument("--counts-per-pixel-max", type=float, default=None)
    parser.add_argument("--blur-sigma-min", type=float, default=0.3)
    parser.add_argument("--blur-sigma-max", type=float, default=1.1)
    parser.add_argument("--edge-padding", type=int, default=24)
    parser.add_argument("--lattice-spacing-min", type=float, default=12.5)
    parser.add_argument("--lattice-spacing-max", type=float, default=16.7)
    parser.add_argument("--lattice-jitter-min", type=float, default=0.0)
    parser.add_argument("--lattice-jitter-max", type=float, default=0.25)
    parser.add_argument("--lattice-vacancy-min", type=float, default=0.0)
    parser.add_argument("--lattice-vacancy-max", type=float, default=0.03)
    parser.add_argument("--lattice-rotation-min", type=float, default=0.0)
    parser.add_argument("--lattice-rotation-max", type=float, default=180.0)
    parser.add_argument("--lattice-min-atoms", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-filters", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--threshold-rel", type=float, default=0.35)
    parser.add_argument("--match-distance", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--offset-plot-range", type=float, default=0.6)
    parser.add_argument("--offset-bins", type=int, default=181)
    parser.add_argument("--gallery-sample-index", type=int, default=0)
    parser.add_argument("--output-sample-indices", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def count_ranges_from_args(args: argparse.Namespace) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    using_counts_per_pixel = (
        args.counts_per_pixel_min is not None
        or args.counts_per_pixel_max is not None
    )
    if using_counts_per_pixel and (
        args.counts_per_pixel_min is None
        or args.counts_per_pixel_max is None
    ):
        raise ValueError(
            "--counts-per-pixel-min and --counts-per-pixel-max must be set together."
        )
    if using_counts_per_pixel:
        return None, (args.counts_per_pixel_min, args.counts_per_pixel_max)
    return (args.total_counts_min, args.total_counts_max), None


def common_image_settings(args: argparse.Namespace) -> Dict[str, object]:
    total_counts_range, counts_per_pixel_range = count_ranges_from_args(args)
    return {
        "image_shape": (args.height, args.width),
        "sigma_range": (args.sigma_min, args.sigma_max),
        "background_range": (args.background_min, args.background_max),
        "inhomogeneous_background_range": (
            args.inhom_background_min,
            args.inhom_background_max,
        ),
        "low_frequency_noise_range": (
            args.low_freq_noise_min,
            args.low_freq_noise_max,
        ),
        "read_noise_std_range": (args.read_noise_min, args.read_noise_max),
        "total_counts_range": total_counts_range,
        "counts_per_pixel_range": counts_per_pixel_range,
        "blur_sigma_range": (args.blur_sigma_min, args.blur_sigma_max),
        "edge_padding": args.edge_padding,
    }


def build_study_configs(args: argparse.Namespace) -> Dict[str, ImageFormationConfig]:
    common = common_image_settings(args)
    return {
        "random_min_distance": RandomMicroscopeImageConfig(
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
            min_separation=args.min_separation,
            min_separation_range=(
                args.min_separation_range_min,
                args.min_separation_range_max,
            ),
            **common,
        ),
        "tight_random_spacing": TightSpacingRandomMicroscopeImageConfig(
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
            nearest_neighbor_spacing_range=(
                args.tight_spacing_min,
                args.tight_spacing_max,
            ),
            spacing_jitter_fraction_range=(
                args.tight_spacing_jitter_min,
                args.tight_spacing_jitter_max,
            ),
            min_spacing_fraction=args.tight_min_spacing_fraction,
            **common,
        ),
        "square_lattice": PeriodicLatticeConfig(
            lattice_type="cubic",
            lattice_spacing_range=(args.lattice_spacing_min, args.lattice_spacing_max),
            rotation_range=(args.lattice_rotation_min, args.lattice_rotation_max),
            jitter_std_range=(args.lattice_jitter_min, args.lattice_jitter_max),
            vacancy_fraction_range=(args.lattice_vacancy_min, args.lattice_vacancy_max),
            min_atoms=args.lattice_min_atoms,
            **common,
        ),
        "hexagonal_lattice": PeriodicLatticeConfig(
            lattice_type="hexagonal",
            lattice_spacing_range=(args.lattice_spacing_min, args.lattice_spacing_max),
            rotation_range=(args.lattice_rotation_min, args.lattice_rotation_max),
            jitter_std_range=(args.lattice_jitter_min, args.lattice_jitter_max),
            vacancy_fraction_range=(args.lattice_vacancy_min, args.lattice_vacancy_max),
            min_atoms=args.lattice_min_atoms,
            **common,
        ),
    }


def make_loader(
    config: ImageFormationConfig,
    samples: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    return_metadata: bool,
) -> DataLoader:
    dataset = SyntheticMicroscopeDataset(
        samples,
        config,
        seed=seed,
        return_metadata=return_metadata,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=metadata_collate if return_metadata else None,
    )


def save_study_config(
    args: argparse.Namespace,
    configs: Dict[str, ImageFormationConfig],
) -> Path:
    output_path = args.output_dir / "study_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "datasets": {
            name: {
                "class": type(config).__name__,
                "config": asdict(config),
            }
            for name, config in configs.items()
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def save_metrics(rows: List[Dict[str, float]], output_dir: Path) -> None:
    fieldnames = [
        "model",
        "test_case",
        "precision",
        "recall",
        "f1",
        "mean_error",
        "median_error",
        "rmse",
        "tp",
        "fp",
        "fn",
        "samples",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "dataset_study_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    (output_dir / "dataset_study_metrics.json").write_text(json.dumps(rows, indent=2))


def save_ranking(rows: List[Dict[str, float]], output_dir: Path) -> Path:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)

    ranking = []
    for model_name, model_rows in grouped.items():
        f1_values = np.asarray([float(row["f1"]) for row in model_rows], dtype=float)
        rmse_values = np.asarray([float(row["rmse"]) for row in model_rows], dtype=float)
        precision_values = np.asarray([float(row["precision"]) for row in model_rows], dtype=float)
        recall_values = np.asarray([float(row["recall"]) for row in model_rows], dtype=float)
        ranking.append(
            {
                "model": model_name,
                "label": CASE_LABELS.get(model_name, model_name),
                "mean_f1": float(np.nanmean(f1_values)),
                "mean_rmse": float(np.nanmean(rmse_values)),
                "mean_precision": float(np.nanmean(precision_values)),
                "mean_recall": float(np.nanmean(recall_values)),
            }
        )
    ranking.sort(key=lambda item: (-item["mean_f1"], item["mean_rmse"]))

    json_path = output_dir / "dataset_study_ranking.json"
    json_path.write_text(json.dumps(ranking, indent=2))
    with (output_dir / "dataset_study_ranking.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking[0].keys()))
        writer.writeheader()
        writer.writerows(ranking)
    return json_path


def plot_study_heatmaps(
    rows: List[Dict[str, float]],
    case_order: Sequence[str],
    output_path: Path,
) -> Path:
    lookup = {(row["model"], row["test_case"]): row for row in rows}
    f1_grid = np.full((len(case_order), len(case_order)), np.nan, dtype=float)
    rmse_grid = np.full_like(f1_grid, np.nan)
    for train_index, train_case in enumerate(case_order):
        for test_index, test_case in enumerate(case_order):
            row = lookup.get((train_case, test_case))
            if row is not None:
                f1_grid[train_index, test_index] = float(row["f1"])
                rmse_grid[train_index, test_index] = float(row["rmse"])

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0), constrained_layout=True)
    labels = [CASE_LABELS.get(case_name, case_name) for case_name in case_order]
    for ax, grid, title, cmap, vmin, vmax in [
        (axes[0], f1_grid, "F1 score", "YlGnBu", 0.0, 1.0),
        (
            axes[1],
            rmse_grid,
            "RMSE (pixels)",
            "magma_r",
            0.0,
            max(float(np.nanmax(rmse_grid)), 1e-6),
        ),
    ]:
        image = ax.imshow(np.nan_to_num(grid, nan=0.0), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(case_order)), labels, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(case_order)), labels)
        ax.set_xlabel("Held-out test dataset")
        ax.set_ylabel("Training dataset")
        for row_index in range(grid.shape[0]):
            for col_index in range(grid.shape[1]):
                value = grid[row_index, col_index]
                text = "NA" if np.isnan(value) else f"{value:.3f}"
                ax.text(col_index, row_index, text, ha="center", va="center", color="black", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Dataset Study: Train/Test Generalization Matrix", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metric_bar_charts(
    rows: List[Dict[str, float]],
    ranking_path: Path,
    case_order: Sequence[str],
    output_dir: Path,
) -> List[Path]:
    lookup = {(row["model"], row["test_case"]): row for row in rows}
    labels = [CASE_LABELS.get(case_name, case_name) for case_name in case_order]
    x = np.arange(len(case_order))
    palette = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    width = 0.8 / max(len(case_order), 1)
    output_paths: List[Path] = []

    fig, axes = plt.subplots(2, 1, figsize=(14.5, 10.0), constrained_layout=True)
    for test_index, test_case in enumerate(case_order):
        offset = (test_index - (len(case_order) - 1) / 2.0) * width
        f1_values = [
            float(lookup[(train_case, test_case)]["f1"])
            if (train_case, test_case) in lookup
            else np.nan
            for train_case in case_order
        ]
        rmse_values = [
            float(lookup[(train_case, test_case)]["rmse"])
            if (train_case, test_case) in lookup
            else np.nan
            for train_case in case_order
        ]
        axes[0].bar(
            x + offset,
            np.nan_to_num(f1_values, nan=0.0),
            width=width,
            label=CASE_LABELS.get(test_case, test_case),
            color=palette[test_index % len(palette)],
        )
        axes[1].bar(
            x + offset,
            np.nan_to_num(rmse_values, nan=0.0),
            width=width,
            label=CASE_LABELS.get(test_case, test_case),
            color=palette[test_index % len(palette)],
        )

    axes[0].set_title("F1 by Training Dataset and Held-out Test Dataset")
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_title("RMSE by Training Dataset and Held-out Test Dataset")
    axes[1].set_ylabel("RMSE (pixels)")
    axes[1].set_ylim(
        0.0,
        max(
            [float(row["rmse"]) for row in rows if not np.isnan(float(row["rmse"]))]
            or [1.0]
        )
        * 1.15,
    )
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_xlabel("Training dataset")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Test dataset", ncols=2)

    path = output_dir / "dataset_study_grouped_bar_charts.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    ranking = json.loads(ranking_path.read_text())
    ranking_labels = [item["label"] for item in ranking]
    ranking_x = np.arange(len(ranking))
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0), constrained_layout=True)
    for ax, key, title, ylabel, color in [
        (axes[0, 0], "mean_f1", "Mean F1 Across All Test Datasets", "Mean F1", "#4e79a7"),
        (axes[0, 1], "mean_rmse", "Mean RMSE Across All Test Datasets", "Mean RMSE (pixels)", "#e15759"),
        (axes[1, 0], "mean_precision", "Mean Precision", "Mean precision", "#59a14f"),
        (axes[1, 1], "mean_recall", "Mean Recall", "Mean recall", "#f28e2b"),
    ]:
        values = [float(item[key]) for item in ranking]
        ax.bar(ranking_x, values, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(ranking_x, ranking_labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if key != "mean_rmse":
            ax.set_ylim(0.0, 1.0)
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    path = output_dir / "dataset_study_mean_metric_bar_charts.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    diagonal_rows = [lookup[(case_name, case_name)] for case_name in case_order]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), constrained_layout=True)
    axes[0].bar(x, [float(row["f1"]) for row in diagonal_rows], color="#4e79a7")
    axes[0].set_title("In-distribution F1")
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].bar(x, [float(row["rmse"]) for row in diagonal_rows], color="#e15759")
    axes[1].set_title("In-distribution RMSE")
    axes[1].set_ylabel("RMSE (pixels)")
    axes[1].set_ylim(
        0.0,
        max(float(row["rmse"]) for row in diagonal_rows if not np.isnan(float(row["rmse"]))) * 1.15,
    )
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_xlabel("Training and test dataset")
        ax.grid(axis="y", alpha=0.25)

    path = output_dir / "dataset_study_in_distribution_bar_charts.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)
    return output_paths


def save_prediction_comparison_gallery(
    models: Dict[str, torch.nn.Module],
    test_loaders: Dict[str, DataLoader],
    case_order: Sequence[str],
    sample_index: int,
    device: torch.device,
    threshold_rel: float,
    output_path: Path,
) -> Path:
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for train_case, model in models.items():
            model.eval()
            for test_case in case_order:
                dataset = test_loaders[test_case].dataset
                image, target, metadata = dataset[int(sample_index)]
                prediction = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                result = evaluate_heatmap_localization(
                    prediction,
                    metadata["coordinates"],
                    threshold_rel=threshold_rel,
                )
                rmse_text = "n/a" if np.isnan(result["rmse"]) else f"{result['rmse']:.3f} px"
                rows.append(
                    {
                        "label": (
                            f"train={CASE_LABELS.get(train_case, train_case)} | "
                            f"test={CASE_LABELS.get(test_case, test_case)}"
                        ),
                        "metrics_text": (
                            f"sample={sample_index}  atoms={len(metadata['coordinates'])}  "
                            f"pred={len(result['predicted_coordinates'])}  "
                            f"F1={result['f1']:.3f}  RMSE={rmse_text}"
                        ),
                        "image": image[0].numpy(),
                        "target": target[0].numpy(),
                        "prediction": prediction,
                        "true_coords": metadata["coordinates"],
                        "predicted_coords": result["predicted_coordinates"],
                    }
                )
    return build_prediction_gallery(rows, output_path)


def save_model_output_series(
    models: Dict[str, torch.nn.Module],
    test_loaders: Dict[str, DataLoader],
    case_order: Sequence[str],
    sample_indices: Sequence[int],
    device: torch.device,
    threshold_rel: float,
    output_dir: Path,
) -> List[Path]:
    series_dir = output_dir / "model_output_series"
    series_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    with torch.no_grad():
        for test_case in case_order:
            dataset = test_loaders[test_case].dataset
            for sample_index in sample_indices:
                if sample_index < 0 or sample_index >= len(dataset):
                    continue
                image, target, metadata = dataset[int(sample_index)]
                true_coords = metadata["coordinates"]
                columns = 2 + 2 * len(models)
                fig, axes = plt.subplots(
                    1,
                    columns,
                    figsize=(3.1 * columns, 3.8),
                    constrained_layout=True,
                )
                axes = np.atleast_1d(axes)
                image_np = image[0].numpy()
                target_np = target[0].numpy()

                axes[0].imshow(image_np, cmap="gray")
                axes[0].set_title(
                    f"{CASE_LABELS.get(test_case, test_case)}\nInput sample {sample_index}"
                )
                axes[1].imshow(target_np, cmap="magma")
                axes[1].set_title(f"Target\natoms={len(true_coords)}")

                for model_index, (train_case, model) in enumerate(models.items()):
                    prediction = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                    result = evaluate_heatmap_localization(
                        prediction,
                        true_coords,
                        threshold_rel=threshold_rel,
                    )
                    pred_axis = axes[2 + 2 * model_index]
                    overlay_axis = axes[3 + 2 * model_index]
                    pred_axis.imshow(prediction, cmap="magma", vmin=0.0, vmax=max(float(prediction.max()), 1e-6))
                    pred_axis.set_title(
                        f"{CASE_LABELS.get(train_case, train_case)}\nheatmap"
                    )
                    overlay_axis.imshow(image_np, cmap="gray")
                    if len(true_coords):
                        overlay_axis.scatter(
                            true_coords[:, 1],
                            true_coords[:, 0],
                            s=18,
                            facecolors="none",
                            edgecolors="lime",
                            linewidths=0.8,
                        )
                    predicted_coords = result["predicted_coordinates"]
                    if len(predicted_coords):
                        overlay_axis.scatter(
                            predicted_coords[:, 1],
                            predicted_coords[:, 0],
                            s=16,
                            c="cyan",
                            marker="x",
                            linewidths=0.8,
                        )
                    rmse_text = "n/a" if np.isnan(result["rmse"]) else f"{result['rmse']:.3f}"
                    overlay_axis.set_title(
                        f"Overlay\nF1={result['f1']:.3f}, RMSE={rmse_text}"
                    )

                for axis in axes:
                    axis.axis("off")

                fig.suptitle(
                    "Model outputs on the same held-out image "
                    f"({CASE_LABELS.get(test_case, test_case)}, sample {sample_index})",
                    fontsize=13,
                )
                output_path = series_dir / f"{test_case}_sample_{sample_index:03d}.png"
                fig.savefig(output_path, dpi=180, bbox_inches="tight")
                plt.close(fig)
                saved_paths.append(output_path)

    return saved_paths


def train_one_model(
    case_name: str,
    config: ImageFormationConfig,
    args: argparse.Namespace,
    device: torch.device,
    train_index: int,
) -> torch.nn.Module:
    train_loader = make_loader(
        config,
        args.train_samples,
        args.seed + train_index * 1_000_000,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        return_metadata=False,
    )
    val_loader = make_loader(
        config,
        args.val_samples,
        args.seed + train_index * 1_000_000 + 100_000,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        return_metadata=False,
    )
    model = build_unet(
        input_channels=1,
        num_classes=1,
        num_filters=args.num_filters,
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = CombinedGaussianLoss(from_logits=True)
    model_dir = args.output_dir / case_name
    model_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Training {case_name} ({CASE_LABELS.get(case_name, case_name)}) "
        f"for {args.epochs} epochs",
        flush=True,
    )
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_name=str(model_dir / case_name),
        progress_interval=args.progress_interval,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    checkpoint = torch.load(model_dir / f"{case_name}_best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = resolve_torch_device(args.device, verbose=True)
    configs = build_study_configs(args)
    case_order = list(configs.keys())
    save_study_config(args, configs)

    print(
        "Starting dataset study with "
        f"device={device}, epochs={args.epochs}, "
        f"train/val/test={args.train_samples}/{args.val_samples}/{args.test_samples}, "
        f"image={args.height}x{args.width}, output={args.output_dir}",
        flush=True,
    )
    print(f"U-Net channel widths: {args.num_filters}", flush=True)

    test_loaders = {
        case_name: make_loader(
            config,
            args.test_samples,
            args.seed + 5_000_000 + index * 100_000,
            args.batch_size,
            args.num_workers,
            shuffle=False,
            return_metadata=True,
        )
        for index, (case_name, config) in enumerate(configs.items())
    }

    models: Dict[str, torch.nn.Module] = {}
    rows: List[Dict[str, float]] = []
    for train_index, (train_case, config) in enumerate(configs.items()):
        model = train_one_model(train_case, config, args, device, train_index)
        models[train_case] = model
        offsets_for_model = {}
        for test_case in case_order:
            summary = evaluate_model_localization(
                model=model,
                dataloader=test_loaders[test_case],
                device=device,
                channel=0,
                apply_sigmoid=True,
                threshold_rel=args.threshold_rel,
                match_distance=args.match_distance,
            )
            row = {
                "model": train_case,
                "test_case": test_case,
                "precision": summary["precision"],
                "recall": summary["recall"],
                "f1": summary["f1"],
                "mean_error": summary["mean_error"],
                "median_error": summary["median_error"],
                "rmse": summary["rmse"],
                "tp": summary["tp"],
                "fp": summary["fp"],
                "fn": summary["fn"],
                "samples": summary["samples"],
            }
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
            offsets_for_model[test_case] = collect_matched_offsets(
                model=model,
                dataloader=test_loaders[test_case],
                device=device,
                threshold_rel=args.threshold_rel,
                match_distance=args.match_distance,
            )

        plot_offset_cloud(
            case_results=offsets_for_model,
            output_path=args.output_dir / f"offset_cloud_trained_on_{train_case}.png",
            model_name=f"trained on {CASE_LABELS.get(train_case, train_case)}",
            plot_range=args.offset_plot_range,
            bins=args.offset_bins,
            point_alpha=0.025,
            point_size=10.0,
            dpi=220,
        )

    save_metrics(rows, args.output_dir)
    ranking_path = save_ranking(rows, args.output_dir)
    plot_metric_bar_charts(
        rows,
        ranking_path=ranking_path,
        case_order=case_order,
        output_dir=args.output_dir,
    )
    plot_generalization_summary(
        rows,
        args.output_dir,
        case_order=case_order,
        model_order=case_order,
        filename="dataset_study_summary.png",
        title="Dataset Study Summary",
    )
    plot_study_heatmaps(
        rows,
        case_order=case_order,
        output_path=args.output_dir / "dataset_study_train_test_heatmaps.png",
    )
    plot_loss_curves(case_order, args.output_dir)
    save_prediction_comparison_gallery(
        models=models,
        test_loaders=test_loaders,
        case_order=case_order,
        sample_index=args.gallery_sample_index,
        device=device,
        threshold_rel=args.threshold_rel,
        output_path=args.output_dir / "dataset_study_prediction_gallery.png",
    )
    save_model_output_series(
        models=models,
        test_loaders=test_loaders,
        case_order=case_order,
        sample_indices=args.output_sample_indices,
        device=device,
        threshold_rel=args.threshold_rel,
        output_dir=args.output_dir,
    )

    ranking = json.loads(ranking_path.read_text())
    best = ranking[0]
    print(
        "Finished dataset study. "
        f"Best mean F1: {best['label']} "
        f"(mean_f1={best['mean_f1']:.4f}, mean_rmse={best['mean_rmse']:.4f}).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
