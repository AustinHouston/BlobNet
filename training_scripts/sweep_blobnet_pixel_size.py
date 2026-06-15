from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GombNet.metrics import (  # noqa: E402
    aggregate_localization_metrics,
    evaluate_heatmap_localization,
)
from GombNet.networks import build_unet  # noqa: E402
from GombNet.synthetic import (  # noqa: E402
    PeriodicLatticeConfig,
    RandomMicroscopeImageConfig,
    generate_microscope_image,
)
from GombNet.utils import resolve_torch_device  # noqa: E402
from training_scripts.io_utils import write_csv, write_json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained BlobNet checkpoint across synthetic datasets rendered "
            "at different pixel sizes, centered on the training pixel size."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/inhom_background_unet_20epoch/unet/unet_best.pth"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-size", type=int, default=64)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument(
        "--dataset",
        choices=["random", "cubic", "hexagonal"],
        default="random",
        help="Synthetic point-layout family to sweep.",
    )
    parser.add_argument(
        "--train-pixel-size-angstrom",
        type=float,
        default=0.1062231596676199,
        help="Pixel size used to interpret the training distribution.",
    )
    parser.add_argument(
        "--pixel-size-factors",
        type=float,
        nargs="+",
        default=[0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0],
        help="Pixel sizes to test as multiples of --train-pixel-size-angstrom.",
    )
    parser.add_argument(
        "--pixel-sizes-angstrom",
        type=float,
        nargs="+",
        default=None,
        help="Explicit pixel sizes to test. Overrides --pixel-size-factors.",
    )
    parser.add_argument("--num-filters", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train-sigma-min", type=float, default=2.8)
    parser.add_argument("--train-sigma-max", type=float, default=4.3)
    parser.add_argument("--train-target-sigma", type=float, default=0.9)
    parser.add_argument("--train-min-separation", type=float, default=14.5)
    parser.add_argument("--train-min-separation-range-min", type=float, default=12.5)
    parser.add_argument("--train-min-separation-range-max", type=float, default=16.7)
    parser.add_argument("--train-lattice-spacing-min", type=float, default=12.5)
    parser.add_argument("--train-lattice-spacing-max", type=float, default=16.7)
    parser.add_argument("--min-atoms", type=int, default=80)
    parser.add_argument("--max-atoms", type=int, default=180)
    parser.add_argument("--lattice-min-atoms", type=int, default=120)
    parser.add_argument("--background-min", type=float, default=0.0)
    parser.add_argument("--background-max", type=float, default=0.30)
    parser.add_argument("--inhom-background-min", type=float, default=0.0)
    parser.add_argument("--inhom-background-max", type=float, default=0.16)
    parser.add_argument("--low-freq-noise-min", type=float, default=0.05)
    parser.add_argument("--low-freq-noise-max", type=float, default=0.28)
    parser.add_argument("--read-noise-min", type=float, default=0.02)
    parser.add_argument("--read-noise-max", type=float, default=0.12)
    parser.add_argument("--total-counts-min", type=float, default=50.0)
    parser.add_argument("--total-counts-max", type=float, default=25_000.0)
    parser.add_argument("--blur-sigma-min", type=float, default=0.2)
    parser.add_argument("--blur-sigma-max", type=float, default=1.1)
    parser.add_argument("--edge-padding", type=int, default=16)
    parser.add_argument("--threshold-rel", type=float, default=0.35)
    parser.add_argument(
        "--threshold-grid",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-pixel-size threshold sweep. If set, best F1 threshold is used at each size.",
    )
    parser.add_argument("--train-match-distance", type=float, default=3.0)
    parser.add_argument("--train-peak-min-distance", type=int, default=3)
    parser.add_argument("--train-peak-window-size", type=int, default=5)
    parser.add_argument(
        "--fixed-evaluation-pixels",
        action="store_true",
        help="Keep peak extraction and match-distance in pixels fixed instead of scaling from training physical units.",
    )
    parser.add_argument("--example-size-count", type=int, default=None)
    return parser.parse_args()


def bottleneck_receptive_field_px(num_filters: Sequence[int]) -> int:
    """Return the receptive field at the end of the U-Net bottleneck block."""

    receptive_field = 1
    jump = 1
    encoder_count = max(0, len(num_filters) - 1)
    for _ in range(encoder_count):
        for _ in range(2):
            receptive_field += 2 * jump
        receptive_field += jump
        jump *= 2
    for _ in range(2):
        receptive_field += 2 * jump
    return int(receptive_field)


def load_blobnet(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = build_unet(
        input_channels=1,
        num_classes=1,
        num_filters=args.num_filters,
        dropout=args.dropout,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def scaled_range(
    train_range: tuple[float, float],
    train_pixel_size: float,
    pixel_size: float,
) -> tuple[float, float]:
    scale = float(train_pixel_size) / float(pixel_size)
    return float(train_range[0]) * scale, float(train_range[1]) * scale


def scaled_scalar(train_value: float, train_pixel_size: float, pixel_size: float) -> float:
    return float(train_value) * float(train_pixel_size) / float(pixel_size)


def common_config_kwargs(args: argparse.Namespace, pixel_size: float) -> Dict[str, Any]:
    sigma_range = scaled_range(
        (args.train_sigma_min, args.train_sigma_max),
        args.train_pixel_size_angstrom,
        pixel_size,
    )
    target_sigma = scaled_scalar(
        args.train_target_sigma,
        args.train_pixel_size_angstrom,
        pixel_size,
    )
    blur_sigma_range = scaled_range(
        (args.blur_sigma_min, args.blur_sigma_max),
        args.train_pixel_size_angstrom,
        pixel_size,
    )
    return {
        "image_shape": (args.height, args.width),
        "sigma_range": sigma_range,
        "target_sigma": max(0.15, target_sigma),
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
        "total_counts_range": (args.total_counts_min, args.total_counts_max),
        "counts_per_pixel_range": None,
        "blur_sigma_range": blur_sigma_range,
        "edge_padding": args.edge_padding,
    }


def build_config(args: argparse.Namespace, pixel_size: float):
    common = common_config_kwargs(args, pixel_size)
    if args.dataset == "random":
        min_separation_range = scaled_range(
            (
                args.train_min_separation_range_min,
                args.train_min_separation_range_max,
            ),
            args.train_pixel_size_angstrom,
            pixel_size,
        )
        return RandomMicroscopeImageConfig(
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
            min_separation=scaled_scalar(
                args.train_min_separation,
                args.train_pixel_size_angstrom,
                pixel_size,
            ),
            min_separation_range=min_separation_range,
            **common,
        )
    spacing_range = scaled_range(
        (args.train_lattice_spacing_min, args.train_lattice_spacing_max),
        args.train_pixel_size_angstrom,
        pixel_size,
    )
    return PeriodicLatticeConfig(
        lattice_type=args.dataset,
        lattice_spacing_range=spacing_range,
        jitter_std_range=(0.0, max(0.0, 0.25 * args.train_pixel_size_angstrom / pixel_size)),
        vacancy_fraction_range=(0.0, 0.03),
        min_atoms=args.lattice_min_atoms,
        **common,
    )


def pixel_sizes_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.pixel_sizes_angstrom is not None:
        values = np.asarray(args.pixel_sizes_angstrom, dtype=float)
    else:
        values = float(args.train_pixel_size_angstrom) * np.asarray(
            args.pixel_size_factors,
            dtype=float,
        )
    if np.any(values <= 0):
        raise ValueError("Pixel sizes must all be positive.")
    return np.asarray(sorted(values.tolist()), dtype=float)


def evaluation_params(args: argparse.Namespace, pixel_size: float) -> Dict[str, Any]:
    if args.fixed_evaluation_pixels:
        return {
            "match_distance": float(args.train_match_distance),
            "min_distance": int(args.train_peak_min_distance),
            "peak_window_size": int(args.train_peak_window_size),
        }
    scale = float(args.train_pixel_size_angstrom) / float(pixel_size)
    return {
        "match_distance": max(0.75, float(args.train_match_distance) * scale),
        "min_distance": max(1, int(round(float(args.train_peak_min_distance) * scale))),
        "peak_window_size": max(3, int(round(float(args.train_peak_window_size) * scale)) | 1),
    }


def predict_heatmap(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None].to(device)
    with torch.no_grad():
        prediction = torch.sigmoid(model(image_tensor))[0, 0].detach().cpu().numpy()
    return prediction.astype(np.float32)


def evaluate_pixel_size(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    pixel_size: float,
    size_index: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    config = build_config(args, pixel_size)
    params = evaluation_params(args, pixel_size)
    thresholds = args.threshold_grid if args.threshold_grid is not None else [args.threshold_rel]
    per_threshold_results: Dict[float, List[Dict[str, Any]]] = {
        float(threshold): [] for threshold in thresholds
    }

    example: Dict[str, Any] = {}
    for sample_index in range(int(args.samples_per_size)):
        seed = int(args.seed) + size_index * 100_000 + sample_index
        record = generate_microscope_image(config, np.random.default_rng(seed))
        image = np.asarray(record["image"], dtype=np.float32)
        heatmap = predict_heatmap(model, image, device)
        true_coordinates = np.asarray(record["coordinates"], dtype=np.float32)
        for threshold in thresholds:
            per_threshold_results[float(threshold)].append(
                evaluate_heatmap_localization(
                    heatmap,
                    true_coordinates,
                    threshold_rel=float(threshold),
                    match_distance=params["match_distance"],
                    min_distance=params["min_distance"],
                    peak_window_size=params["peak_window_size"],
                )
            )
        if sample_index == 0:
            default_threshold = float(thresholds[0])
            result = evaluate_heatmap_localization(
                heatmap,
                true_coordinates,
                threshold_rel=default_threshold,
                match_distance=params["match_distance"],
                min_distance=params["min_distance"],
                peak_window_size=params["peak_window_size"],
            )
            example = {
                "image": image,
                "target": np.asarray(record["target"], dtype=np.float32),
                "heatmap": heatmap,
                "coordinates": true_coordinates,
                "predicted_coordinates": result["predicted_coordinates"],
            }

    summaries = {
        threshold: aggregate_localization_metrics(results)
        for threshold, results in per_threshold_results.items()
    }
    best_threshold = sorted(
        summaries,
        key=lambda threshold: (
            -float(summaries[threshold]["f1"]),
            float(summaries[threshold]["rmse"])
            if not math.isnan(float(summaries[threshold]["rmse"]))
            else float("inf"),
        ),
    )[0]
    best_summary = summaries[best_threshold]
    if example:
        params_for_example = evaluation_params(args, pixel_size)
        example_result = evaluate_heatmap_localization(
            example["heatmap"],
            example["coordinates"],
            threshold_rel=float(best_threshold),
            match_distance=params_for_example["match_distance"],
            min_distance=params_for_example["min_distance"],
            peak_window_size=params_for_example["peak_window_size"],
        )
        example["predicted_coordinates"] = example_result["predicted_coordinates"]

    sigma_range = config.sigma_range
    spacing_range = (
        config.min_separation_range
        if isinstance(config, RandomMicroscopeImageConfig)
        else config.lattice_spacing_range
    )
    row = {
        "pixel_size_angstrom": float(pixel_size),
        "pixel_size_factor": float(pixel_size) / float(args.train_pixel_size_angstrom),
        "dataset": args.dataset,
        "threshold_rel": float(best_threshold),
        "precision": float(best_summary["precision"]),
        "recall": float(best_summary["recall"]),
        "f1": float(best_summary["f1"]),
        "mean_error_px": float(best_summary["mean_error"]),
        "median_error_px": float(best_summary["median_error"]),
        "rmse_px": float(best_summary["rmse"]),
        "tp": int(best_summary["tp"]),
        "fp": int(best_summary["fp"]),
        "fn": int(best_summary["fn"]),
        "samples": int(best_summary["samples"]),
        "sigma_min_px": float(sigma_range[0]),
        "sigma_max_px": float(sigma_range[1]),
        "sigma_mean_px": float(np.mean(sigma_range)),
        "feature_fwhm_mean_px": float(2.355 * np.mean(sigma_range)),
        "spacing_min_px": float(spacing_range[0]),
        "spacing_max_px": float(spacing_range[1]),
        "spacing_mean_px": float(np.mean(spacing_range)),
        "target_sigma_px": float(config.target_sigma),
        "match_distance_px": float(params["match_distance"]),
        "peak_min_distance_px": int(params["min_distance"]),
        "peak_window_size_px": int(params["peak_window_size"]),
    }
    return row, example


def plot_accuracy(rows: Sequence[Dict[str, Any]], rf_px: int, args: argparse.Namespace) -> Path:
    output_path = args.output_dir / "pixel_size_accuracy_vs_feature_rf.png"
    pixel_sizes = np.asarray([float(row["pixel_size_angstrom"]) for row in rows], dtype=float)
    f1 = np.asarray([float(row["f1"]) for row in rows], dtype=float)
    rmse = np.asarray([float(row["rmse_px"]) for row in rows], dtype=float)
    sigma_ratio = np.asarray([float(row["feature_fwhm_mean_px"]) / rf_px for row in rows], dtype=float)
    spacing_ratio = np.asarray([float(row["spacing_mean_px"]) / rf_px for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.5), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"BlobNet Pixel-Size Sweep ({args.dataset})",
        fontsize=20,
        fontweight="bold",
    )
    axes[0].plot(pixel_sizes, f1, marker="o", linewidth=2.2, color="#2f7f6f", label="F1")
    axes[0].axvline(args.train_pixel_size_angstrom, color="black", linestyle="--", linewidth=1.3, label="training pixel size")
    axes[0].set_ylabel("Localization F1", fontsize=13)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower left")
    for x_value, y_value in zip(pixel_sizes, f1):
        axes[0].text(x_value, y_value + 0.025, f"{y_value:.3f}", ha="center", fontsize=9)

    rmse_axis = axes[0].twinx()
    rmse_axis.plot(pixel_sizes, rmse, marker="s", linewidth=1.8, color="#b65f3b", label="RMSE")
    rmse_axis.set_ylabel("RMSE (px)", fontsize=13, color="#8f462c")
    rmse_axis.tick_params(axis="y", labelcolor="#8f462c")

    axes[1].plot(
        pixel_sizes,
        sigma_ratio,
        marker="o",
        linewidth=2.2,
        color="#4065a3",
        label="mean feature FWHM / bottleneck RF",
    )
    axes[1].plot(
        pixel_sizes,
        spacing_ratio,
        marker="s",
        linewidth=2.2,
        color="#8b5a9f",
        label="mean atom spacing / bottleneck RF",
    )
    axes[1].axvline(args.train_pixel_size_angstrom, color="black", linestyle="--", linewidth=1.3)
    axes[1].set_xlabel("Pixel size (angstrom / px)", fontsize=13)
    axes[1].set_ylabel(f"Size relative to bottleneck RF ({rf_px} px)", fontsize=13)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    for row in rows:
        axes[1].text(
            float(row["pixel_size_angstrom"]),
            float(row["spacing_mean_px"]) / rf_px + 0.01,
            f"{float(row['spacing_mean_px']):.1f}px spacing",
            ha="center",
            fontsize=8,
            rotation=20,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def normalize_image(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = array - float(array.min())
    peak = float(array.max())
    if peak > 0:
        array = array / peak
    return array


def plot_examples(
    rows: Sequence[Dict[str, Any]],
    examples: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Path:
    output_path = args.output_dir / "pixel_size_target_output_examples.png"
    if args.example_size_count is not None and args.example_size_count > 0:
        indices = np.linspace(0, len(examples) - 1, min(args.example_size_count, len(examples)))
        selected_indices = sorted({int(round(index)) for index in indices})
    else:
        selected_indices = list(range(len(examples)))

    columns = ["Input", "Target", "BlobNet output", "Overlay"]
    fig, axes = plt.subplots(
        len(selected_indices),
        len(columns),
        figsize=(13.5, 3.35 * max(1, len(selected_indices))),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row_position, example_index in enumerate(selected_indices):
        row = rows[example_index]
        example = examples[example_index]
        image = example["image"]
        target = example["target"]
        heatmap = example["heatmap"]
        true_coords = np.asarray(example["coordinates"], dtype=np.float32)
        pred_coords = np.asarray(example["predicted_coordinates"], dtype=np.float32)

        axes[row_position, 0].imshow(image, cmap="gray")
        axes[row_position, 1].imshow(target, cmap="magma", vmin=0.0, vmax=max(1e-6, float(target.max())))
        axes[row_position, 2].imshow(heatmap, cmap="magma", vmin=0.0, vmax=max(1e-6, float(heatmap.max())))
        axes[row_position, 3].imshow(image, cmap="gray")
        if len(true_coords):
            axes[row_position, 3].scatter(
                true_coords[:, 1],
                true_coords[:, 0],
                s=14,
                facecolors="none",
                edgecolors="lime",
                linewidths=0.7,
                label="true",
            )
        if len(pred_coords):
            axes[row_position, 3].scatter(
                pred_coords[:, 1],
                pred_coords[:, 0],
                s=12,
                c="cyan",
                marker="x",
                linewidths=0.7,
                label="pred",
            )
        label_text = (
            f"{float(row['pixel_size_angstrom']):.4f} A/px "
            f"({float(row['pixel_size_factor']):.2f}x train)\n"
            f"F1={float(row['f1']):.3f}, RMSE={float(row['rmse_px']):.3f}px, "
            f"sigma={float(row['sigma_mean_px']):.2f}px"
        )
        axes[row_position, 0].set_ylabel(label_text, fontsize=10)
        for column_index, column_name in enumerate(columns):
            if row_position == 0:
                axes[row_position, column_index].set_title(column_name, fontsize=13)
            axes[row_position, column_index].axis("off")

    fig.suptitle(
        f"Targets and BlobNet Outputs Across Pixel Size ({args.dataset})",
        fontsize=18,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_run_config(args: argparse.Namespace, rf_px: int, pixel_sizes: np.ndarray) -> None:
    payload = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "bottleneck_receptive_field_px": rf_px,
        "tested_pixel_sizes_angstrom": pixel_sizes.tolist(),
        "training_physical_priors": {
            "sigma_range_angstrom": [
                args.train_sigma_min * args.train_pixel_size_angstrom,
                args.train_sigma_max * args.train_pixel_size_angstrom,
            ],
            "target_sigma_angstrom": args.train_target_sigma * args.train_pixel_size_angstrom,
            "min_separation_range_angstrom": [
                args.train_min_separation_range_min * args.train_pixel_size_angstrom,
                args.train_min_separation_range_max * args.train_pixel_size_angstrom,
            ],
            "lattice_spacing_range_angstrom": [
                args.train_lattice_spacing_min * args.train_pixel_size_angstrom,
                args.train_lattice_spacing_max * args.train_pixel_size_angstrom,
            ],
        },
    }
    write_json(payload, args.output_dir / "pixel_size_sweep_config.json")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device, verbose=True)
    model = load_blobnet(args, device)
    pixel_sizes = pixel_sizes_from_args(args)
    rf_px = bottleneck_receptive_field_px(args.num_filters)
    save_run_config(args, rf_px, pixel_sizes)

    rows: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []
    for size_index, pixel_size in enumerate(pixel_sizes):
        print(
            f"Evaluating pixel_size={pixel_size:.5f} A/px "
            f"({pixel_size / args.train_pixel_size_angstrom:.2f}x training)",
            flush=True,
        )
        row, example = evaluate_pixel_size(args, model, device, float(pixel_size), size_index)
        row["bottleneck_receptive_field_px"] = rf_px
        row["bottleneck_receptive_field_angstrom"] = rf_px * float(pixel_size)
        row["feature_fwhm_over_bottleneck_rf"] = (
            float(row["feature_fwhm_mean_px"]) / float(rf_px)
        )
        row["spacing_over_bottleneck_rf"] = float(row["spacing_mean_px"]) / float(rf_px)
        rows.append(row)
        examples.append(example)
        print(
            f"  F1={row['f1']:.4f}, RMSE={row['rmse_px']:.4f}px, "
            f"sigma_mean={row['sigma_mean_px']:.2f}px, spacing_mean={row['spacing_mean_px']:.2f}px",
            flush=True,
        )

    write_csv(rows, args.output_dir / "pixel_size_metrics.csv")
    write_json(rows, args.output_dir / "pixel_size_metrics.json")
    accuracy_path = plot_accuracy(rows, rf_px, args)
    examples_path = plot_examples(rows, examples, args)
    best = sorted(rows, key=lambda row: (-float(row["f1"]), float(row["rmse_px"])))[0]
    print(
        "Finished pixel-size sweep. "
        f"Best F1={best['f1']:.4f} at {best['pixel_size_angstrom']:.5f} A/px "
        f"({best['pixel_size_factor']:.2f}x training).",
        flush=True,
    )
    print(f"Saved metrics to {args.output_dir / 'pixel_size_metrics.csv'}", flush=True)
    print(f"Saved accuracy plot to {accuracy_path}", flush=True)
    print(f"Saved example plot to {examples_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
