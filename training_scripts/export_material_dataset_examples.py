from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GombNet.synthetic import generate_microscope_image
from training_scripts.material_dataset_configs import (
    MATERIAL_CASE_LABELS,
    add_material_dataset_args,
    build_material_study_configs,
    summarize_config,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export side-by-side examples for blob and ASE-material datasets with "
            "matched pixel size and feature-size settings."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    add_material_dataset_args(parser)
    return parser.parse_args()


def _add_scalebar(ax, pixel_size_angstrom: float, length_angstrom: float = 10.0) -> None:
    bar_px = length_angstrom / pixel_size_angstrom
    x0 = 0.06
    y0 = 0.92
    ax.plot(
        [x0, x0 + bar_px / ax.images[0].get_array().shape[1]],
        [y0, y0],
        color="white",
        linewidth=3.0,
        transform=ax.transAxes,
        solid_capstyle="butt",
    )
    ax.text(
        x0,
        y0 - 0.045,
        f"{length_angstrom:g} A",
        color="white",
        fontsize=8,
        transform=ax.transAxes,
        ha="left",
        va="top",
    )


def plot_examples(args: argparse.Namespace) -> Path:
    configs = build_material_study_configs(args)
    case_order = list(configs.keys())
    columns_per_example = 3
    fig, axes = plt.subplots(
        len(case_order),
        max(1, args.examples) * columns_per_example,
        figsize=(3.25 * max(1, args.examples) * columns_per_example, 3.25 * len(case_order)),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(case_order), -1)

    for row_index, case_name in enumerate(case_order):
        config = configs[case_name]
        for example_index in range(max(1, args.examples)):
            record = generate_microscope_image(
                config,
                np.random.default_rng(args.seed + row_index * 100_000 + example_index),
            )
            image = record["image"]
            target = record["target"]
            coordinates = np.asarray(record["coordinates"], dtype=np.float32)
            base_col = example_index * columns_per_example
            label = MATERIAL_CASE_LABELS.get(case_name, case_name)
            title_suffix = (
                f"sample {example_index}\n"
                f"n={len(coordinates)}, sigma={config.sigma_range[0]:.1f}-{config.sigma_range[1]:.1f}px"
            )

            ax = axes[row_index, base_col]
            ax.imshow(image, cmap="gray")
            ax.set_title(f"{label}\ninput, {title_suffix}", fontsize=10)
            _add_scalebar(ax, args.pixel_size_angstrom)

            ax = axes[row_index, base_col + 1]
            ax.imshow(target, cmap="magma", vmin=0.0, vmax=max(float(target.max()), 1e-6))
            ax.set_title("target heatmap", fontsize=10)

            ax = axes[row_index, base_col + 2]
            ax.imshow(image, cmap="gray")
            if len(coordinates):
                ax.scatter(
                    coordinates[:, 1],
                    coordinates[:, 0],
                    s=9,
                    facecolors="none",
                    edgecolors="#7CFF4F",
                    linewidths=0.45,
                )
            ax.set_title("input + target centers", fontsize=10)
            _add_scalebar(ax, args.pixel_size_angstrom)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        "Matched Dataset Examples: Blob Controls vs ASE STO and ASE mx2('WS2')",
        fontsize=15,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "material_dataset_examples.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_summary(args: argparse.Namespace) -> tuple[Path, Path]:
    configs = build_material_study_configs(args)
    summaries = [
        summarize_config(case_name, config, args, seed=args.seed + index * 10_000)
        for index, (case_name, config) in enumerate(configs.items())
    ]
    json_path = args.output_dir / "material_dataset_config_summary.json"
    json_path.write_text(json.dumps(summaries, indent=2))

    csv_path = args.output_dir / "material_dataset_config_summary.csv"
    fieldnames = [
        "case",
        "label",
        "class",
        "pixel_size_angstrom",
        "field_of_view_angstrom",
        "feature_sigma_px",
        "feature_sigma_angstrom",
        "target_sigma_px",
        "target_sigma_angstrom",
        "visible_points_in_example",
        "median_nn_px",
        "median_nn_angstrom",
        "p10_nn_px",
        "p90_nn_px",
        "spacing_range_px",
        "spacing_range_angstrom",
        "structure_name",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    key: json.dumps(summary.get(key))
                    if isinstance(summary.get(key), (list, dict))
                    else summary.get(key)
                    for key in fieldnames
                }
            )
    return json_path, csv_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = plot_examples(args)
    json_path, csv_path = save_summary(args)
    print(f"Saved example gallery: {image_path}", flush=True)
    print(f"Saved config summary: {json_path}", flush=True)
    print(f"Saved CSV summary: {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
