from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np

from GombNet.synthetic import (
    AseStructureProjectionConfig,
    ImageFormationConfig,
    RandomMicroscopeImageConfig,
    TightSpacingRandomMicroscopeImageConfig,
    generate_microscope_image,
)


MATERIAL_CASE_LABELS = {
    "random_min_distance": "Blob, min distance",
    "tight_random_spacing": "Blob, tight spacing",
    "sto_ase": "STO, ASE cubic",
    "ws2_mx2": "WS2, ASE mx2 hex",
}


def add_material_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--pixel-size-angstrom", type=float, default=0.1062231596676199)
    parser.add_argument("--physical-spacing-min-angstrom", type=float, default=1.80)
    parser.add_argument("--physical-spacing-max-angstrom", type=float, default=2.10)
    parser.add_argument("--random-min-atoms", type=int, default=650)
    parser.add_argument("--random-max-atoms", type=int, default=950)
    parser.add_argument("--min-separation-px", type=float, default=None)
    parser.add_argument("--min-separation-range-min-px", type=float, default=None)
    parser.add_argument("--min-separation-range-max-px", type=float, default=None)
    parser.add_argument("--tight-spacing-min-px", type=float, default=None)
    parser.add_argument("--tight-spacing-max-px", type=float, default=None)
    parser.add_argument("--tight-spacing-jitter-min", type=float, default=0.03)
    parser.add_argument("--tight-spacing-jitter-max", type=float, default=0.08)
    parser.add_argument("--tight-min-spacing-fraction", type=float, default=0.86)
    parser.add_argument("--sigma-min", type=float, default=2.8)
    parser.add_argument("--sigma-max", type=float, default=4.3)
    parser.add_argument("--target-sigma", type=float, default=0.9)
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
    parser.add_argument("--material-rotation-min", type=float, default=0.0)
    parser.add_argument("--material-rotation-max", type=float, default=180.0)
    parser.add_argument("--material-jitter-min-angstrom", type=float, default=0.0)
    parser.add_argument("--material-jitter-max-angstrom", type=float, default=0.08)
    parser.add_argument("--material-column-merge-tolerance-angstrom", type=float, default=0.08)
    parser.add_argument(
        "--no-merge-material-columns",
        action="store_true",
        help="Keep every atom as a separate projected target instead of merging z-stacked columns.",
    )
    parser.add_argument("--sto-species-intensity-power", type=float, default=1.45)
    parser.add_argument("--ws2-species-intensity-power", type=float, default=1.60)


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


def physical_spacing_range_px(args: argparse.Namespace) -> Tuple[float, float]:
    return (
        float(args.physical_spacing_min_angstrom) / float(args.pixel_size_angstrom),
        float(args.physical_spacing_max_angstrom) / float(args.pixel_size_angstrom),
    )


def blob_spacing_ranges_px(args: argparse.Namespace) -> tuple[tuple[float, float], tuple[float, float], float]:
    derived_min, derived_max = physical_spacing_range_px(args)
    min_sep_min = (
        float(args.min_separation_range_min_px)
        if args.min_separation_range_min_px is not None
        else derived_min
    )
    min_sep_max = (
        float(args.min_separation_range_max_px)
        if args.min_separation_range_max_px is not None
        else derived_max
    )
    min_separation = (
        float(args.min_separation_px)
        if args.min_separation_px is not None
        else 0.5 * (min_sep_min + min_sep_max)
    )
    tight_min = (
        float(args.tight_spacing_min_px)
        if args.tight_spacing_min_px is not None
        else derived_min
    )
    tight_max = (
        float(args.tight_spacing_max_px)
        if args.tight_spacing_max_px is not None
        else derived_max
    )
    return (min_sep_min, min_sep_max), (tight_min, tight_max), min_separation


def common_image_settings(args: argparse.Namespace) -> Dict[str, object]:
    total_counts_range, counts_per_pixel_range = count_ranges_from_args(args)
    return {
        "image_shape": (args.height, args.width),
        "sigma_range": (args.sigma_min, args.sigma_max),
        "target_sigma": args.target_sigma,
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


def build_material_study_configs(args: argparse.Namespace) -> Dict[str, ImageFormationConfig]:
    common = common_image_settings(args)
    min_sep_range, tight_spacing_range, min_separation = blob_spacing_ranges_px(args)
    material_rotation = (args.material_rotation_min, args.material_rotation_max)
    material_jitter = (
        args.material_jitter_min_angstrom,
        args.material_jitter_max_angstrom,
    )
    merge_material_columns = not args.no_merge_material_columns
    return {
        "random_min_distance": RandomMicroscopeImageConfig(
            min_atoms=args.random_min_atoms,
            max_atoms=args.random_max_atoms,
            min_separation=min_separation,
            min_separation_range=min_sep_range,
            **common,
        ),
        "tight_random_spacing": TightSpacingRandomMicroscopeImageConfig(
            min_atoms=args.random_min_atoms,
            max_atoms=args.random_max_atoms,
            nearest_neighbor_spacing_range=tight_spacing_range,
            spacing_jitter_fraction_range=(
                args.tight_spacing_jitter_min,
                args.tight_spacing_jitter_max,
            ),
            min_spacing_fraction=args.tight_min_spacing_fraction,
            **common,
        ),
        "sto_ase": AseStructureProjectionConfig(
            structure_name="sto",
            pixel_size_angstrom=args.pixel_size_angstrom,
            rotation_range=material_rotation,
            position_jitter_std_range=material_jitter,
            species_intensity_power=args.sto_species_intensity_power,
            merge_projected_columns=merge_material_columns,
            column_merge_tolerance_angstrom=args.material_column_merge_tolerance_angstrom,
            **common,
        ),
        "ws2_mx2": AseStructureProjectionConfig(
            structure_name="ws2_mx2",
            pixel_size_angstrom=args.pixel_size_angstrom,
            rotation_range=material_rotation,
            position_jitter_std_range=material_jitter,
            species_intensity_power=args.ws2_species_intensity_power,
            merge_projected_columns=merge_material_columns,
            column_merge_tolerance_angstrom=args.material_column_merge_tolerance_angstrom,
            **common,
        ),
    }


def _nearest_neighbor_stats(coordinates: np.ndarray) -> dict[str, float]:
    coordinates = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
    if len(coordinates) < 2:
        return {"median_nn_px": float("nan"), "p10_nn_px": float("nan"), "p90_nn_px": float("nan")}
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=-1))
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    nearest = nearest[np.isfinite(nearest) & (nearest > 1.0e-3)]
    if len(nearest) == 0:
        return {"median_nn_px": float("nan"), "p10_nn_px": float("nan"), "p90_nn_px": float("nan")}
    return {
        "median_nn_px": float(np.median(nearest)),
        "p10_nn_px": float(np.percentile(nearest, 10.0)),
        "p90_nn_px": float(np.percentile(nearest, 90.0)),
    }


def summarize_config(
    case_name: str,
    config: ImageFormationConfig,
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, object]:
    record = generate_microscope_image(config, np.random.default_rng(seed))
    stats = _nearest_neighbor_stats(record["coordinates"])
    pixel_size = float(args.pixel_size_angstrom)
    summary = {
        "case": case_name,
        "label": MATERIAL_CASE_LABELS.get(case_name, case_name),
        "class": type(config).__name__,
        "image_shape": list(config.image_shape),
        "pixel_size_angstrom": pixel_size,
        "field_of_view_angstrom": [
            float(config.image_shape[0] * pixel_size),
            float(config.image_shape[1] * pixel_size),
        ],
        "feature_sigma_px": list(config.sigma_range),
        "feature_sigma_angstrom": [
            float(config.sigma_range[0] * pixel_size),
            float(config.sigma_range[1] * pixel_size),
        ],
        "target_sigma_px": float(config.target_sigma),
        "target_sigma_angstrom": float(config.target_sigma * pixel_size),
        "visible_points_in_example": int(len(record["coordinates"])),
        "median_nn_px": stats["median_nn_px"],
        "median_nn_angstrom": float(stats["median_nn_px"] * pixel_size),
        "p10_nn_px": stats["p10_nn_px"],
        "p90_nn_px": stats["p90_nn_px"],
        "config": asdict(config),
    }
    if isinstance(config, RandomMicroscopeImageConfig):
        summary["spacing_range_px"] = list(config.min_separation_range or (config.min_separation, config.min_separation))
    elif isinstance(config, TightSpacingRandomMicroscopeImageConfig):
        summary["spacing_range_px"] = list(config.nearest_neighbor_spacing_range)
    elif isinstance(config, AseStructureProjectionConfig):
        summary["structure_name"] = config.structure_name
    if "spacing_range_px" in summary:
        summary["spacing_range_angstrom"] = [
            float(value * pixel_size) for value in summary["spacing_range_px"]
        ]
    return summary
