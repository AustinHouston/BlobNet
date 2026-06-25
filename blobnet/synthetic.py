from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImageFormationConfig:
    """Settings shared by every synthetic image once atom positions are known."""

    image_shape: Tuple[int, int] = (128, 128)
    sigma_range: Tuple[float, float] = (0.7, 1.8)
    intensity_range: Tuple[float, float] = (0.3, 1.0)
    target_sigma: float = 0.9
    background_range: Tuple[float, float] = (0.02, 0.15)
    gradient_range: Tuple[float, float] = (-0.08, 0.08)
    inhomogeneous_background_range: Tuple[float, float] = (0.0, 0.0)
    inhomogeneous_background_sigma_fraction_range: Tuple[float, float] = (0.18, 0.45)
    low_frequency_noise_range: Tuple[float, float] = (0.0, 0.12)
    low_frequency_sigma_fraction_range: Tuple[float, float] = (0.04, 0.12)
    read_noise_std_range: Tuple[float, float] = (0.005, 0.04)
    # Choose either total_counts_range or counts_per_pixel_range, not both.
    total_counts_range: Optional[Tuple[float, float]] = (3_000.0, 20_000.0)
    counts_per_pixel_range: Optional[Tuple[float, float]] = None
    blur_sigma_range: Tuple[float, float] = (0.0, 1.0)
    edge_padding: int = 0
    normalize_input: bool = True
    clamp_target: bool = True


@dataclass(frozen=True)
class RandomAtomImageConfig(ImageFormationConfig):
    """Randomly placed atom-like peaks."""

    min_atoms: int = 8
    max_atoms: int = 36
    min_separation: float = 4.0
    min_separation_range: Optional[Tuple[float, float]] = None
    subpixel_jitter_range: Tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class PeriodicLatticeConfig(ImageFormationConfig):
    """Periodic cubic or hexagonal point locations."""

    lattice_type: str = "hexagonal"
    lattice_spacing_range: Tuple[float, float] = (8.0, 12.0)
    rotation_range: Tuple[float, float] = (0.0, 180.0)
    jitter_std_range: Tuple[float, float] = (0.0, 0.15)
    vacancy_fraction_range: Tuple[float, float] = (0.0, 0.02)
    min_atoms: int = 24


@dataclass(frozen=True)
class AseStructureProjectionConfig(ImageFormationConfig):
    """Projected atomic structures generated from small ASE unit cells."""

    image_shape: Tuple[int, int] = (256, 256)
    structure_name: str = "graphene"
    pixel_size_angstrom: float = 0.12
    rotation_range: Tuple[float, float] = (0.0, 180.0)
    position_jitter_std_range: Tuple[float, float] = (0.0, 0.08)
    species_intensity_power: float = 1.6
    repeat_thickness: int = 1
    merge_projected_columns: bool = False
    column_merge_tolerance_angstrom: float = 0.08


@dataclass(frozen=True)
class PointCloud:
    """Locations and optional per-point brightness used by the shared renderer."""

    coordinates: np.ndarray
    target_coordinates: np.ndarray
    intensities: Optional[np.ndarray] = None
    sigmas: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


def _as_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _sample_scalar(rng: np.random.Generator, value_range: Sequence[float]) -> float:
    low, high = float(value_range[0]), float(value_range[1])
    if high < low:
        raise ValueError(f"Expected an ascending range, got {value_range!r}")
    return float(rng.uniform(low, high))


def _sample_count_scale(
    rng: np.random.Generator,
    config: ImageFormationConfig,
    image_shape: Tuple[int, int],
) -> int:
    total_counts_range = config.total_counts_range
    counts_per_pixel_range = config.counts_per_pixel_range
    if (total_counts_range is None) == (counts_per_pixel_range is None):
        raise ValueError(
            "Set exactly one of total_counts_range or counts_per_pixel_range."
        )

    if total_counts_range is not None:
        count_scale = _sample_scalar(rng, total_counts_range)
    else:
        counts_per_pixel = _sample_scalar(rng, counts_per_pixel_range)
        count_scale = counts_per_pixel * float(image_shape[0] * image_shape[1])
    return max(0, int(round(count_scale)))


def _make_coordinate_grids(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    y = np.arange(shape[0], dtype=np.float32)
    x = np.arange(shape[1], dtype=np.float32)
    return np.meshgrid(y, x, indexing="ij")


def _support_margin_for_sigma(sigma_range: Sequence[float]) -> float:
    return 4.0 * float(sigma_range[1])


def _expanded_shape(shape: Tuple[int, int], padding: int) -> Tuple[int, int]:
    return int(shape[0]) + 2 * padding, int(shape[1]) + 2 * padding


def _shift_coordinates(
    coordinates: np.ndarray,
    delta_yx: Sequence[float],
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.size == 0:
        return coordinates.reshape(0, 2)
    return (coordinates + np.asarray(delta_yx, dtype=np.float32)).astype(np.float32)


def _in_frame_mask(coordinates: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.size == 0:
        return np.zeros((0,), dtype=bool)
    height, width = shape
    return (
        (coordinates[:, 0] >= 0.0)
        & (coordinates[:, 0] < float(height))
        & (coordinates[:, 1] >= 0.0)
        & (coordinates[:, 1] < float(width))
    )


def _limit_visible_coordinates(
    coordinates: np.ndarray,
    shape: Tuple[int, int],
    max_visible_count: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    visible = _in_frame_mask(coordinates, shape)
    visible_indices = np.flatnonzero(visible)
    if len(visible_indices) <= int(max_visible_count):
        return coordinates, coordinates[visible]

    keep_visible = rng.choice(
        visible_indices,
        size=int(max_visible_count),
        replace=False,
    )
    keep_mask = ~visible
    keep_mask[keep_visible] = True
    trimmed_coordinates = coordinates[keep_mask]
    return trimmed_coordinates, trimmed_coordinates[_in_frame_mask(trimmed_coordinates, shape)]


def _stamp_gaussian(
    image: np.ndarray,
    center_yx: Sequence[float],
    sigma: float,
    amplitude: float = 1.0,
    mode: str = "sum",
    truncate: float = 4.0,
) -> None:
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    center_y, center_x = float(center_yx[0]), float(center_yx[1])
    height, width = image.shape
    radius = max(1, int(np.ceil(truncate * sigma)))
    y0 = max(0, int(np.floor(center_y)) - radius)
    y1 = min(height, int(np.floor(center_y)) + radius + 2)
    x0 = max(0, int(np.floor(center_x)) - radius)
    x1 = min(width, int(np.floor(center_x)) + radius + 2)

    yy, xx = _make_coordinate_grids((y1 - y0, x1 - x0))
    yy = yy + y0 - center_y
    xx = xx + x0 - center_x
    patch = amplitude * np.exp(-(yy**2 + xx**2) / (2.0 * sigma**2))

    if mode == "sum":
        image[y0:y1, x0:x1] += patch.astype(np.float32)
    elif mode == "max":
        image[y0:y1, x0:x1] = np.maximum(
            image[y0:y1, x0:x1], patch.astype(np.float32)
        )
    else:
        raise ValueError(f"Unsupported stamping mode: {mode}")


def _normalize(image: np.ndarray) -> np.ndarray:
    image = image - image.min()
    peak = float(image.max())
    if peak > 0:
        image = image / peak
    return image


def _smooth_unit_field(
    rng: np.random.Generator,
    shape: Tuple[int, int],
    sigma_fraction: float,
) -> np.ndarray:
    field = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    field = gaussian_filter(field, sigma=max(shape) * float(sigma_fraction), mode="reflect")
    return _normalize(field).astype(np.float32)


def _image_settings_from(config: ImageFormationConfig) -> Dict[str, Any]:
    fields = ImageFormationConfig.__dataclass_fields__
    return {name: getattr(config, name) for name in fields}


def _valid_sampling_bounds(
    shape: Tuple[int, int],
    boundary_margin: float,
) -> tuple[float, float, float, float]:
    height, width = int(shape[0]), int(shape[1])
    margin = max(0.0, float(boundary_margin))
    y_min = margin
    x_min = margin
    y_max = float(height) - margin
    x_max = float(width) - margin
    if y_max <= y_min or x_max <= x_min:
        raise RuntimeError(
            f'No valid random-atom sampling area for shape={shape} and margin={margin:.3f}.'
        )
    return y_min, y_max, x_min, x_max


def _grid_index_for_coordinate(
    coordinate: np.ndarray,
    y_min: float,
    x_min: float,
    cell_size: float,
) -> tuple[int, int]:
    row = int((float(coordinate[0]) - y_min) / cell_size)
    col = int((float(coordinate[1]) - x_min) / cell_size)
    return row, col


def _is_poisson_candidate_valid(
    candidate: np.ndarray,
    coordinates: list[np.ndarray],
    grid: np.ndarray,
    y_min: float,
    x_min: float,
    cell_size: float,
    min_separation: float,
) -> bool:
    row, col = _grid_index_for_coordinate(candidate, y_min, x_min, cell_size)
    if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
        return False

    row0 = max(0, row - 2)
    row1 = min(grid.shape[0], row + 3)
    col0 = max(0, col - 2)
    col1 = min(grid.shape[1], col + 3)
    min_squared = float(min_separation) ** 2

    for nearby_index in grid[row0:row1, col0:col1].ravel():
        if nearby_index < 0:
            continue
        delta = candidate - coordinates[int(nearby_index)]
        if float(delta @ delta) < min_squared:
            return False
    return True


def _store_poisson_coordinate(
    coordinate: np.ndarray,
    coordinates: list[np.ndarray],
    active_indices: list[int],
    grid: np.ndarray,
    y_min: float,
    x_min: float,
    cell_size: float,
) -> None:
    coordinates.append(coordinate.astype(np.float32))
    active_indices.append(len(coordinates) - 1)
    row, col = _grid_index_for_coordinate(coordinate, y_min, x_min, cell_size)
    grid[row, col] = len(coordinates) - 1


def sample_atom_coordinates(
    config: RandomAtomImageConfig,
    rng: Optional[np.random.Generator] = None,
    atom_count: Optional[int] = None,
    min_separation: Optional[float] = None,
    boundary_margin: Optional[float] = None,
) -> np.ndarray:
    """Sample random atom centers with Bridson Poisson-disk sampling."""

    rng = _as_rng(rng)
    height, width = config.image_shape
    atom_count = (
        int(atom_count)
        if atom_count is not None
        else int(rng.integers(config.min_atoms, config.max_atoms + 1))
    )
    min_separation = (
        float(min_separation)
        if min_separation is not None
        else float(config.min_separation)
    )
    margin = (
        float(boundary_margin)
        if boundary_margin is not None
        else max(min_separation, _support_margin_for_sigma(config.sigma_range))
    )

    y_min, y_max, x_min, x_max = _valid_sampling_bounds((height, width), margin)
    cell_size = max(float(min_separation) / np.sqrt(2.0), 1e-6)
    grid_shape = (
        max(1, int(np.ceil((y_max - y_min) / cell_size))),
        max(1, int(np.ceil((x_max - x_min) / cell_size))),
    )
    grid = np.full(grid_shape, -1, dtype=np.int32)
    coordinates: list[np.ndarray] = []
    active_indices: list[int] = []
    first = np.array(
        [
            rng.uniform(y_min, y_max),
            rng.uniform(x_min, x_max),
        ],
        dtype=np.float32,
    )
    _store_poisson_coordinate(first, coordinates, active_indices, grid, y_min, x_min, cell_size)

    candidates_per_active_point = 30
    while active_indices and len(coordinates) < atom_count:
        active_list_index = int(rng.integers(0, len(active_indices)))
        source = coordinates[active_indices[active_list_index]]
        accepted = False
        angles = rng.uniform(0.0, 2.0 * np.pi, size=candidates_per_active_point)
        radii = min_separation * np.sqrt(rng.uniform(1.0, 4.0, size=candidates_per_active_point))
        for angle, radius in zip(angles, radii):
            candidate = source + np.array(
                [radius * np.sin(angle), radius * np.cos(angle)],
                dtype=np.float32,
            )
            if not (y_min <= float(candidate[0]) < y_max and x_min <= float(candidate[1]) < x_max):
                continue
            if not _is_poisson_candidate_valid(
                candidate,
                coordinates,
                grid,
                y_min,
                x_min,
                cell_size,
                min_separation,
            ):
                continue
            _store_poisson_coordinate(candidate, coordinates, active_indices, grid, y_min, x_min, cell_size)
            accepted = True
            if len(coordinates) >= atom_count:
                break
        if not accepted:
            active_indices.pop(active_list_index)

    if not coordinates:
        raise RuntimeError("Failed to sample any atom coordinates.")
    return np.stack(coordinates, axis=0).astype(np.float32)


def sample_random_atom_points(
    config: RandomAtomImageConfig,
    rng: Optional[np.random.Generator] = None,
) -> PointCloud:
    rng = _as_rng(rng)
    min_separation = (
        _sample_scalar(rng, config.min_separation_range)
        if config.min_separation_range is not None
        else float(config.min_separation)
    )
    desired_visible_count = int(rng.integers(config.min_atoms, config.max_atoms + 1))
    padding = max(0, int(config.edge_padding))

    if padding <= 0:
        coordinates = sample_atom_coordinates(
            config, rng, atom_count=desired_visible_count, min_separation=min_separation
        )
        subpixel_jitter = 0.0
        target_coordinates = coordinates
    else:
        expanded_shape = _expanded_shape(config.image_shape, padding)
        expanded_area = float(expanded_shape[0] * expanded_shape[1])
        image_area = float(config.image_shape[0] * config.image_shape[1])
        total_atom_count = max(
            desired_visible_count,
            int(np.ceil(desired_visible_count * expanded_area / max(image_area, 1.0))),
        )
        sampling_config = RandomAtomImageConfig(
            **{**asdict(config), "image_shape": expanded_shape}
        )
        boundary_margin = max(0.0, _support_margin_for_sigma(config.sigma_range) - padding)
        best_coordinates = np.zeros((0, 2), dtype=np.float32)
        best_target = np.zeros((0, 2), dtype=np.float32)
        best_score: Optional[Tuple[int, int]] = None
        best_subpixel_jitter = 0.0

        for _ in range(2):
            padded = sample_atom_coordinates(
                sampling_config,
                rng,
                atom_count=total_atom_count,
                min_separation=min_separation,
                boundary_margin=boundary_margin,
            )
            coordinates = _shift_coordinates(padded, (-padding, -padding))
            subpixel_jitter = 0.0
            coordinates, target_coordinates = _limit_visible_coordinates(
                coordinates,
                config.image_shape,
                config.max_atoms,
                rng,
            )
            score = (
                0 if config.min_atoms <= len(target_coordinates) <= config.max_atoms else 1,
                abs(len(target_coordinates) - desired_visible_count),
            )
            if best_score is None or score < best_score:
                best_coordinates, best_target, best_score = coordinates, target_coordinates, score
                best_subpixel_jitter = subpixel_jitter
            if score[0] == 0:
                break
        coordinates, target_coordinates = best_coordinates, best_target
        subpixel_jitter = best_subpixel_jitter

    return PointCloud(
        coordinates=coordinates,
        target_coordinates=target_coordinates,
        metadata={
            "image_type": "random_atom_image",
            "sampled_min_separation": float(min_separation),
            "sampled_subpixel_jitter": float(subpixel_jitter),
            "visible_atom_count": int(len(target_coordinates)),
            "rendered_atom_count": int(len(coordinates)),
        },
    )


def _rotation_matrix_xy(theta_degrees: float) -> np.ndarray:
    theta = np.deg2rad(theta_degrees)
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )


def sample_lattice_points(
    config: PeriodicLatticeConfig,
    rng: Optional[np.random.Generator] = None,
) -> PointCloud:
    rng = _as_rng(rng)
    padding = max(0, int(config.edge_padding))
    render_height, render_width = _expanded_shape(config.image_shape, padding)
    margin = max(0.0, _support_margin_for_sigma(config.sigma_range) - padding)

    for _ in range(64):
        spacing = _sample_scalar(rng, config.lattice_spacing_range)
        angle = _sample_scalar(rng, config.rotation_range)
        jitter_std = _sample_scalar(rng, config.jitter_std_range)
        vacancy_fraction = _sample_scalar(rng, config.vacancy_fraction_range)

        if config.lattice_type == "cubic":
            e1 = np.array([spacing, 0.0], dtype=np.float32)
            e2 = np.array([0.0, spacing], dtype=np.float32)
        elif config.lattice_type == "hexagonal":
            e1 = np.array([spacing, 0.0], dtype=np.float32)
            e2 = np.array([0.5 * spacing, (np.sqrt(3.0) / 2.0) * spacing], dtype=np.float32)
        else:
            raise ValueError("lattice_type must be 'cubic' or 'hexagonal'.")

        rotation = _rotation_matrix_xy(angle)
        e1, e2 = rotation @ e1, rotation @ e2
        center_xy = np.array([render_width / 2.0, render_height / 2.0], dtype=np.float32)
        offset_xy = rng.uniform(0.0, 1.0) * e1 + rng.uniform(0.0, 1.0) * e2 - 0.5 * (e1 + e2)
        index_limit = max(6, int(np.ceil((np.hypot(render_height, render_width) + 4.0 * spacing) / spacing)))

        points_xy: List[np.ndarray] = []
        for i in range(-index_limit, index_limit + 1):
            for j in range(-index_limit, index_limit + 1):
                point_xy = center_xy + offset_xy + i * e1 + j * e2
                if jitter_std > 0:
                    point_xy = point_xy + rng.normal(0.0, jitter_std, size=2).astype(np.float32)
                x, y = float(point_xy[0]), float(point_xy[1])
                if margin <= x < render_width - margin and margin <= y < render_height - margin:
                    points_xy.append(np.array([x, y], dtype=np.float32))

        if not points_xy:
            continue
        coordinates_xy = np.stack(points_xy, axis=0)
        if vacancy_fraction > 0:
            coordinates_xy = coordinates_xy[rng.random(len(coordinates_xy)) >= vacancy_fraction]
        coordinates = _shift_coordinates(coordinates_xy[:, [1, 0]], (-padding, -padding))
        target_coordinates = coordinates[_in_frame_mask(coordinates, config.image_shape)]
        if int(len(target_coordinates)) >= config.min_atoms:
            return PointCloud(
                coordinates=coordinates.astype(np.float32),
                target_coordinates=target_coordinates.astype(np.float32),
                metadata={
                    "image_type": "periodic",
                    "lattice_type": config.lattice_type,
                    "visible_atom_count": int(len(target_coordinates)),
                    "rendered_atom_count": int(len(coordinates)),
                },
            )

    raise RuntimeError("Failed to sample a periodic lattice with enough atoms.")


def _require_ase():
    try:
        from ase import Atoms
    except ImportError as exc:
        raise ImportError(
            "ASE is required for projected-structure datasets. Install the 'ase' package."
        ) from exc
    return Atoms


def build_ase_structure_unit_cell(structure_name: str):
    Atoms = _require_ase()
    structure_name = structure_name.lower()

    if structure_name == "graphene":
        a = 2.46
        return Atoms(
            "C2",
            scaled_positions=[(0.0, 0.0, 0.5), (1.0 / 3.0, 2.0 / 3.0, 0.5)],
            cell=[(a, 0.0, 0.0), (0.5 * a, np.sqrt(3.0) * 0.5 * a, 0.0), (0.0, 0.0, 18.0)],
            pbc=(True, True, False),
        )
    if structure_name in {"ws2", "ws2_mx2", "ws2-ase", "ws2_ase"}:
        try:
            from ase.build import mx2
        except ImportError as exc:
            raise ImportError(
                "ASE's mx2 builder is required for the WS2 structure."
            ) from exc
        atoms = mx2("WS2")
        atoms.cell[2, 2] = 20.0
        atoms.pbc = (True, True, False)
        return atoms
    if structure_name in {"sto", "srtio3"}:
        a = 3.905
        return Atoms(
            "SrTiO3",
            scaled_positions=[
                (0.0, 0.0, 0.0),
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.0),
                (0.5, 0.0, 0.5),
                (0.0, 0.5, 0.5),
            ],
            cell=[(a, 0.0, 0.0), (0.0, a, 0.0), (0.0, 0.0, a)],
            pbc=(True, True, True),
        )
    raise ValueError("Unsupported ASE structure. Expected graphene, ws2, ws2_mx2, or sto.")


def _merge_projected_columns(
    xy: np.ndarray,
    atomic_numbers: np.ndarray,
    symbols: np.ndarray,
    *,
    species_intensity_power: float,
    tolerance_angstrom: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tolerance = max(float(tolerance_angstrom), 1e-6)
    keys = np.round(np.asarray(xy, dtype=np.float32) / tolerance).astype(np.int64)
    groups: Dict[tuple[int, int], List[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault((int(key[0]), int(key[1])), []).append(index)

    merged_xy = []
    merged_numbers = []
    merged_symbols = []
    merged_weights = []
    weights = atomic_numbers.astype(np.float32) ** float(species_intensity_power)
    for indices in groups.values():
        index_array = np.asarray(indices, dtype=np.int64)
        group_weights = weights[index_array]
        merged_xy.append(np.average(xy[index_array], axis=0, weights=group_weights))
        merged_numbers.append(int(np.max(atomic_numbers[index_array])))
        merged_symbols.append("/".join(sorted(set(symbols[index_array].tolist()))))
        merged_weights.append(float(np.sum(group_weights)))

    return (
        np.asarray(merged_xy, dtype=np.float32),
        np.asarray(merged_numbers, dtype=np.int32),
        np.asarray(merged_symbols),
        np.asarray(merged_weights, dtype=np.float32),
    )


def sample_structure_points(
    config: AseStructureProjectionConfig,
    rng: Optional[np.random.Generator] = None,
) -> PointCloud:
    rng = _as_rng(rng)
    unit = build_ase_structure_unit_cell(config.structure_name)
    padding = max(0, int(config.edge_padding))
    render_height, render_width = _expanded_shape(config.image_shape, padding)
    pixel_size = float(config.pixel_size_angstrom)
    field_diag = float(np.hypot(render_width * pixel_size, render_height * pixel_size))

    cell = np.asarray(unit.cell.array, dtype=np.float32)
    ax = max(float(np.linalg.norm(cell[0, :2])), 1e-6)
    ay = max(float(np.linalg.norm(cell[1, :2])), 1e-6)
    repeat_x = max(3, int(np.ceil(field_diag / ax)) + 3)
    repeat_y = max(3, int(np.ceil(field_diag / ay)) + 3)
    repeat_z = max(1, int(config.repeat_thickness))
    supercell = unit.repeat((repeat_x, repeat_y, repeat_z))

    positions = np.asarray(supercell.get_positions(), dtype=np.float32)
    atomic_numbers = np.asarray(supercell.get_atomic_numbers(), dtype=np.int32)
    symbols = np.asarray(supercell.get_chemical_symbols())
    positions[:, 0] -= float(positions[:, 0].mean())
    positions[:, 1] -= float(positions[:, 1].mean())

    theta = _sample_scalar(rng, config.rotation_range)
    xy = positions[:, :2] @ _rotation_matrix_xy(theta).T
    if config.merge_projected_columns:
        xy, atomic_numbers, symbols, z_weight = _merge_projected_columns(
            xy,
            atomic_numbers,
            symbols,
            species_intensity_power=config.species_intensity_power,
            tolerance_angstrom=config.column_merge_tolerance_angstrom,
        )
    else:
        z_weight = atomic_numbers.astype(np.float32) ** float(config.species_intensity_power)

    jitter_std = _sample_scalar(rng, config.position_jitter_std_range)
    if jitter_std > 0:
        xy = xy + rng.normal(0.0, jitter_std, size=xy.shape).astype(np.float32)
    xy[:, 0] += float(rng.uniform(-0.5 * ax, 0.5 * ax))
    xy[:, 1] += float(rng.uniform(-0.5 * ay, 0.5 * ay))

    render_coordinates = np.stack(
        [xy[:, 1] / pixel_size + render_height / 2.0, xy[:, 0] / pixel_size + render_width / 2.0],
        axis=1,
    ).astype(np.float32)
    margin = max(0.0, _support_margin_for_sigma(config.sigma_range) - padding)
    keep = (
        (render_coordinates[:, 0] >= margin)
        & (render_coordinates[:, 0] < render_height - margin)
        & (render_coordinates[:, 1] >= margin)
        & (render_coordinates[:, 1] < render_width - margin)
    )
    coordinates = _shift_coordinates(render_coordinates[keep], (-padding, -padding))
    atomic_numbers = atomic_numbers[keep]
    symbols = symbols[keep]
    z_weight = z_weight[keep]
    visible = _in_frame_mask(coordinates, config.image_shape)
    if int(visible.sum()) == 0:
        raise RuntimeError(f"{config.structure_name} projection produced no in-frame atoms.")

    z_min, z_max = float(z_weight.min()), float(z_weight.max())
    normalized = np.full_like(z_weight, 0.5) if z_max <= z_min else (z_weight - z_min) / (z_max - z_min)
    low, high = config.intensity_range
    intensities = (float(low) + normalized * float(high - low)).astype(np.float32)
    sigmas = rng.uniform(config.sigma_range[0], config.sigma_range[1], size=len(coordinates)).astype(np.float32)

    return PointCloud(
        coordinates=coordinates.astype(np.float32),
        target_coordinates=coordinates[visible].astype(np.float32),
        intensities=intensities,
        sigmas=sigmas,
        metadata={
            "image_type": "ase_projected",
            "structure_name": config.structure_name,
            "atomic_numbers": atomic_numbers.astype(np.int32),
            "symbols": symbols.tolist(),
            "rotation_degrees": float(theta),
            "position_jitter_std": float(jitter_std),
            "repeat_xy": [int(repeat_x), int(repeat_y), int(repeat_z)],
            "visible_atom_count": int(visible.sum()),
            "rendered_atom_count": int(len(coordinates)),
        },
    )


def point_cloud_from_atoms(
    atoms: Any,
    image_shape: Tuple[int, int],
    *,
    species_intensity_power: float = 1.45,
    intensity_range: Tuple[float, float] = (0.3, 1.0),
    atom_sigma_range: Optional[Tuple[float, float]] = None,
    coordinate_scale: float = 1.0,
) -> PointCloud:
    """Convert an ASE Atoms object with x/y positions into renderer-ready points."""

    positions = np.asarray(atoms.get_positions(), dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("Expected atoms positions with at least x/y columns.")

    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.float32)
    if len(positions) != len(atomic_numbers):
        raise ValueError("Atom positions and atomic numbers have different lengths.")

    coordinates = (positions[:, [1, 0]] * float(coordinate_scale)).astype(np.float32)
    visible = _in_frame_mask(coordinates, image_shape)

    z_weights = atomic_numbers**float(species_intensity_power)
    weight_min, weight_max = float(z_weights.min()), float(z_weights.max())
    if weight_max > weight_min:
        normalized = (z_weights - weight_min) / (weight_max - weight_min)
    else:
        normalized = np.ones_like(z_weights, dtype=np.float32)

    intensity_low, intensity_high = intensity_range
    intensities = (
        float(intensity_low) + normalized * float(intensity_high - intensity_low)
    ).astype(np.float32)

    sigmas = None
    if atom_sigma_range is not None:
        sigma_low, sigma_high = atom_sigma_range
        sigmas = (
            float(sigma_low) + normalized * float(sigma_high - sigma_low)
        ).astype(np.float32)

    symbols = (
        atoms.get_chemical_symbols()
        if hasattr(atoms, "get_chemical_symbols")
        else [str(int(value)) for value in atomic_numbers]
    )
    return PointCloud(
        coordinates=coordinates,
        target_coordinates=coordinates[visible].astype(np.float32),
        intensities=intensities,
        sigmas=sigmas,
        metadata={
            "image_type": "ase_atoms",
            "atomic_numbers": atomic_numbers.astype(np.int32),
            "symbols": list(symbols),
            "visible_atom_count": int(visible.sum()),
            "rendered_atom_count": int(len(coordinates)),
        },
    )


def generate_atoms_image(
    atoms: Any,
    config: ImageFormationConfig,
    rng: Optional[np.random.Generator] = None,
    *,
    species_intensity_power: float = 1.45,
    atom_sigma_range: Optional[Tuple[float, float]] = None,
    coordinate_scale: float = 1.0,
) -> Dict[str, Any]:
    """Render a user-constructed ASE Atoms object through the shared image model."""

    points = point_cloud_from_atoms(
        atoms,
        config.image_shape,
        species_intensity_power=species_intensity_power,
        intensity_range=config.intensity_range,
        atom_sigma_range=atom_sigma_range,
        coordinate_scale=coordinate_scale,
    )
    return render_atom_image(
        points.coordinates,
        config,
        rng,
        intensities=points.intensities,
        sigmas=points.sigmas,
        target_coordinates=points.target_coordinates,
        metadata=points.metadata,
    )


def point_cloud_from_config(
    config: ImageFormationConfig,
    rng: Optional[np.random.Generator] = None,
) -> PointCloud:
    if isinstance(config, RandomAtomImageConfig):
        return sample_random_atom_points(config, rng)
    if isinstance(config, PeriodicLatticeConfig):
        return sample_lattice_points(config, rng)
    if isinstance(config, AseStructureProjectionConfig):
        return sample_structure_points(config, rng)
    raise TypeError(f"Unsupported synthetic config: {type(config).__name__}")


def render_atom_image(
    coordinates: np.ndarray,
    config: ImageFormationConfig,
    rng: Optional[np.random.Generator] = None,
    *,
    intensities: Optional[np.ndarray] = None,
    sigmas: Optional[np.ndarray] = None,
    target_coordinates: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render coordinates and brightnesses into an image/target pair."""

    rng = _as_rng(rng)
    coordinates = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
    target_coordinates = (
        coordinates
        if target_coordinates is None
        else np.asarray(target_coordinates, dtype=np.float32).reshape(-1, 2)
    )
    intensities = (
        rng.uniform(*config.intensity_range, size=len(coordinates)).astype(np.float32)
        if intensities is None
        else np.asarray(intensities, dtype=np.float32)
    )
    sigmas = (
        rng.uniform(*config.sigma_range, size=len(coordinates)).astype(np.float32)
        if sigmas is None
        else np.asarray(sigmas, dtype=np.float32)
    )

    padding = max(0, int(config.edge_padding))
    render_shape = _expanded_shape(config.image_shape, padding)
    image = np.zeros(render_shape, dtype=np.float32)
    target = np.zeros(render_shape, dtype=np.float32)
    render_coordinates = _shift_coordinates(coordinates, (padding, padding))
    for coord, amplitude, sigma in zip(render_coordinates, intensities, sigmas):
        _stamp_gaussian(image, coord, float(sigma), amplitude=float(amplitude), mode="sum")
    for coord in _shift_coordinates(target_coordinates, (padding, padding)):
        _stamp_gaussian(target, coord, float(config.target_sigma), amplitude=1.0, mode="max")

    image += _sample_scalar(rng, config.background_range)
    yy, xx = _make_coordinate_grids(render_shape)
    image += _sample_scalar(rng, config.gradient_range) * yy / max(render_shape[0] - 1, 1)
    image += _sample_scalar(rng, config.gradient_range) * xx / max(render_shape[1] - 1, 1)

    inhomogeneous_strength = _sample_scalar(rng, config.inhomogeneous_background_range)
    if inhomogeneous_strength > 0:
        sigma_fraction = _sample_scalar(
            rng,
            config.inhomogeneous_background_sigma_fraction_range,
        )
        image += inhomogeneous_strength * _smooth_unit_field(rng, render_shape, sigma_fraction)

    low_freq_strength = _sample_scalar(rng, config.low_frequency_noise_range)
    if low_freq_strength > 0:
        low_freq_noise = rng.normal(0.0, low_freq_strength, size=render_shape).astype(np.float32)
        smooth_sigma = max(render_shape) * _sample_scalar(rng, config.low_frequency_sigma_fraction_range)
        image += gaussian_filter(low_freq_noise, sigma=smooth_sigma, mode="reflect")

    blur_sigma = _sample_scalar(rng, config.blur_sigma_range)
    if blur_sigma > 0:
        image = gaussian_filter(image, sigma=blur_sigma, mode="reflect")
    if config.normalize_input:
        image = _normalize(image)

    if padding > 0:
        crop_y = slice(padding, padding + config.image_shape[0])
        crop_x = slice(padding, padding + config.image_shape[1])
        image = image[crop_y, crop_x]
        target = target[crop_y, crop_x]

    poisson_ready = np.clip(image, 0.0, None)
    poisson_ready = poisson_ready / max(float(poisson_ready.max()), 1e-6)
    count_scale = _sample_count_scale(rng, config, poisson_ready.shape)
    if count_scale <= 0:
        count_map = np.zeros_like(poisson_ready, dtype=np.float32)
        image = np.zeros_like(poisson_ready, dtype=np.float32)
    else:
        count_map = rng.poisson(poisson_ready * count_scale).astype(np.float32)
        image = count_map / float(count_scale)

    read_noise = rng.normal(
        0.0,
        _sample_scalar(rng, config.read_noise_std_range),
        size=image.shape,
    ).astype(np.float32)
    image += read_noise
    image = _normalize(image)
    if config.clamp_target:
        target = np.clip(target, 0.0, 1.0)

    image_record = {
        "image": image.astype(np.float32),
        "target": target.astype(np.float32),
        "coordinates": target_coordinates.astype(np.float32),
        "rendered_coordinates": coordinates.astype(np.float32),
        "intensities": intensities.astype(np.float32),
        "sigmas": sigmas.astype(np.float32),
        "count_map": count_map.astype(np.float32),
        "total_counts": int(count_map.sum()),
        "count_scale": int(count_scale),
        "config": asdict(config),
    }
    if metadata:
        image_record.update(metadata)
    return image_record


def generate_atom_image(
    config: Optional[ImageFormationConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Generate points from the config, then render them through the shared image model."""

    config = config or RandomAtomImageConfig()
    rng = _as_rng(rng)
    points = point_cloud_from_config(config, rng)
    return render_atom_image(
        points.coordinates,
        config,
        rng,
        intensities=points.intensities,
        sigmas=points.sigmas,
        target_coordinates=points.target_coordinates,
        metadata=points.metadata,
    )


class GeneratedAtomImageDataset(Dataset):
    """Generate atom image/target pairs on demand."""

    def __init__(
        self,
        num_samples: int,
        config: Optional[ImageFormationConfig] = None,
        seed: Optional[int] = None,
        return_metadata: bool = False,
    ) -> None:
        self.num_samples = int(num_samples)
        self.config = config or RandomAtomImageConfig()
        self.seed = seed
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return self.num_samples

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + idx)

    def __getitem__(self, idx: int):
        image_record = generate_atom_image(self.config, self._rng_for_index(idx))
        image = torch.from_numpy(image_record["image"]).unsqueeze(0)
        target = torch.from_numpy(image_record["target"]).unsqueeze(0)
        if not self.return_metadata:
            return image, target

        metadata = {
            key: value
            for key, value in image_record.items()
            if key not in {"image", "target", "count_map", "config"}
        }
        return image, target, metadata


class SavedAtomImageDataset(Dataset):
    """Load saved atom image/target NPZ samples."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.paths = sorted(self.directory.glob('*.npz'))
        if not self.paths:
            raise FileNotFoundError(f'No NPZ samples found in {self.directory}')

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.paths[index], allow_pickle=False) as sample:
            image = torch.from_numpy(sample['image'].astype(np.float32)).unsqueeze(0)
            target = torch.from_numpy(sample['target'].astype(np.float32)).unsqueeze(0)
        return image, target


def metadata_collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    metadata = [item[2] for item in batch]
    return images, targets, metadata


def _resolve_worker_count(num_workers: int, num_samples: int) -> int:
    if num_samples <= 1:
        return 1
    if num_workers == 0:
        return max(1, min(os.cpu_count() or 1, num_samples))
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0. Use 0 for all available CPUs.")
    return max(1, min(int(num_workers), num_samples))


def _generate_and_save_one_sample(args: tuple[Path, int, ImageFormationConfig, int, str]) -> Path:
    output_dir, idx, config, seed, prefix = args
    image_record = generate_atom_image(config, np.random.default_rng(seed + idx))
    file_path = output_dir / f"{prefix}_{idx:05d}.npz"
    np.savez_compressed(
        file_path,
        image=image_record["image"],
        target=image_record["target"],
        coordinates=image_record["coordinates"],
        intensities=image_record["intensities"],
        sigmas=image_record["sigmas"],
        count_map=image_record["count_map"],
        total_counts=image_record["total_counts"],
        count_scale=image_record["count_scale"],
        config_json=json.dumps(image_record["config"]),
    )
    return file_path


def generate_and_save_samples(
    output_dir: str | Path,
    num_samples: int,
    config: Optional[ImageFormationConfig] = None,
    seed: int = 0,
    prefix: str = "image",
    num_workers: int = 0,
) -> List[Path]:
    """Generate atom image samples and save one NPZ file per sample."""

    config = config or RandomAtomImageConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = int(num_samples)
    worker_count = _resolve_worker_count(int(num_workers), num_samples)
    tasks = [(output_dir, idx, config, int(seed), prefix) for idx in range(num_samples)]

    if worker_count == 1:
        return [_generate_and_save_one_sample(task) for task in tasks]

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_generate_and_save_one_sample, tasks))


def generate_and_save_dataset_splits(
    output_dir: str | Path,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    config: Optional[ImageFormationConfig] = None,
    seed: int = 0,
    prefix: str = "image",
    num_workers: int = 0,
) -> Dict[str, List[Path]]:
    config = config or RandomAtomImageConfig()
    split_specs = [
        ("train", int(train_samples), seed),
        ("val", int(val_samples), seed + 1_000_000),
        ("test", int(test_samples), seed + 2_000_000),
    ]
    return {
        split: generate_and_save_samples(
            Path(output_dir) / split,
            count,
            config,
            split_seed,
            prefix,
            num_workers=num_workers,
        )
        for split, count, split_seed in split_specs
    }
