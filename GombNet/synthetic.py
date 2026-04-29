from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset

RESAMPLE_NEAREST = (
    Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
)


@dataclass(frozen=True)
class RandomGaussianConfig:
    """
    Configuration for synthetic atom-like Gaussian data.

    Coordinates are represented in pixel units as ``(y, x)``.
    """

    image_shape: Tuple[int, int] = (128, 128)
    min_atoms: int = 8
    max_atoms: int = 36
    min_separation: float = 4.0
    min_separation_range: Optional[Tuple[float, float]] = None
    sigma_range: Tuple[float, float] = (0.7, 1.8)
    intensity_range: Tuple[float, float] = (0.3, 1.0)
    # The supervision target is a normalized Gaussian peak with amplitude 1
    # at every atom center, independent of image brightness.
    target_sigma: float = 0.9
    background_range: Tuple[float, float] = (0.02, 0.15)
    gradient_range: Tuple[float, float] = (-0.08, 0.08)
    low_frequency_noise_range: Tuple[float, float] = (0.0, 0.12)
    low_frequency_sigma_fraction_range: Tuple[float, float] = (0.04, 0.12)
    read_noise_std_range: Tuple[float, float] = (0.005, 0.04)
    poisson_counts_range: Tuple[float, float] = (3_000.0, 20_000.0)
    blur_sigma_range: Tuple[float, float] = (0.0, 1.0)
    edge_padding: int = 0
    normalize_input: bool = True
    clamp_target: bool = True


@dataclass(frozen=True)
class PeriodicLatticeConfig:
    """
    Configuration for periodic lattice evaluation data.

    `cubic` is represented here as a square 2D lattice, which is the relevant
    projected geometry for this localization benchmark.
    """

    image_shape: Tuple[int, int] = (128, 128)
    lattice_type: str = "hexagonal"
    lattice_spacing_range: Tuple[float, float] = (8.0, 12.0)
    rotation_range: Tuple[float, float] = (0.0, 180.0)
    jitter_std_range: Tuple[float, float] = (0.0, 0.15)
    vacancy_fraction_range: Tuple[float, float] = (0.0, 0.02)
    sigma_range: Tuple[float, float] = (0.7, 1.8)
    intensity_range: Tuple[float, float] = (0.3, 1.0)
    target_sigma: float = 0.9
    background_range: Tuple[float, float] = (0.02, 0.15)
    gradient_range: Tuple[float, float] = (-0.08, 0.08)
    low_frequency_noise_range: Tuple[float, float] = (0.0, 0.12)
    low_frequency_sigma_fraction_range: Tuple[float, float] = (0.04, 0.12)
    read_noise_std_range: Tuple[float, float] = (0.005, 0.04)
    poisson_counts_range: Tuple[float, float] = (3_000.0, 20_000.0)
    blur_sigma_range: Tuple[float, float] = (0.0, 1.0)
    edge_padding: int = 0
    normalize_input: bool = True
    clamp_target: bool = True
    min_margin: float = 6.0
    min_atoms: int = 24


@dataclass(frozen=True)
class AseStructureProjectionConfig:
    """
    Configuration for evaluation images generated from projected ASE structures.

    The structure is built as a 3D `ase.Atoms` object, repeated into a supercell,
    projected onto the image plane, and rendered into a STEM-like image.
    """

    image_shape: Tuple[int, int] = (256, 256)
    structure_name: str = "graphene"
    pixel_size_angstrom: float = 0.12
    rotation_range: Tuple[float, float] = (0.0, 180.0)
    position_jitter_std_range: Tuple[float, float] = (0.0, 0.08)
    sigma_range: Tuple[float, float] = (0.7, 1.8)
    intensity_range: Tuple[float, float] = (0.3, 1.0)
    target_sigma: float = 0.9
    background_range: Tuple[float, float] = (0.02, 0.15)
    gradient_range: Tuple[float, float] = (-0.08, 0.08)
    low_frequency_noise_range: Tuple[float, float] = (0.0, 0.12)
    low_frequency_sigma_fraction_range: Tuple[float, float] = (0.04, 0.12)
    read_noise_std_range: Tuple[float, float] = (0.005, 0.04)
    poisson_counts_range: Tuple[float, float] = (3_000.0, 20_000.0)
    blur_sigma_range: Tuple[float, float] = (0.0, 1.0)
    edge_padding: int = 0
    normalize_input: bool = True
    clamp_target: bool = True
    min_margin: float = 6.0
    species_intensity_power: float = 1.6
    repeat_thickness: int = 1


def _as_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _require_ase():
    try:
        from ase import Atoms
    except ImportError as exc:
        raise ImportError(
            "ASE is required for projected-structure datasets. Install the 'ase' package."
        ) from exc
    return Atoms


def _sample_scalar(
    rng: np.random.Generator, value_range: Sequence[float]
) -> float:
    low, high = float(value_range[0]), float(value_range[1])
    if high < low:
        raise ValueError(f"Expected an ascending range, got {value_range!r}")
    return float(rng.uniform(low, high))


def _make_coordinate_grids(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    y = np.arange(shape[0], dtype=np.float32)
    x = np.arange(shape[1], dtype=np.float32)
    return np.meshgrid(y, x, indexing="ij")


def _support_margin_for_sigma(sigma_range: Sequence[float]) -> float:
    return 4.0 * float(sigma_range[1])


def _edge_padding(
    config: RandomGaussianConfig | PeriodicLatticeConfig | AseStructureProjectionConfig,
) -> int:
    return max(0, int(getattr(config, "edge_padding", 0)))


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


def _stamp_gaussian(
    image: np.ndarray,
    center_yx: Sequence[float],
    sigma: float,
    amplitude: float = 1.0,
    mode: str = "sum",
    truncate: float = 4.0,
) -> None:
    """Render one sub-pixel Gaussian into an image."""

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


def sample_atom_coordinates(
    config: RandomGaussianConfig,
    rng: Optional[np.random.Generator] = None,
    atom_count: Optional[int] = None,
    min_separation: Optional[float] = None,
    boundary_margin: Optional[float] = None,
) -> np.ndarray:
    """Sample random atom centers with a minimum separation."""

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

    coordinates = np.empty((atom_count, 2), dtype=np.float32)
    accepted = 0
    max_attempts = atom_count * 200

    for _ in range(max_attempts):
        if accepted >= atom_count:
            break

        candidate = np.array(
            [
                rng.uniform(margin, height - margin),
                rng.uniform(margin, width - margin),
            ],
            dtype=np.float32,
        )

        if accepted == 0:
            coordinates[accepted] = candidate
            accepted += 1
            continue

        distances = np.linalg.norm(coordinates[:accepted] - candidate, axis=1)
        if np.all(distances >= min_separation):
            coordinates[accepted] = candidate
            accepted += 1

    if accepted == 0:
        raise RuntimeError("Failed to sample any atom coordinates.")

    return coordinates[:accepted].copy()


def _render_sample_from_coordinates(
    coordinates: np.ndarray,
    config: RandomGaussianConfig | PeriodicLatticeConfig | AseStructureProjectionConfig,
    rng: np.random.Generator,
    metadata: Optional[Dict[str, Any]] = None,
    intensities: Optional[np.ndarray] = None,
    sigmas: Optional[np.ndarray] = None,
    target_coordinates: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    target_coordinates = coordinates if target_coordinates is None else target_coordinates
    padding = _edge_padding(config)
    render_shape = _expanded_shape(config.image_shape, padding)
    image = np.zeros(render_shape, dtype=np.float32)
    target = np.zeros(render_shape, dtype=np.float32)
    render_coordinates = _shift_coordinates(coordinates, (padding, padding))
    render_target_coordinates = _shift_coordinates(target_coordinates, (padding, padding))

    if intensities is None:
        intensities = rng.uniform(
            config.intensity_range[0],
            config.intensity_range[1],
            size=len(coordinates),
        ).astype(np.float32)
    else:
        intensities = np.asarray(intensities, dtype=np.float32)

    if sigmas is None:
        sigmas = rng.uniform(
            config.sigma_range[0],
            config.sigma_range[1],
            size=len(coordinates),
        ).astype(np.float32)
    else:
        sigmas = np.asarray(sigmas, dtype=np.float32)

    for coord, amplitude, sigma in zip(render_coordinates, intensities, sigmas):
        _stamp_gaussian(image, coord, float(sigma), amplitude=float(amplitude), mode="sum")

    for coord in render_target_coordinates:
        _stamp_gaussian(
            target,
            coord,
            float(config.target_sigma),
            amplitude=1.0,
            mode="max",
        )

    background = _sample_scalar(rng, config.background_range)
    image += background

    yy, xx = _make_coordinate_grids(render_shape)
    height, width = render_shape
    yy = yy / max(height - 1, 1)
    xx = xx / max(width - 1, 1)
    gradient_y = _sample_scalar(rng, config.gradient_range)
    gradient_x = _sample_scalar(rng, config.gradient_range)
    image += gradient_y * yy + gradient_x * xx

    low_freq_strength = _sample_scalar(rng, config.low_frequency_noise_range)
    if low_freq_strength > 0:
        low_freq_noise = rng.normal(0.0, low_freq_strength, size=render_shape).astype(
            np.float32
        )
        sigma_fraction = _sample_scalar(
            rng, config.low_frequency_sigma_fraction_range
        )
        smooth_sigma = max(render_shape) * sigma_fraction
        low_freq_noise = gaussian_filter(low_freq_noise, sigma=smooth_sigma, mode="reflect")
        image += low_freq_noise

    blur_sigma = _sample_scalar(rng, config.blur_sigma_range)
    if blur_sigma > 0:
        image = gaussian_filter(image, sigma=blur_sigma, mode="reflect")

    if config.normalize_input:
        image = image - image.min()
        peak = float(image.max())
        if peak > 0:
            image = image / peak

    poisson_counts = _sample_scalar(rng, config.poisson_counts_range)
    poisson_ready = np.clip(image, 0.0, None)
    poisson_ready = poisson_ready / max(float(poisson_ready.max()), 1e-6)
    image = rng.poisson(poisson_ready * poisson_counts).astype(np.float32) / poisson_counts

    read_noise = rng.normal(
        0.0,
        _sample_scalar(rng, config.read_noise_std_range),
        size=render_shape,
    ).astype(np.float32)
    image += read_noise

    image = image - image.min()
    peak = float(image.max())
    if peak > 0:
        image = image / peak

    if config.clamp_target:
        target = np.clip(target, 0.0, 1.0)

    if padding > 0:
        crop_y = slice(padding, padding + config.image_shape[0])
        crop_x = slice(padding, padding + config.image_shape[1])
        image = image[crop_y, crop_x]
        target = target[crop_y, crop_x]

    sample = {
        "image": image.astype(np.float32),
        "target": target.astype(np.float32),
        "coordinates": np.asarray(target_coordinates, dtype=np.float32),
        "intensities": intensities.astype(np.float32),
        "sigmas": sigmas.astype(np.float32),
        "config": asdict(config),
    }
    if metadata:
        sample.update(metadata)
    return sample


def _rotation_matrix_xy(theta_degrees: float) -> np.ndarray:
    theta = np.deg2rad(theta_degrees)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)


def sample_periodic_lattice_coordinates(
    config: PeriodicLatticeConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample a periodic 2D lattice in image coordinates.

    Returns coordinates in ``(y, x)`` order.
    """

    rng = _as_rng(rng)
    height, width = config.image_shape
    padding = _edge_padding(config)
    render_height, render_width = _expanded_shape(config.image_shape, padding)
    margin = max(0.0, _support_margin_for_sigma(config.sigma_range) - float(padding))

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
            raise ValueError(
                f"Unsupported lattice_type '{config.lattice_type}'. "
                "Expected 'cubic' or 'hexagonal'."
            )

        rotation = _rotation_matrix_xy(angle)
        e1 = rotation @ e1
        e2 = rotation @ e2

        center_xy = np.array([render_width / 2.0, render_height / 2.0], dtype=np.float32)
        offset_xy = (
            rng.uniform(0.0, 1.0) * e1
            + rng.uniform(0.0, 1.0) * e2
            - 0.5 * (e1 + e2)
        )

        extent = np.hypot(render_height, render_width) + 4.0 * spacing
        index_limit = max(6, int(np.ceil(extent / max(spacing, 1e-6))))

        points_xy: List[np.ndarray] = []
        for i in range(-index_limit, index_limit + 1):
            for j in range(-index_limit, index_limit + 1):
                point_xy = center_xy + offset_xy + i * e1 + j * e2
                if jitter_std > 0:
                    point_xy = point_xy + rng.normal(0.0, jitter_std, size=2).astype(np.float32)

                x, y = float(point_xy[0]), float(point_xy[1])
                if (
                    margin <= x < render_width - margin
                    and margin <= y < render_height - margin
                ):
                    points_xy.append(np.array([x, y], dtype=np.float32))

        if not points_xy:
            continue

        coordinates = np.stack(points_xy, axis=0)

        if vacancy_fraction > 0:
            keep_mask = rng.random(len(coordinates)) >= vacancy_fraction
            coordinates = coordinates[keep_mask]

        coordinates_yx = _shift_coordinates(coordinates[:, [1, 0]], (-padding, -padding))

        if int(_in_frame_mask(coordinates_yx, config.image_shape).sum()) < config.min_atoms:
            continue
        return coordinates_yx.astype(np.float32)

    raise RuntimeError("Failed to sample a periodic lattice with enough atoms.")


def generate_random_gaussian_sample(
    config: Optional[RandomGaussianConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Generate one STEM-like image with atom centers and a localization target.

    Returns a dict with ``image``, ``target``, ``coordinates``, ``intensities``,
    ``sigmas``, and ``config``.
    """

    config = config or RandomGaussianConfig()
    rng = _as_rng(rng)

    sampled_min_separation = (
        _sample_scalar(rng, config.min_separation_range)
        if config.min_separation_range is not None
        else float(config.min_separation)
    )
    desired_visible_count = int(rng.integers(config.min_atoms, config.max_atoms + 1))
    padding = _edge_padding(config)

    if padding > 0:
        expanded_shape = _expanded_shape(config.image_shape, padding)
        expanded_area = float(expanded_shape[0] * expanded_shape[1])
        final_area = float(config.image_shape[0] * config.image_shape[1])
        total_atom_count = max(
            desired_visible_count,
            int(np.ceil(desired_visible_count * expanded_area / max(final_area, 1.0))),
        )
        boundary_margin = max(
            0.0,
            _support_margin_for_sigma(config.sigma_range) - float(padding),
        )
        sampling_config = replace(config, image_shape=expanded_shape)
        best_coordinates = None
        best_visible_coordinates = None
        best_score = None

        for _ in range(24):
            padded_coordinates = sample_atom_coordinates(
                sampling_config,
                rng,
                atom_count=total_atom_count,
                min_separation=sampled_min_separation,
                boundary_margin=boundary_margin,
            )
            coordinates = _shift_coordinates(padded_coordinates, (-padding, -padding))
            visible_coordinates = coordinates[_in_frame_mask(coordinates, config.image_shape)]
            score = (
                0 if config.min_atoms <= len(visible_coordinates) <= config.max_atoms else 1,
                abs(len(visible_coordinates) - desired_visible_count),
            )
            if best_score is None or score < best_score:
                best_coordinates = coordinates
                best_visible_coordinates = visible_coordinates
                best_score = score
            if score == (0, 0):
                break

        coordinates = (
            best_coordinates
            if best_coordinates is not None
            else np.zeros((0, 2), dtype=np.float32)
        )
        target_coordinates = (
            best_visible_coordinates
            if best_visible_coordinates is not None
            else np.zeros((0, 2), dtype=np.float32)
        )
    else:
        coordinates = sample_atom_coordinates(
            config,
            rng,
            atom_count=desired_visible_count,
            min_separation=sampled_min_separation,
        )
        target_coordinates = coordinates

    return _render_sample_from_coordinates(
        coordinates=coordinates,
        config=config,
        rng=rng,
        target_coordinates=target_coordinates,
        metadata={
            "sample_type": "random",
            "sampled_min_separation": float(sampled_min_separation),
            "visible_atom_count": int(len(target_coordinates)),
            "rendered_atom_count": int(len(coordinates)),
        },
    )


def generate_periodic_lattice_sample(
    config: Optional[PeriodicLatticeConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Generate one STEM-like sample with a periodic cubic or hexagonal lattice.
    """

    config = config or PeriodicLatticeConfig()
    rng = _as_rng(rng)
    coordinates = sample_periodic_lattice_coordinates(config, rng)
    target_coordinates = coordinates[_in_frame_mask(coordinates, config.image_shape)]
    return _render_sample_from_coordinates(
        coordinates=coordinates,
        config=config,
        rng=rng,
        target_coordinates=target_coordinates,
        metadata={
            "sample_type": "periodic",
            "lattice_type": config.lattice_type,
            "visible_atom_count": int(len(target_coordinates)),
            "rendered_atom_count": int(len(coordinates)),
        },
    )


def build_ase_structure_unit_cell(structure_name: str):
    Atoms = _require_ase()
    structure_name = structure_name.lower()

    if structure_name == "graphene":
        a = 2.46
        return Atoms(
            "C2",
            scaled_positions=[
                (0.0, 0.0, 0.5),
                (1.0 / 3.0, 2.0 / 3.0, 0.5),
            ],
            cell=[
                (a, 0.0, 0.0),
                (0.5 * a, np.sqrt(3.0) * 0.5 * a, 0.0),
                (0.0, 0.0, 18.0),
            ],
            pbc=(True, True, False),
        )

    if structure_name == "ws2":
        a = 3.153
        return Atoms(
            "WS2",
            scaled_positions=[
                (0.0, 0.0, 0.5),
                (1.0 / 3.0, 2.0 / 3.0, 0.58),
                (2.0 / 3.0, 1.0 / 3.0, 0.42),
            ],
            cell=[
                (a, 0.0, 0.0),
                (0.5 * a, np.sqrt(3.0) * 0.5 * a, 0.0),
                (0.0, 0.0, 20.0),
            ],
            pbc=(True, True, False),
        )

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

    raise ValueError(
        f"Unsupported ASE structure '{structure_name}'. Expected graphene, ws2, or sto."
    )


def _rotate_xy(points_xy: np.ndarray, theta_degrees: float) -> np.ndarray:
    theta = np.deg2rad(float(theta_degrees))
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    return points_xy @ rotation.T


def sample_ase_projected_coordinates(
    config: AseStructureProjectionConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    rng = _as_rng(rng)
    unit = build_ase_structure_unit_cell(config.structure_name)
    height, width = config.image_shape
    padding = _edge_padding(config)
    render_height, render_width = _expanded_shape(config.image_shape, padding)
    pixel_size = float(config.pixel_size_angstrom)
    field_x = render_width * pixel_size
    field_y = render_height * pixel_size
    field_diag = float(np.hypot(field_x, field_y))

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
    xy = _rotate_xy(positions[:, :2], theta)

    jitter_std = _sample_scalar(rng, config.position_jitter_std_range)
    if jitter_std > 0:
        xy = xy + rng.normal(0.0, jitter_std, size=xy.shape).astype(np.float32)

    offset_x = rng.uniform(-0.5 * ax, 0.5 * ax)
    offset_y = rng.uniform(-0.5 * ay, 0.5 * ay)
    xy[:, 0] += float(offset_x)
    xy[:, 1] += float(offset_y)

    x_px = xy[:, 0] / pixel_size + render_width / 2.0
    y_px = xy[:, 1] / pixel_size + render_height / 2.0
    render_coordinates = np.stack([y_px, x_px], axis=1).astype(np.float32)

    margin = max(0.0, _support_margin_for_sigma(config.sigma_range) - float(padding))
    keep_mask = (
        (render_coordinates[:, 1] >= margin)
        & (render_coordinates[:, 1] < render_width - margin)
        & (render_coordinates[:, 0] >= margin)
        & (render_coordinates[:, 0] < render_height - margin)
    )
    coordinates = _shift_coordinates(render_coordinates[keep_mask], (-padding, -padding))
    atomic_numbers = atomic_numbers[keep_mask]
    symbols = symbols[keep_mask]

    visible_mask = _in_frame_mask(coordinates, config.image_shape)
    if int(visible_mask.sum()) == 0:
        raise RuntimeError(
            f"Projected ASE structure '{config.structure_name}' did not produce any in-frame atoms."
        )

    z_weight = atomic_numbers.astype(np.float32) ** float(config.species_intensity_power)
    z_min = float(z_weight.min())
    z_max = float(z_weight.max())
    if z_max <= z_min:
        normalized = np.full_like(z_weight, 0.5, dtype=np.float32)
    else:
        normalized = (z_weight - z_min) / (z_max - z_min)
    intensity_low, intensity_high = config.intensity_range
    intensities = (
        float(intensity_low)
        + normalized * float(intensity_high - intensity_low)
    ).astype(np.float32)
    sigmas = rng.uniform(
        config.sigma_range[0],
        config.sigma_range[1],
        size=len(coordinates),
    ).astype(np.float32)

    return {
        "coordinates": coordinates.astype(np.float32),
        "intensities": intensities.astype(np.float32),
        "sigmas": sigmas.astype(np.float32),
        "atomic_numbers": atomic_numbers.astype(np.int32),
        "symbols": symbols.tolist(),
        "visible_mask": visible_mask.astype(bool),
        "rotation_degrees": float(theta),
        "position_jitter_std": float(jitter_std),
        "repeat_xy": [int(repeat_x), int(repeat_y), int(repeat_z)],
    }


def generate_ase_projected_sample(
    config: Optional[AseStructureProjectionConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    config = config or AseStructureProjectionConfig()
    rng = _as_rng(rng)
    projected = sample_ase_projected_coordinates(config, rng)
    visible_mask = np.asarray(projected["visible_mask"], dtype=bool)
    return _render_sample_from_coordinates(
        coordinates=projected["coordinates"],
        config=config,
        rng=rng,
        target_coordinates=projected["coordinates"][visible_mask],
        metadata={
            "sample_type": "ase_projected",
            "structure_name": config.structure_name,
            "atomic_numbers": projected["atomic_numbers"],
            "symbols": projected["symbols"],
            "rotation_degrees": projected["rotation_degrees"],
            "position_jitter_std": projected["position_jitter_std"],
            "repeat_xy": projected["repeat_xy"],
            "visible_atom_count": int(visible_mask.sum()),
            "rendered_atom_count": int(len(projected["coordinates"])),
        },
        intensities=projected["intensities"],
        sigmas=projected["sigmas"],
    )


class RandomGaussianDataset(Dataset):
    """Synthetic dataset that can either generate samples on the fly or save them."""

    def __init__(
        self,
        num_samples: int,
        config: Optional[RandomGaussianConfig] = None,
        seed: Optional[int] = None,
        return_metadata: bool = False,
    ) -> None:
        self.num_samples = int(num_samples)
        self.config = config or RandomGaussianConfig()
        self.seed = seed
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return self.num_samples

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + idx)

    def __getitem__(self, idx: int):
        sample = generate_random_gaussian_sample(self.config, self._rng_for_index(idx))
        image = torch.from_numpy(sample["image"]).unsqueeze(0)
        target = torch.from_numpy(sample["target"]).unsqueeze(0)

        if not self.return_metadata:
            return image, target

        metadata = {
            "coordinates": sample["coordinates"],
            "intensities": sample["intensities"],
            "sigmas": sample["sigmas"],
        }
        return image, target, metadata


class PeriodicLatticeDataset(Dataset):
    """Synthetic periodic lattice dataset for evaluation-only generalization tests."""

    def __init__(
        self,
        num_samples: int,
        config: Optional[PeriodicLatticeConfig] = None,
        seed: Optional[int] = None,
        return_metadata: bool = True,
    ) -> None:
        self.num_samples = int(num_samples)
        self.config = config or PeriodicLatticeConfig()
        self.seed = seed
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return self.num_samples

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + idx)

    def __getitem__(self, idx: int):
        sample = generate_periodic_lattice_sample(self.config, self._rng_for_index(idx))
        image = torch.from_numpy(sample["image"]).unsqueeze(0)
        target = torch.from_numpy(sample["target"]).unsqueeze(0)

        if not self.return_metadata:
            return image, target

        metadata = {
            "coordinates": sample["coordinates"],
            "intensities": sample["intensities"],
            "sigmas": sample["sigmas"],
            "sample_type": sample["sample_type"],
            "lattice_type": sample["lattice_type"],
        }
        return image, target, metadata


class AseStructureProjectionDataset(Dataset):
    """Projected-structure dataset generated from ASE `Atoms` objects."""

    def __init__(
        self,
        num_samples: int,
        config: Optional[AseStructureProjectionConfig] = None,
        seed: Optional[int] = None,
        return_metadata: bool = True,
    ) -> None:
        self.num_samples = int(num_samples)
        self.config = config or AseStructureProjectionConfig()
        self.seed = seed
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return self.num_samples

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + idx)

    def __getitem__(self, idx: int):
        sample = generate_ase_projected_sample(self.config, self._rng_for_index(idx))
        image = torch.from_numpy(sample["image"]).unsqueeze(0)
        target = torch.from_numpy(sample["target"]).unsqueeze(0)

        if not self.return_metadata:
            return image, target

        metadata = {
            "coordinates": sample["coordinates"],
            "intensities": sample["intensities"],
            "sigmas": sample["sigmas"],
            "sample_type": sample["sample_type"],
            "structure_name": sample["structure_name"],
            "atomic_numbers": sample["atomic_numbers"],
            "symbols": sample["symbols"],
            "rotation_degrees": sample["rotation_degrees"],
            "position_jitter_std": sample["position_jitter_std"],
            "repeat_xy": sample["repeat_xy"],
        }
        return image, target, metadata


def metadata_collate(batch):
    """Collate synthetic batches while keeping variable-length metadata as a list."""

    images = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    metadata = [item[2] for item in batch]
    return images, targets, metadata


def build_synthetic_dataloaders(
    config: Optional[RandomGaussianConfig] = None,
    train_samples: int = 4_000,
    val_samples: int = 800,
    test_samples: int = 800,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    config = config or RandomGaussianConfig()

    train_dataset = RandomGaussianDataset(
        num_samples=train_samples,
        config=config,
        seed=seed,
        return_metadata=False,
    )
    val_dataset = RandomGaussianDataset(
        num_samples=val_samples,
        config=config,
        seed=seed + 100_000,
        return_metadata=False,
    )
    test_dataset = RandomGaussianDataset(
        num_samples=test_samples,
        config=config,
        seed=seed + 200_000,
        return_metadata=True,
    )

    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        collate_fn=metadata_collate,
    )
    return train_loader, val_loader, test_loader


def build_generalization_dataloaders(
    random_config: Optional[RandomGaussianConfig] = None,
    cubic_config: Optional[PeriodicLatticeConfig] = None,
    hexagonal_config: Optional[PeriodicLatticeConfig] = None,
    structure_configs: Optional[Dict[str, AseStructureProjectionConfig]] = None,
    train_samples: int = 4_000,
    val_samples: int = 800,
    random_test_samples: int = 800,
    periodic_test_samples: int = 800,
    structure_test_samples: int = 0,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
    """
    Build loaders for training on random data and testing on multiple held-out
    distributions: random, cubic, and hexagonal.
    """

    random_config = random_config or RandomGaussianConfig()
    cubic_config = cubic_config or PeriodicLatticeConfig(
        image_shape=random_config.image_shape,
        lattice_type="cubic",
        target_sigma=random_config.target_sigma,
    )
    hexagonal_config = hexagonal_config or PeriodicLatticeConfig(
        image_shape=random_config.image_shape,
        lattice_type="hexagonal",
        target_sigma=random_config.target_sigma,
    )

    persistent_workers = num_workers > 0

    train_dataset = RandomGaussianDataset(
        num_samples=train_samples,
        config=random_config,
        seed=seed,
        return_metadata=False,
    )
    val_dataset = RandomGaussianDataset(
        num_samples=val_samples,
        config=random_config,
        seed=seed + 100_000,
        return_metadata=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    test_datasets = {
        "random": RandomGaussianDataset(
            num_samples=random_test_samples,
            config=random_config,
            seed=seed + 200_000,
            return_metadata=True,
        ),
        "cubic": PeriodicLatticeDataset(
            num_samples=periodic_test_samples,
            config=cubic_config,
            seed=seed + 300_000,
            return_metadata=True,
        ),
        "hexagonal": PeriodicLatticeDataset(
            num_samples=periodic_test_samples,
            config=hexagonal_config,
            seed=seed + 400_000,
            return_metadata=True,
        ),
    }

    if structure_configs:
        for structure_idx, (name, structure_config) in enumerate(structure_configs.items(), start=1):
            test_datasets[name] = AseStructureProjectionDataset(
                num_samples=structure_test_samples,
                config=structure_config,
                seed=seed + (400_000 + structure_idx * 100_000),
                return_metadata=True,
            )

    test_loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=metadata_collate,
        )
        for name, dataset in test_datasets.items()
    }
    return train_loader, val_loader, test_loaders


def save_gaussian_dataset(
    output_dir: str | Path,
    num_samples: int,
    config: Optional[RandomGaussianConfig] = None,
    seed: int = 0,
    prefix: str = "sample",
) -> List[Path]:
    """Save a generated dataset as one NPZ file per sample."""

    config = config or RandomGaussianConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for idx in range(num_samples):
        sample = generate_random_gaussian_sample(
            config=config, rng=np.random.default_rng(seed + idx)
        )
        file_path = output_dir / f"{prefix}_{idx:05d}.npz"
        np.savez_compressed(
            file_path,
            image=sample["image"],
            target=sample["target"],
            coordinates=sample["coordinates"],
            intensities=sample["intensities"],
            sigmas=sample["sigmas"],
            config_json=json.dumps(sample["config"]),
        )
        paths.append(file_path)

    return paths


def save_gaussian_dataset_splits(
    output_dir: str | Path,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    config: Optional[RandomGaussianConfig] = None,
    seed: int = 0,
    prefix: str = "sample",
) -> Dict[str, List[Path]]:
    """
    Save train/val/test synthetic datasets into separate directories.
    """

    config = config or RandomGaussianConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_specs = [
        ("train", int(train_samples), seed),
        ("val", int(val_samples), seed + 1_000_000),
        ("test", int(test_samples), seed + 2_000_000),
    ]

    saved_paths: Dict[str, List[Path]] = {}
    for split_name, num_samples, split_seed in split_specs:
        split_dir = output_dir / split_name
        saved_paths[split_name] = save_gaussian_dataset(
            output_dir=split_dir,
            num_samples=num_samples,
            config=config,
            seed=split_seed,
            prefix=prefix,
        )

    return saved_paths


def plot_gaussian_sample(
    sample: Dict[str, Any],
    figsize: Tuple[float, float] = (12.0, 4.0),
):
    """Visualize a synthetic sample and its target/coordinates."""

    import matplotlib.pyplot as plt

    image = sample["image"]
    target = sample["target"]
    coordinates = np.asarray(sample["coordinates"])

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(target, cmap="magma")
    axes[1].set_title("Target Heatmap")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    axes[2].scatter(
        coordinates[:, 1],
        coordinates[:, 0],
        s=20,
        c="cyan",
        edgecolors="black",
        linewidths=0.5,
    )
    axes[2].set_title("Atom Centers")
    axes[2].axis("off")

    return fig, axes


def save_gaussian_preview(
    sample: Dict[str, Any],
    output_path: str | Path,
    scale: int = 3,
    separator: int = 6,
) -> Path:
    """Save a simple preview image without requiring Matplotlib."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = np.clip(sample["image"], 0.0, 1.0)
    target = np.clip(sample["target"], 0.0, 1.0)
    coordinates = np.asarray(sample["coordinates"], dtype=np.float32)

    input_panel = np.stack([image, image, image], axis=-1)
    target_panel = np.stack([target, target, target], axis=-1)
    overlay_panel = input_panel.copy()

    overlay_image = Image.fromarray((overlay_panel * 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(overlay_image)
    for y, x in coordinates:
        left = float(x - 2)
        top = float(y - 2)
        right = float(x + 2)
        bottom = float(y + 2)
        draw.ellipse((left, top, right, bottom), outline=(0, 255, 255), width=1)

    overlay_panel = np.asarray(overlay_image, dtype=np.uint8) / 255.0

    height, width = image.shape
    preview = np.ones((height, width * 3 + separator * 2, 3), dtype=np.float32)
    preview[:, :width] = input_panel
    preview[:, width + separator : 2 * width + separator] = target_panel
    preview[:, 2 * width + 2 * separator :] = overlay_panel

    preview_image = Image.fromarray((preview * 255).astype(np.uint8), mode="RGB")
    if scale > 1:
        preview_image = preview_image.resize(
            (preview_image.width * scale, preview_image.height * scale),
            resample=RESAMPLE_NEAREST,
        )
    preview_image.save(output_path)
    return output_path
