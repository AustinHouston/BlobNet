from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

_CACHE_DIR = Path(tempfile.gettempdir()) / 'blobnet-mpl-cache'
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_CACHE_DIR))
os.environ.setdefault('XDG_CACHE_HOME', str(_CACHE_DIR))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from blobnet.metrics import extract_subpixel_peak_positions, match_coordinate_sets
from blobnet.networks import build_unet
from blobnet.synthetic import (
    GeneratedAtomImageDataset,
    ImageFormationConfig,
    PeriodicLatticeConfig,
    RandomAtomImageConfig,
    build_ase_structure_unit_cell,
    generate_atom_image,
    generate_atoms_image,
    metadata_collate,
    render_atom_image,
)
from blobnet.visualization import collect_matched_offsets


DATASET_TYPES = {
    'random': RandomAtomImageConfig,
    'periodic_lattice': PeriodicLatticeConfig,
}

MODEL_CMAPS = {
    'square': 'Blues',
    'hexagonal': 'Oranges',
    'random': 'Greens',
}

MODEL_COLORS = {
    'square': '#2f69bf',
    'hexagonal': '#dd7a1f',
    'random': '#2f8f4e',
}

AXIS_LABEL_SIZE = 14
AXIS_TICK_SIZE = 12
ANNOTATION_SIZE = 10


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    checkpoint: Path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    config: ImageFormationConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _device_from_name(name: str) -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _read_yaml_config(path: Path) -> ImageFormationConfig:
    raw = yaml.safe_load(path.read_text())
    dataset_type = raw['dataset']['type']
    if dataset_type not in DATASET_TYPES:
        raise ValueError(f'Unsupported dataset type {dataset_type!r} in {path}')
    return DATASET_TYPES[dataset_type](**raw['parameters'])


def _normalize_image(image: np.ndarray, low: float = 1.0, high: float = 99.8) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [low, high])
    image = np.clip((image - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)
    return image.astype(np.float32)


def _load_blobnet_model(
    checkpoint: Path,
    device: torch.device,
    num_filters: list[int],
    dropout: float,
) -> torch.nn.Module:
    if not checkpoint.exists():
        raise FileNotFoundError(f'Missing checkpoint: {checkpoint}')
    model = build_unet(input_channels=1, num_classes=1, num_filters=num_filters, dropout=dropout)
    try:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location='cpu')
    state_dict = payload['model_state_dict'] if isinstance(payload, dict) and 'model_state_dict' in payload else payload
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _axis_missing(ax: plt.Axes, message: str) -> None:
    ax.set_facecolor('#f4f1eb')
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=8, color='#5c554f', wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#d4cdc4')


def _predict_array(model: torch.nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
    return output.astype(np.float32)


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _predict_tiled(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: int,
    overlap: int,
    batch_size: int,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    height, width = image.shape
    if height < tile_size or width < tile_size:
        padded = np.zeros((max(height, tile_size), max(width, tile_size)), dtype=np.float32)
        padded[:height, :width] = image
        return _predict_tiled(model, padded, device, tile_size, overlap, batch_size)[:height, :width]

    stride = max(1, tile_size - overlap)
    y_starts = _tile_starts(height, tile_size, stride)
    x_starts = _tile_starts(width, tile_size, stride)
    accumulator = np.zeros_like(image, dtype=np.float32)
    weights = np.zeros_like(image, dtype=np.float32)
    window_1d = np.hanning(tile_size).astype(np.float32)
    window_1d = np.maximum(window_1d, 0.08)
    window = np.outer(window_1d, window_1d).astype(np.float32)

    tiles: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    for y0 in y_starts:
        for x0 in x_starts:
            tiles.append(image[y0 : y0 + tile_size, x0 : x0 + tile_size])
            origins.append((y0, x0))

    for start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[start : start + batch_size]
        batch = torch.from_numpy(np.stack(batch_tiles, axis=0)).unsqueeze(1).to(device)
        with torch.inference_mode():
            predictions = torch.sigmoid(model(batch))[:, 0].detach().cpu().numpy()
        for prediction, (y0, x0) in zip(predictions, origins[start : start + batch_size]):
            accumulator[y0 : y0 + tile_size, x0 : x0 + tile_size] += prediction * window
            weights[y0 : y0 + tile_size, x0 : x0 + tile_size] += window

    return accumulator / np.maximum(weights, 1e-8)


def _find_haadf_with_pytemlib(path: Path) -> np.ndarray | None:
    try:
        import pyTEMlib.file_tools as ft
    except ImportError:
        return None

    dataset = ft.open_file(str(path))
    for key in dataset.keys():
        candidate = dataset[key]
        if getattr(candidate, 'title', '') == 'HAADF':
            return np.asarray(candidate, dtype=np.float32)
    return None


def _find_haadf_with_h5py(path: Path) -> np.ndarray:
    arrays: list[tuple[int, str, np.ndarray]] = []
    with h5py.File(path, 'r') as handle:
        def visit(name: str, obj: Any) -> None:
            if not hasattr(obj, 'shape') or name.endswith('/Metadata'):
                return
            shape = tuple(int(value) for value in obj.shape)
            if len(shape) >= 2 and np.issubdtype(obj.dtype, np.number):
                data = np.asarray(obj)
                data = np.squeeze(data)
                if data.ndim == 2:
                    arrays.append((data.size, name, data.astype(np.float32)))

        handle.visititems(visit)

    if not arrays:
        raise ValueError(f'No 2D numeric image dataset found in {path}')
    return max(arrays, key=lambda item: item[0])[2]


def _load_experimental_image(path: Path) -> np.ndarray:
    image = _find_haadf_with_pytemlib(path)
    if image is None:
        image = _find_haadf_with_h5py(path)
    return _normalize_image(np.squeeze(image))


def _center_crop_or_pad(image: np.ndarray, size: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    output = np.zeros((size, size), dtype=np.float32)
    src_y0 = max((image.shape[0] - size) // 2, 0)
    src_x0 = max((image.shape[1] - size) // 2, 0)
    src_y1 = min(src_y0 + size, image.shape[0])
    src_x1 = min(src_x0 + size, image.shape[1])
    crop = image[src_y0:src_y1, src_x0:src_x1]
    dst_y0 = max((size - crop.shape[0]) // 2, 0)
    dst_x0 = max((size - crop.shape[1]) // 2, 0)
    output[dst_y0 : dst_y0 + crop.shape[0], dst_x0 : dst_x0 + crop.shape[1]] = crop
    return output


def _read_experimental_feature_measurements(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing experimental feature measurements: {path}')
    payload = json.loads(path.read_text())
    return {
        str(record['image']): record
        for record in payload['images']
    }


def _make_feature_matched_experimental_view(
    image: np.ndarray,
    measurement: dict[str, Any],
    dog_small: float,
    dog_large: float,
    target_sigma_px: float,
    crop_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    dog = gaussian_filter(image, dog_small) - gaussian_filter(image, dog_large)
    dog = _normalize_image(dog)
    sigma_px = float(measurement['sigma_px_median'])
    scale = float(target_sigma_px) / sigma_px
    scaled = zoom(dog, zoom=scale, order=1, mode='nearest', prefilter=False)
    view = _center_crop_or_pad(_normalize_image(scaled), int(crop_size))
    return view, {
        'feature_match_zoom': scale,
        'expected_sigma_px': sigma_px * scale,
        'expected_fwhm_px': float(measurement['fwhm_px_median']) * scale,
        'expected_spacing_px': float(measurement['spacing_px_median']) * scale,
    }


def _plot_clean_image(ax: plt.Axes, image: np.ndarray, title: str, cmap: str = 'gray') -> None:
    ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
    if title:
        ax.set_title(title, fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])


def _make_dataset_specs(repo_root: Path, args: argparse.Namespace | None = None) -> list[DatasetSpec]:
    configs = {
        'square': getattr(args, 'square_dataset_config', None) or repo_root / 'configs/dataset_configs/square.yaml',
        'hexagonal': getattr(args, 'hexagonal_dataset_config', None) or repo_root / 'configs/dataset_configs/hexagonal.yaml',
        'random': getattr(args, 'random_dataset_config', None) or repo_root / 'configs/dataset_configs/random.yaml',
    }
    labels = {'square': 'Square', 'hexagonal': 'Hexagonal', 'random': 'Random'}
    return [DatasetSpec(key, labels[key], _read_yaml_config(path)) for key, path in configs.items()]


def _make_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    return [
        ModelSpec('square', 'Square model', args.square_checkpoint),
        ModelSpec('hexagonal', 'Hexagonal model', args.hexagonal_checkpoint),
        ModelSpec('random', 'Random model', args.random_checkpoint),
    ]


def _collect_offsets_for_model(
    model: torch.nn.Module,
    dataset: DatasetSpec,
    device: torch.device,
    samples: int,
    batch_size: int,
    seed: int,
    threshold_rel: float,
    match_distance: float,
) -> dict[str, np.ndarray | float | int]:
    loader = DataLoader(
        GeneratedAtomImageDataset(samples, dataset.config, seed=seed, return_metadata=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=metadata_collate,
    )
    return collect_matched_offsets(model, loader, device, threshold_rel, match_distance)


def _annotate_probability_metric(ax: plt.Axes, result: dict[str, np.ndarray | float | int] | None) -> None:
    if result is None:
        return
    ax.text(
        0.04,
        0.96,
        f"F1={float(result['f1']):.3f}",
        transform=ax.transAxes,
        ha='left',
        va='top',
        color='#1f1f1f',
        fontsize=ANNOTATION_SIZE,
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.78, 'boxstyle': 'round,pad=0.18'},
    )


def make_figure_1(args: argparse.Namespace) -> Path:
    repo_root = _repo_root()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    datasets = _make_dataset_specs(repo_root, args)
    models = _make_model_specs(args)

    examples = {
        dataset.key: generate_atom_image(dataset.config, np.random.default_rng(args.seed + index))
        for index, dataset in enumerate(datasets)
    }
    loaded_models: dict[str, torch.nn.Module | None] = {}
    for spec in models:
        if spec.checkpoint.exists():
            loaded_models[spec.key] = _load_blobnet_model(spec.checkpoint, device, args.num_filters, args.dropout)
        else:
            loaded_models[spec.key] = None

    predictions: dict[tuple[str, str], np.ndarray] = {}
    offset_results: dict[tuple[str, str], dict[str, np.ndarray | float | int]] = {}
    for model_index, model_spec in enumerate(models):
        model = loaded_models[model_spec.key]
        if model is None:
            continue
        for dataset_index, dataset in enumerate(datasets):
            predictions[(model_spec.key, dataset.key)] = _predict_array(model, examples[dataset.key]['image'], device)
            offset_results[(model_spec.key, dataset.key)] = _collect_offsets_for_model(
                model,
                dataset,
                device,
                samples=args.offset_samples,
                batch_size=args.batch_size,
                seed=args.seed + 10_000 + model_index * 1_000 + dataset_index * 100,
                threshold_rel=args.threshold_rel,
                match_distance=args.match_distance,
            )

    fig = plt.figure(figsize=(19, 8.8))
    grid = fig.add_gridspec(
        3,
        8,
        left=0.035,
        right=0.99,
        top=0.985,
        bottom=0.075,
        wspace=0.17,
        hspace=0.28,
    )

    for row, dataset in enumerate(datasets):
        ax = fig.add_subplot(grid[row, 0])
        _plot_clean_image(ax, examples[dataset.key]['image'], '')
        ax = fig.add_subplot(grid[row, 1])
        _plot_clean_image(ax, examples[dataset.key]['target'], '', cmap='magma')

    for row, dataset in enumerate(datasets):
        for col, model_spec in enumerate(models):
            ax = fig.add_subplot(grid[row, col + 2])
            prediction = predictions.get((model_spec.key, dataset.key))
            if prediction is None:
                _axis_missing(ax, 'Checkpoint missing')
            else:
                cmap = MODEL_CMAPS.get(model_spec.key, 'viridis')
                ax.imshow(prediction, cmap=cmap, vmin=0.0, vmax=max(float(prediction.max()), 1e-6))
                if col == 0:
                    ax.set_ylabel(f'{dataset.label} test', fontsize=AXIS_LABEL_SIZE)
                ax.set_xticks([])
                ax.set_yticks([])
                _annotate_probability_metric(ax, offset_results.get((model_spec.key, dataset.key)))

        for col, model_spec in enumerate(models):
            ax = fig.add_subplot(grid[row, col + 5])
            result = offset_results.get((model_spec.key, dataset.key))
            if result is None:
                _axis_missing(ax, 'Checkpoint missing')
                continue
            offsets = np.asarray(result['offsets_xy'], dtype=np.float32)
            if len(offsets):
                hist, x_edges, y_edges = np.histogram2d(
                    offsets[:, 0],
                    offsets[:, 1],
                    bins=args.offset_bins,
                    range=[[-args.offset_range, args.offset_range], [-args.offset_range, args.offset_range]],
                )
                ax.imshow(
                    hist.T,
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                    origin='lower',
                    cmap='magma',
                    norm=LogNorm(vmin=1, vmax=max(float(hist.max()), 1.0)),
                )
                ax.scatter(
                    offsets[:, 0],
                    offsets[:, 1],
                    s=3,
                    c=MODEL_COLORS.get(model_spec.key, 'white'),
                    alpha=0.35,
                    linewidths=0,
                )
            ax.axhline(0.0, color='white', linewidth=0.7, alpha=0.65)
            ax.axvline(0.0, color='white', linewidth=0.7, alpha=0.65)
            ax.set_xlim(-args.offset_range, args.offset_range)
            ax.set_ylim(-args.offset_range, args.offset_range)
            ax.set_aspect('equal')
            ax.set_facecolor('#17121f')
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            ax.tick_params(labelsize=AXIS_TICK_SIZE)
            ax.text(
                0.04,
                0.96,
                f"F1={float(result['f1']):.3f}\nRMSE={float(result['rmse']):.2f}px",
                transform=ax.transAxes,
                ha='left',
                va='top',
                color='white',
                fontsize=ANNOTATION_SIZE,
            )

    output_path = output_dir / 'figure1_training_geometry_generalization.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    missing = [str(spec.checkpoint) for spec in models if loaded_models[spec.key] is None]
    manifest = {
        'figure': 'figure1_training_geometry_generalization',
        'output_path': str(output_path),
        'missing_checkpoints': missing,
        'offset_samples_per_dataset': args.offset_samples,
        'threshold_rel': args.threshold_rel,
        'match_distance': args.match_distance,
    }
    (output_dir / 'figure1_training_geometry_generalization.json').write_text(json.dumps(manifest, indent=2))
    return output_path


def make_figure_2(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    model = _load_blobnet_model(args.checkpoint, device, args.num_filters, args.dropout)
    measurements = _read_experimental_feature_measurements(args.experimental_measurements)

    files = [
        ('WS2 grain boundary', args.data_dir / 'WS2.emd'),
        ('Quasicrystal', args.data_dir / 'QuasiCrystal.emd'),
        ('Twin boundary', args.data_dir / 'TwinBoundary.emd'),
        ('Twins overview', args.data_dir / 'TwinsOverview.emd'),
    ]
    images: list[tuple[str, np.ndarray, dict[str, float]]] = []
    outputs: list[np.ndarray] = []
    for label, path in files:
        image = _load_experimental_image(path)
        model_input, transform = _make_feature_matched_experimental_view(
            image,
            measurements[path.stem],
            dog_small=args.dog_small,
            dog_large=args.dog_large,
            target_sigma_px=args.feature_match_sigma_px,
            crop_size=args.experimental_crop_size,
        )
        prediction = _predict_tiled(model, model_input, device, args.tile_size, args.tile_overlap, args.batch_size)
        images.append((label, model_input, transform))
        outputs.append(prediction)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), constrained_layout=True)
    for col, ((label, image, _transform), output) in enumerate(zip(images, outputs)):
        _plot_clean_image(axes[0, col], image, '', cmap='gray')
        axes[1, col].imshow(output, cmap=MODEL_CMAPS['random'], vmin=0.0, vmax=max(float(output.max()), 1e-6))
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

    output_path = output_dir / 'figure2_experimental_haadf_outputs.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    summary = [
        {
            'label': label,
            'shape_y': int(image.shape[0]),
            'shape_x': int(image.shape[1]),
            'display_processing': 'DoG background-subtracted, feature-sigma matched, center-cropped',
            'target_sigma_px': float(args.feature_match_sigma_px),
            **transform,
            'output_mean': float(output.mean()),
            'output_max': float(output.max()),
        }
        for (label, image, transform), output in zip(images, outputs)
    ]
    (output_dir / 'figure2_experimental_haadf_outputs.json').write_text(json.dumps(summary, indent=2))
    return output_path


def _stem_like_render_config(
    shape: tuple[int, int],
    sigma_range: tuple[float, float],
) -> ImageFormationConfig:
    return ImageFormationConfig(
        image_shape=shape,
        sigma_range=sigma_range,
        intensity_range=(0.3, 1.0),
        target_sigma=0.9,
        background_range=(0.0, 0.18),
        gradient_range=(-0.055, 0.055),
        inhomogeneous_background_range=(0.03, 0.12),
        inhomogeneous_background_sigma_fraction_range=(0.18, 0.45),
        low_frequency_noise_range=(0.03, 0.13),
        low_frequency_sigma_fraction_range=(0.04, 0.11),
        read_noise_std_range=(0.018, 0.055),
        total_counts_range=(5_000.0, 18_000.0),
        counts_per_pixel_range=None,
        blur_sigma_range=(0.15, 0.65),
        edge_padding=12,
        normalize_input=True,
        clamp_target=True,
    )


def _render_atoms_stem_panel(
    atoms: Any,
    shape: tuple[int, int],
    seed: int,
    atom_sigma_range: tuple[float, float],
) -> np.ndarray:
    rendered = generate_atoms_image(
        atoms,
        _stem_like_render_config(shape, atom_sigma_range),
        np.random.default_rng(seed),
        atom_sigma_range=atom_sigma_range,
    )
    return rendered['image']


def _merge_projected_columns_for_rendering(
    xy: np.ndarray,
    atomic_numbers: np.ndarray,
    tolerance: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.round(np.asarray(xy, dtype=np.float32) / float(tolerance)).astype(np.int32)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault((int(key[0]), int(key[1])), []).append(index)

    merged_xy = []
    merged_numbers = []
    weights = np.asarray(atomic_numbers, dtype=np.float32) ** 1.45
    for indices in groups.values():
        index_array = np.asarray(indices, dtype=np.int64)
        merged_xy.append(np.average(xy[index_array], axis=0, weights=weights[index_array]))
        merged_numbers.append(int(np.max(atomic_numbers[index_array])))
    return np.asarray(merged_xy, dtype=np.float32), np.asarray(merged_numbers, dtype=np.int32)


def _make_tmd_edge_record(
    shape: tuple[int, int],
    seed: int,
    sigma_range: tuple[float, float],
    total_counts_range: tuple[float, float] = (35.0, 250.0),
    structure_name: str = 'ws2',
    quiet_background: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    structure_name = str(structure_name).lower()
    unit = build_ase_structure_unit_cell(structure_name)
    repeated = unit.repeat((44, 44, 1))
    positions = np.asarray(repeated.get_positions(), dtype=np.float32)
    numbers = np.asarray(repeated.get_atomic_numbers(), dtype=np.int32)
    xy_angstrom = positions[:, :2]
    xy_angstrom -= xy_angstrom.mean(axis=0, keepdims=True)
    xy_angstrom, numbers = _merge_projected_columns_for_rendering(xy_angstrom, numbers)

    theta = np.deg2rad(17.0)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    xy = xy_angstrom @ rotation.T
    tmd_pixel_size_angstrom = 0.12
    xy = xy / tmd_pixel_size_angstrom
    xy[:, 0] += shape[1] * 0.49
    xy[:, 1] += shape[0] * 0.54

    x = xy[:, 0]
    y = xy[:, 1]
    edge_line = 150.0 + 0.26 * x + 8.0 * np.sin(x / 42.0)
    terrace_mask = y > edge_line
    notch_mask = ((x - 330.0) ** 2 + (y - 238.0) ** 2) > 52.0**2
    island_mask = ((x - 168.0) ** 2 / 80.0**2 + (y - 340.0) ** 2 / 45.0**2) < 1.0
    frame_mask = (x >= -16.0) & (x < shape[1] + 16.0) & (y >= -16.0) & (y < shape[0] + 16.0)
    keep = frame_mask & ((terrace_mask & notch_mask) | island_mask)
    xy = xy[keep]
    numbers = numbers[keep]

    edge_distance = y[keep] - edge_line[keep]
    edge_zone = (edge_distance > -6.0) & (edge_distance < 22.0)
    vacancy_probability = np.where(edge_zone, 0.18, 0.025)
    keep_vacancy = rng.random(len(xy)) > vacancy_probability
    xy = xy[keep_vacancy]
    numbers = numbers[keep_vacancy]
    edge_zone = edge_zone[keep_vacancy]

    xy = xy + rng.normal(0.0, np.where(edge_zone[:, None], 0.32, 0.11), size=xy.shape).astype(np.float32)
    coordinates_yx = xy[:, [1, 0]].astype(np.float32)
    visible = _in_frame_mask_for_figure(coordinates_yx, shape)
    coordinates_yx = coordinates_yx[visible]
    numbers = numbers[visible]

    heavy_atomic_number = 42 if structure_name.startswith('mos2') else 74
    weights = numbers.astype(np.float32) ** 1.45
    intensities = (weights / max(float(weights.max()), 1e-6)).astype(np.float32)
    sigma_min, sigma_max = float(sigma_range[0]), float(sigma_range[1])
    sigmas = np.where(numbers >= heavy_atomic_number, sigma_max, sigma_min).astype(np.float32)

    config = _edge_figure_render_config(
        shape,
        (sigma_min, sigma_max),
        total_counts_range=total_counts_range,
        quiet_background=quiet_background,
    )
    return render_atom_image(
        coordinates_yx,
        config,
        rng,
        intensities=intensities,
        sigmas=sigmas,
        target_coordinates=coordinates_yx,
        metadata={
            'image_type': f'{structure_name}_monolayer_edge',
            'visible_atom_count': int(len(coordinates_yx)),
            'heavy_columns': int(np.sum(numbers >= heavy_atomic_number)),
            's_columns': int(np.sum(numbers < heavy_atomic_number)),
            'pixel_size_angstrom': float(tmd_pixel_size_angstrom),
        },
    )


def _make_ws2_edge_record(
    shape: tuple[int, int],
    seed: int,
    sigma_range: tuple[float, float],
    total_counts_range: tuple[float, float] = (35.0, 250.0),
    quiet_background: bool = False,
) -> dict[str, Any]:
    return _make_tmd_edge_record(shape, seed, sigma_range, total_counts_range=total_counts_range, structure_name='ws2', quiet_background=quiet_background)


def _make_mos2_edge_record(
    shape: tuple[int, int],
    seed: int,
    sigma_range: tuple[float, float],
    total_counts_range: tuple[float, float] = (35.0, 250.0),
    quiet_background: bool = False,
) -> dict[str, Any]:
    return _make_tmd_edge_record(shape, seed, sigma_range, total_counts_range=total_counts_range, structure_name='mos2', quiet_background=quiet_background)


def _edge_figure_render_config(
    shape: tuple[int, int],
    sigma_range: tuple[float, float],
    total_counts_range: tuple[float, float] = (35.0, 250.0),
    quiet_background: bool = False,
) -> ImageFormationConfig:
    sigma_min, sigma_max = float(sigma_range[0]), float(sigma_range[1])
    if quiet_background:
        background_range = (0.018, 0.090)
        gradient_range = (-0.035, 0.035)
        inhomogeneous_background_range = (0.025, 0.075)
        low_frequency_noise_range = (0.018, 0.075)
        read_noise_std_range = (0.018, 0.060)
    else:
        background_range = (0.04, 0.32)
        gradient_range = (-0.08, 0.08)
        inhomogeneous_background_range = (0.07, 0.18)
        low_frequency_noise_range = (0.05, 0.20)
        read_noise_std_range = (0.08, 0.22)
    return ImageFormationConfig(
        image_shape=shape,
        sigma_range=(sigma_min, sigma_max),
        intensity_range=(0.2, 1.0),
        target_sigma=2.0,
        background_range=background_range,
        gradient_range=gradient_range,
        inhomogeneous_background_range=inhomogeneous_background_range,
        inhomogeneous_background_sigma_fraction_range=(0.16, 0.42),
        low_frequency_noise_range=low_frequency_noise_range,
        low_frequency_sigma_fraction_range=(0.05, 0.14),
        read_noise_std_range=read_noise_std_range,
        total_counts_range=total_counts_range,
        counts_per_pixel_range=None,
        blur_sigma_range=(0.15, 0.85),
        edge_padding=0,
        normalize_input=True,
        clamp_target=True,
    )


def _normalized_species_intensities(numbers: np.ndarray, low: float = 0.22, high: float = 1.0) -> np.ndarray:
    weights = np.asarray(numbers, dtype=np.float32) ** 1.45
    span = max(float(weights.max() - weights.min()), 1e-6)
    weights = (weights - weights.min()) / span
    return (float(low) + float(high - low) * weights).astype(np.float32)


def _sigmas_from_species(numbers: np.ndarray, sigma_range: tuple[float, float]) -> np.ndarray:
    weights = np.asarray(numbers, dtype=np.float32)
    span = max(float(weights.max() - weights.min()), 1e-6)
    normalized = (weights - weights.min()) / span
    sigma_min, sigma_max = float(sigma_range[0]), float(sigma_range[1])
    return (sigma_min + normalized * (sigma_max - sigma_min)).astype(np.float32)


def _make_sto_edge_record(
    shape: tuple[int, int],
    seed: int,
    sigma_range: tuple[float, float],
    total_counts_range: tuple[float, float] = (35.0, 250.0),
    quiet_background: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unit = build_ase_structure_unit_cell('sto')
    repeated = unit.repeat((44, 44, 1))
    positions = np.asarray(repeated.get_positions(), dtype=np.float32)
    numbers = np.asarray(repeated.get_atomic_numbers(), dtype=np.int32)
    xy_angstrom = positions[:, :2]
    xy_angstrom -= xy_angstrom.mean(axis=0, keepdims=True)
    xy_angstrom, numbers = _merge_projected_columns_for_rendering(xy_angstrom, numbers, tolerance=0.45)

    theta = np.deg2rad(-8.0)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    pixel_size_angstrom = 0.13
    xy = (xy_angstrom @ rotation.T) / pixel_size_angstrom
    xy[:, 0] += shape[1] * 0.50
    xy[:, 1] += shape[0] * 0.55

    x = xy[:, 0]
    y = xy[:, 1]
    edge_line = 135.0 + 0.22 * x + 7.5 * np.sin(x / 48.0)
    terrace_mask = y > edge_line
    notch_mask = ((x - 340.0) ** 2 + (y - 258.0) ** 2) > 42.0**2
    frame_mask = (x >= -16.0) & (x < shape[1] + 16.0) & (y >= -16.0) & (y < shape[0] + 16.0)
    keep = frame_mask & terrace_mask & notch_mask
    xy = xy[keep]
    numbers = numbers[keep]

    edge_distance = y[keep] - edge_line[keep]
    edge_zone = (edge_distance > -5.0) & (edge_distance < 24.0)
    keep_vacancy = rng.random(len(xy)) > np.where(edge_zone, 0.13, 0.02)
    xy = xy[keep_vacancy]
    numbers = numbers[keep_vacancy]
    edge_zone = edge_zone[keep_vacancy]

    xy = xy + rng.normal(0.0, np.where(edge_zone[:, None], 0.38, 0.10), size=xy.shape).astype(np.float32)
    coordinates_yx = xy[:, [1, 0]].astype(np.float32)
    visible = _in_frame_mask_for_figure(coordinates_yx, shape)
    coordinates_yx = coordinates_yx[visible]
    numbers = numbers[visible]

    return render_atom_image(
        coordinates_yx,
        _edge_figure_render_config(shape, sigma_range, total_counts_range=total_counts_range, quiet_background=quiet_background),
        rng,
        intensities=_normalized_species_intensities(numbers),
        sigmas=_sigmas_from_species(numbers, sigma_range),
        target_coordinates=coordinates_yx,
        metadata={
            'image_type': 'srtio3_edge',
            'visible_atom_count': int(len(coordinates_yx)),
            'pixel_size_angstrom': float(pixel_size_angstrom),
        },
    )


def _make_graphene_rattled_edge_record(
    shape: tuple[int, int],
    seed: int,
    sigma_range: tuple[float, float],
    total_counts_range: tuple[float, float] = (35.0, 250.0),
    quiet_background: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    nearest_neighbor_px = 15.0
    a1 = np.array([np.sqrt(3.0) * nearest_neighbor_px, 0.0], dtype=np.float32)
    a2 = np.array([0.5 * np.sqrt(3.0) * nearest_neighbor_px, 1.5 * nearest_neighbor_px], dtype=np.float32)
    basis = [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.0, nearest_neighbor_px], dtype=np.float32),
    ]
    points = []
    for i in range(-28, 29):
        for j in range(-28, 29):
            origin = i * a1 + j * a2
            for offset in basis:
                points.append(origin + offset)
    xy = np.asarray(points, dtype=np.float32)
    xy -= xy.mean(axis=0, keepdims=True)
    theta = np.deg2rad(10.0)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    xy = xy @ rotation.T
    xy[:, 0] += shape[1] * 0.48
    xy[:, 1] += shape[0] * 0.56

    x = xy[:, 0]
    y = xy[:, 1]
    edge_line = 145.0 + 0.18 * x + 9.0 * np.sin(x / 34.0)
    terrace_mask = y > edge_line
    notch_mask = ((x - 315.0) ** 2 + (y - 248.0) ** 2) > 46.0**2
    frame_mask = (x >= -16.0) & (x < shape[1] + 16.0) & (y >= -16.0) & (y < shape[0] + 16.0)
    keep = frame_mask & terrace_mask & notch_mask
    xy = xy[keep]
    numbers = np.full((len(xy),), 6, dtype=np.int32)

    edge_distance = y[keep] - edge_line[keep]
    edge_zone = (edge_distance > -8.0) & (edge_distance < 44.0)
    keep_vacancy = rng.random(len(xy)) > np.where(edge_zone, 0.10, 0.01)
    xy = xy[keep_vacancy]
    numbers = numbers[keep_vacancy]
    edge_zone = edge_zone[keep_vacancy]

    rattle = np.where(edge_zone[:, None], 2.45, 0.18)
    xy = xy + rng.normal(0.0, rattle, size=xy.shape).astype(np.float32)
    coordinates_yx = xy[:, [1, 0]].astype(np.float32)
    visible = _in_frame_mask_for_figure(coordinates_yx, shape)
    coordinates_yx = coordinates_yx[visible]
    numbers = numbers[visible]

    sigma_min, sigma_max = float(sigma_range[0]), float(sigma_range[1])
    sigmas = rng.uniform(sigma_min, sigma_max, size=len(coordinates_yx)).astype(np.float32)
    return render_atom_image(
        coordinates_yx,
        _edge_figure_render_config(shape, sigma_range, total_counts_range=total_counts_range, quiet_background=quiet_background),
        rng,
        intensities=np.full((len(coordinates_yx),), 0.78, dtype=np.float32),
        sigmas=sigmas,
        target_coordinates=coordinates_yx,
        metadata={
            'image_type': 'graphene_rattled_edge',
            'visible_atom_count': int(len(coordinates_yx)),
            'nearest_neighbor_px': float(nearest_neighbor_px),
            'edge_rattle_std_px': 2.45,
            'bulk_rattle_std_px': 0.18,
        },
    )


def _in_frame_mask_for_figure(coordinates: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return (
        (coordinates[:, 0] >= 0.0)
        & (coordinates[:, 0] < float(shape[0]))
        & (coordinates[:, 1] >= 0.0)
        & (coordinates[:, 1] < float(shape[1]))
    )


def _unmatched_coordinate_mask(coordinates: np.ndarray, matched_coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
    matched_coordinates = np.asarray(matched_coordinates, dtype=np.float32).reshape(-1, 2)
    if len(coordinates) == 0:
        return np.zeros((0,), dtype=bool)
    if len(matched_coordinates) == 0:
        return np.ones((len(coordinates),), dtype=bool)
    distances = np.linalg.norm(coordinates[:, None, :] - matched_coordinates[None, :, :], axis=2)
    return np.min(distances, axis=1) > 1e-4


def _localization_classes_for_figure(
    prediction: np.ndarray,
    true_coordinates: np.ndarray,
    threshold_rel: float,
    min_distance: int,
    peak_window_size: int,
    match_distance: float,
) -> dict[str, Any]:
    predicted_coordinates = extract_subpixel_peak_positions(
        prediction,
        threshold_rel=threshold_rel,
        min_distance=min_distance,
        window_size=peak_window_size,
    )
    matches = match_coordinate_sets(predicted_coordinates, true_coordinates, max_distance=match_distance)
    matched_predicted = np.asarray(matches['matched_predicted'], dtype=np.float32).reshape(-1, 2)
    matched_truth = np.asarray(matches['matched_truth'], dtype=np.float32).reshape(-1, 2)
    false_positives = predicted_coordinates[_unmatched_coordinate_mask(predicted_coordinates, matched_predicted)]
    false_negatives = true_coordinates[_unmatched_coordinate_mask(true_coordinates, matched_truth)]
    tp = int(matches['tp'])
    fp = int(matches['fp'])
    fn = int(matches['fn'])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        'predicted_coordinates': predicted_coordinates,
        'true_positives': matched_truth,
        'matched_predicted': matched_predicted,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'errors': np.asarray(matches['errors'], dtype=np.float32),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


def _plot_localization_scatter_for_figure(
    ax: plt.Axes,
    classes: dict[str, Any],
    shape: tuple[int, int],
    title: str,
    show_legend: bool = False,
    marker_color: str = '#2e7d32',
) -> None:
    ax.set_facecolor('#fbfbf7')
    true_positive = np.asarray(classes['true_positives'], dtype=np.float32)
    false_positive = np.asarray(classes['false_positives'], dtype=np.float32)
    false_negative = np.asarray(classes['false_negatives'], dtype=np.float32)

    if len(true_positive):
        ax.scatter(true_positive[:, 1], true_positive[:, 0], s=13, c=marker_color, marker='o', linewidths=0, alpha=0.86, label='TP')
    if len(false_positive):
        ax.scatter(false_positive[:, 1], false_positive[:, 0], s=22, c='#c62828', marker='x', linewidths=0.9, alpha=0.9, label='FP')
    if len(false_negative):
        ax.scatter(false_negative[:, 1], false_negative[:, 0], s=28, facecolors='none', edgecolors='#6a1b9a', marker='o', linewidths=1.0, alpha=0.9, label='FN')

    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0], 0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if show_legend:
        handles = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor=marker_color, markeredgecolor=marker_color, markersize=5, label=f"TP: {int(classes['tp'])}"),
            Line2D([0], [0], marker='x', color='#c62828', markersize=6, linestyle='None', label=f"FP: {int(classes['fp'])}"),
            Line2D([0], [0], marker='o', color='#6a1b9a', markerfacecolor='none', markersize=6, linestyle='None', label=f"FN: {int(classes['fn'])}"),
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=9, frameon=True, handlelength=0.9, borderpad=0.25, labelspacing=0.25)


def _plot_ground_truth_for_figure(
    ax: plt.Axes,
    target: np.ndarray,
) -> None:
    ax.imshow(np.asarray(target, dtype=np.float32), cmap='magma', vmin=0.0, vmax=max(float(np.max(target)), 1e-6))
    ax.set_xticks([])
    ax.set_yticks([])


def make_figure_4(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    models = [
        ModelSpec('square', 'Square model', args.square_checkpoint),
        ModelSpec('hexagonal', 'Hexagonal model', args.hexagonal_checkpoint),
        ModelSpec('random', 'Measured random model', args.random_checkpoint),
    ]
    loaded_models = {
        spec.key: _load_blobnet_model(spec.checkpoint, device, args.num_filters, args.dropout)
        for spec in models
    }

    record = _make_ws2_edge_record((args.height, args.width), args.seed, (args.feature_sigma_min, args.feature_sigma_max))
    image = np.asarray(record['image'], dtype=np.float32)
    target = np.asarray(record['target'], dtype=np.float32)
    coordinates = np.asarray(record['coordinates'], dtype=np.float32)
    shape = (int(args.height), int(args.width))
    if len(coordinates) > 1:
        nearest_spacing = cKDTree(coordinates).query(coordinates, k=2)[0][:, 1]
    else:
        nearest_spacing = np.asarray([], dtype=np.float32)
    predictions = {
        spec.key: _predict_array(loaded_models[spec.key], image, device)
        for spec in models
    }
    localization = {
        spec.key: _localization_classes_for_figure(
            predictions[spec.key],
            coordinates,
            threshold_rel=args.localization_threshold_rel,
            min_distance=args.peak_min_distance,
            peak_window_size=args.peak_window_size,
            match_distance=args.localization_match_distance,
        )
        for spec in models
    }

    fig = plt.figure(figsize=(16.0, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.05, 1.0, 1.0, 1.0])

    ax_input = fig.add_subplot(grid[0, 0])
    _plot_clean_image(ax_input, image, '')
    ax_target = fig.add_subplot(grid[1, 0])
    _plot_clean_image(ax_target, target, '', cmap='magma')

    for column, spec in enumerate(models, start=1):
        prediction = predictions[spec.key]
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(prediction, cmap=MODEL_CMAPS.get(spec.key, 'viridis'), vmin=0.0, vmax=max(float(prediction.max()), 1e-6))
        ax.set_xticks([])
        ax.set_yticks([])

        scatter_ax = fig.add_subplot(grid[1, column])
        _plot_localization_scatter_for_figure(
            scatter_ax,
            localization[spec.key],
            shape,
            spec.label,
            show_legend=column == 1,
        )

    output_path = output_dir / 'figure4_ws2_edge_model_comparison.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'output_path': str(output_path),
        'seed': int(args.seed),
        'shape': [int(args.height), int(args.width)],
        'visible_atom_count': int(record.get('visible_atom_count', len(record['coordinates']))),
        'w_columns': int(record.get('w_columns', 0)),
        's_columns': int(record.get('s_columns', 0)),
        'pixel_size_angstrom': float(record.get('pixel_size_angstrom', 0.0)),
        'feature_sigma_range_px': [float(args.feature_sigma_min), float(args.feature_sigma_max)],
        'poisson_total_counts_range': [35.0, 250.0],
        'nearest_neighbor_spacing_px': {
            'min': float(np.min(nearest_spacing)) if len(nearest_spacing) else None,
            'p10': float(np.percentile(nearest_spacing, 10.0)) if len(nearest_spacing) else None,
            'median': float(np.median(nearest_spacing)) if len(nearest_spacing) else None,
            'mean': float(np.mean(nearest_spacing)) if len(nearest_spacing) else None,
            'p90': float(np.percentile(nearest_spacing, 90.0)) if len(nearest_spacing) else None,
            'max': float(np.max(nearest_spacing)) if len(nearest_spacing) else None,
        },
        'localization_settings': {
            'threshold_rel': float(args.localization_threshold_rel),
            'match_distance_px': float(args.localization_match_distance),
            'peak_min_distance_px': int(args.peak_min_distance),
            'peak_window_size_px': int(args.peak_window_size),
        },
        'checkpoints': {spec.key: str(spec.checkpoint) for spec in models},
        'localization_metrics': {
            key: {
                'tp': int(value['tp']),
                'fp': int(value['fp']),
                'fn': int(value['fn']),
                'precision': float(value['precision']),
                'recall': float(value['recall']),
                'f1': float(value['f1']),
                'mean_error': float(np.mean(value['errors'])) if len(value['errors']) else None,
                'rmse': float(np.sqrt(np.mean(value['errors'] ** 2))) if len(value['errors']) else None,
            }
            for key, value in localization.items()
        },
        'prediction_stats': {
            key: {
                'mean': float(value.mean()),
                'max': float(value.max()),
                'p99': float(np.percentile(value, 99.0)),
            }
            for key, value in predictions.items()
        },
    }
    (output_dir / 'figure4_ws2_edge_model_comparison.json').write_text(json.dumps(summary, indent=2))
    return output_path


def _nearest_spacing_summary(coordinates: np.ndarray) -> dict[str, float | None]:
    coordinates = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
    if len(coordinates) <= 1:
        return {'min': None, 'p10': None, 'median': None, 'mean': None, 'p90': None, 'max': None}
    nearest_spacing = cKDTree(coordinates).query(coordinates, k=2)[0][:, 1]
    return {
        'min': float(np.min(nearest_spacing)),
        'p10': float(np.percentile(nearest_spacing, 10.0)),
        'median': float(np.median(nearest_spacing)),
        'mean': float(np.mean(nearest_spacing)),
        'p90': float(np.percentile(nearest_spacing, 90.0)),
        'max': float(np.max(nearest_spacing)),
    }


def _localization_metric_summary(classes: dict[str, Any]) -> dict[str, float | int | None]:
    errors = np.asarray(classes['errors'], dtype=np.float32)
    return {
        'tp': int(classes['tp']),
        'fp': int(classes['fp']),
        'fn': int(classes['fn']),
        'precision': float(classes['precision']),
        'recall': float(classes['recall']),
        'f1': float(classes['f1']),
        'mean_error': float(np.mean(errors)) if len(errors) else None,
        'rmse': float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
    }


def make_figure_5(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    shape = (int(args.height), int(args.width))
    sigma_range = (float(args.feature_sigma_min), float(args.feature_sigma_max))
    models = [
        ModelSpec('square', 'Square model', args.square_checkpoint),
        ModelSpec('hexagonal', 'Hexagonal model', args.hexagonal_checkpoint),
        ModelSpec('random', 'Measured random model', args.random_checkpoint),
    ]
    loaded_models = {
        spec.key: _load_blobnet_model(spec.checkpoint, device, args.num_filters, args.dropout)
        for spec in models
    }
    figure5_counts = (16.0, 16.0)
    cases = [
        ('mos2_edge', _make_mos2_edge_record(shape, args.seed, sigma_range, total_counts_range=figure5_counts, quiet_background=True)),
        ('srtio3_edge', _make_sto_edge_record(shape, args.seed + 101, sigma_range, total_counts_range=figure5_counts, quiet_background=True)),
        ('graphene_rattled_edge', _make_graphene_rattled_edge_record(shape, args.seed + 202, sigma_range, total_counts_range=figure5_counts, quiet_background=True)),
    ]

    fig = plt.figure(figsize=(20.0, 11.4), constrained_layout=True)
    grid = fig.add_gridspec(len(cases), 5, width_ratios=[1.05, 1.0, 1.0, 1.0, 1.0], wspace=0.05, hspace=0.05)
    summary: dict[str, Any] = {
        'output_path': str(output_dir / 'figure5_edge_lattice_model_diagnostics.png'),
        'seed': int(args.seed),
        'shape': [int(args.height), int(args.width)],
        'feature_sigma_range_px': [float(args.feature_sigma_min), float(args.feature_sigma_max)],
        'poisson_total_counts_range': [float(figure5_counts[0]), float(figure5_counts[1])],
        'background_profile': 'medium_background_poisson_count_16',
        'localization_settings': {
            'threshold_rel': float(args.localization_threshold_rel),
            'match_distance_px': float(args.localization_match_distance),
            'peak_min_distance_px': int(args.peak_min_distance),
            'peak_window_size_px': int(args.peak_window_size),
        },
        'checkpoints': {spec.key: str(spec.checkpoint) for spec in models},
        'cases': {},
    }

    for row, (case_key, record) in enumerate(cases):
        image = np.asarray(record['image'], dtype=np.float32)
        target = np.asarray(record['target'], dtype=np.float32)
        coordinates = np.asarray(record['coordinates'], dtype=np.float32)
        ax_image = fig.add_subplot(grid[row, 0])
        _plot_clean_image(ax_image, image, '')

        ax_ground_truth = fig.add_subplot(grid[row, 1])
        _plot_ground_truth_for_figure(ax_ground_truth, target)

        case_predictions = {
            spec.key: _predict_array(loaded_models[spec.key], image, device)
            for spec in models
        }
        case_thresholds = {
            spec.key: 0.50
            for spec in models
        }
        if case_key == 'mos2_edge':
            case_thresholds['hexagonal'] = 0.30
            case_thresholds['random'] = 0.24
        elif case_key == 'srtio3_edge':
            case_thresholds['random'] = 0.42
        elif case_key == 'graphene_rattled_edge':
            case_thresholds['hexagonal'] = 0.95
            case_thresholds['random'] = 0.62
        case_localization = {
            spec.key: _localization_classes_for_figure(
                case_predictions[spec.key],
                coordinates,
                threshold_rel=case_thresholds[spec.key],
                min_distance=args.peak_min_distance,
                peak_window_size=args.peak_window_size,
                match_distance=args.localization_match_distance,
            )
            for spec in models
        }

        for column, spec in enumerate(models, start=2):
            scatter_ax = fig.add_subplot(grid[row, column])
            _plot_localization_scatter_for_figure(
                scatter_ax,
                case_localization[spec.key],
                shape,
                spec.label,
                show_legend=True,
                marker_color=MODEL_COLORS.get(spec.key, '#2e7d32'),
            )

        summary['cases'][case_key] = {
            'image_type': str(record.get('image_type', case_key)),
            'visible_atom_count': int(record.get('visible_atom_count', len(coordinates))),
            'pixel_size_angstrom': float(record.get('pixel_size_angstrom', 0.0)),
            'nearest_neighbor_spacing_px': _nearest_spacing_summary(coordinates),
            'localization_threshold_rel': {
                key: float(value)
                for key, value in case_thresholds.items()
            },
            'localization_metrics': {
                key: _localization_metric_summary(value)
                for key, value in case_localization.items()
            },
            'prediction_stats': {
                key: {
                    'mean': float(value.mean()),
                    'max': float(value.max()),
                    'p99': float(np.percentile(value, 99.0)),
                }
                for key, value in case_predictions.items()
            },
        }

    output_path = output_dir / 'figure5_edge_lattice_model_diagnostics.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    summary['output_path'] = str(output_path)
    (output_dir / 'figure5_edge_lattice_model_diagnostics.json').write_text(json.dumps(summary, indent=2))
    return output_path


def _spacing_variation_image(atom_sigma: float = 2.7) -> np.ndarray:
    from ase import Atoms

    shape = (360, 360)
    positions = []
    numbers = []
    for y, spacing in zip(np.linspace(32, 328, 8), [10.0, 10.5, 14.5, 20.5, 30.0, 45.0, 68.0, 104.0]):
        atom_count = max(3, int(round((shape[1] - 56) / spacing)) + 1)
        xs = np.linspace(28, shape[1] - 28, atom_count, dtype=np.float32)
        for x in xs:
            positions.append((float(x), float(y), 0.0))
            numbers.append(74)
    atoms = Atoms(numbers=numbers, positions=positions, cell=[shape[1], shape[0], 30.0], pbc=False)
    return _render_atoms_stem_panel(atoms, shape, seed=33, atom_sigma_range=(float(atom_sigma), float(atom_sigma)))


def _size_variation_image(atom_sigma_range: tuple[float, float] = (1.15, 6.15)) -> np.ndarray:
    from ase import Atoms

    shape = (360, 360)
    positions = []
    numbers = []
    row_specs = [6, 14, 22, 32, 42, 56, 74, 92]
    for y, atomic_number in zip(np.linspace(32, 328, 8), row_specs):
        for x in np.arange(34, shape[1] - 28, 26, dtype=np.float32):
            positions.append((float(x), float(y), 0.0))
            numbers.append(atomic_number)
    atoms = Atoms(numbers=numbers, positions=positions, cell=[shape[1], shape[0], 30.0], pbc=False)
    return _render_atoms_stem_panel(atoms, shape, seed=57, atom_sigma_range=atom_sigma_range)


def _read_sweep_rows(csv_path: Path) -> list[dict[str, float]]:
    with csv_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            numeric_row = {}
            for key, value in row.items():
                if value in {'', None}:
                    continue
                try:
                    numeric_row[key] = float(value)
                except ValueError:
                    continue
            rows.append(numeric_row)
        return rows


def _generate_missing_sweep_csv(args: argparse.Namespace) -> None:
    if args.sweep_csv.exists() and not args.regenerate_sweep:
        sweep_config_path = args.sweep_csv.parent / 'pixel_size_sweep_config.json'
        if sweep_config_path.is_file():
            try:
                sweep_config = json.loads(sweep_config_path.read_text())
            except json.JSONDecodeError:
                sweep_config = {}
            if sweep_config.get('training_parameter_source'):
                return
        print('Existing sweep CSV has no training-parameter metadata; regenerating it.', flush=True)
    elif args.sweep_csv.exists():
        print(f'Regenerating pixel-size sweep at {args.sweep_csv.parent}', flush=True)
    else:
        print(f'Missing sweep CSV; generating pixel-size sweep at {args.sweep_csv.parent}', flush=True)

    if args.sweep_csv.exists() and args.sweep_csv.name != 'pixel_size_metrics.csv':
        return
    if args.sweep_csv.name != 'pixel_size_metrics.csv':
        raise FileNotFoundError(
            f'Missing sweep CSV: {args.sweep_csv}. Automatic sweep generation expects '
            "the output filename to be 'pixel_size_metrics.csv'."
        )

    from scripts.run_pixel_size_sweep import DEFAULT_PIXEL_SIZE_FACTORS, DEFAULT_THRESHOLD_GRID, generate_pixel_size_sweep

    pixel_size_factors = getattr(args, 'sweep_pixel_size_factors', None) or DEFAULT_PIXEL_SIZE_FACTORS
    threshold_grid = getattr(args, 'sweep_threshold_grid', None) or DEFAULT_THRESHOLD_GRID

    sweep_args = argparse.Namespace(
        output_dir=args.sweep_csv.parent,
        checkpoint=args.checkpoint,
        device=args.device,
        seed=getattr(args, 'seed', 0),
        samples_per_size=args.sweep_samples,
        batch_size=args.batch_size,
        num_workers=0,
        dataset_config=args.sweep_dataset_config,
        height=None,
        width=None,
        train_pixel_size_angstrom=0.1062231596676199,
        pixel_size_factors=pixel_size_factors,
        pixel_sizes_angstrom=None,
        num_filters=args.num_filters,
        dropout=args.dropout,
        train_sigma_min=None,
        train_sigma_max=None,
        train_target_sigma=None,
        train_min_separation_range_min=None,
        train_min_separation_range_max=None,
        min_atoms=None,
        max_atoms=None,
        background_min=None,
        background_max=None,
        inhom_background_min=None,
        inhom_background_max=None,
        low_freq_noise_min=None,
        low_freq_noise_max=None,
        read_noise_min=None,
        read_noise_max=None,
        total_counts_min=None,
        total_counts_max=None,
        blur_sigma_min=None,
        blur_sigma_max=None,
        edge_padding=None,
        threshold_grid=threshold_grid,
        train_match_distance=None,
        train_peak_min_distance=None,
        train_peak_window_size=None,
        fixed_evaluation_pixels=False,
        example_size_count=None,
    )
    generated_path = generate_pixel_size_sweep(sweep_args)
    if generated_path != args.sweep_csv:
        raise FileNotFoundError(f'Expected sweep CSV at {args.sweep_csv}, generated {generated_path}')


def _training_pixel_size(rows: list[dict[str, float]]) -> float:
    by_factor = min(rows, key=lambda row: abs(row.get('pixel_size_factor', 0.0) - 1.0))
    return float(by_factor['pixel_size_angstrom'])


def _plot_pixel_size_sweep_stack(top_ax: plt.Axes, bottom_ax: plt.Axes, rows: list[dict[str, float]]) -> None:
    rows = sorted(rows, key=lambda row: row['pixel_size_angstrom'])
    pixel_size = np.asarray([row['pixel_size_angstrom'] for row in rows], dtype=np.float32)
    f1 = np.asarray([row['f1'] for row in rows], dtype=np.float32)
    rmse = np.asarray([row['rmse_px'] for row in rows], dtype=np.float32)
    feature_ratio = np.asarray([row['feature_fwhm_over_bottleneck_rf'] for row in rows], dtype=np.float32)
    spacing_ratio = np.asarray([row['spacing_over_bottleneck_rf'] for row in rows], dtype=np.float32)
    train_pixel_size = _training_pixel_size(rows)

    f1_color = '#2f7f73'
    rmse_color = '#b9653e'
    feature_color = '#426aa8'
    spacing_color = '#8a58a2'

    top_ax.plot(pixel_size, f1, color=f1_color, marker='o', linewidth=2.2, markersize=5.5)
    top_ax.axvline(train_pixel_size, color='black', linestyle='--', linewidth=1.35)
    top_ax.set_ylabel('Localization F1', fontsize=AXIS_LABEL_SIZE, color=f1_color)
    top_ax.tick_params(axis='y', colors=f1_color, labelsize=AXIS_TICK_SIZE)
    top_ax.tick_params(axis='x', labelbottom=False, labelsize=AXIS_TICK_SIZE)
    top_ax.set_ylim(0.0, 1.02)
    top_ax.grid(alpha=0.25)

    rmse_ax = top_ax.twinx()
    rmse_ax.plot(pixel_size, rmse, color=rmse_color, marker='s', linewidth=2.0, markersize=5.0)
    rmse_ax.set_ylabel('Localization RMSE (px)', fontsize=AXIS_LABEL_SIZE, color=rmse_color)
    rmse_ax.tick_params(axis='y', colors=rmse_color, labelsize=AXIS_TICK_SIZE)
    rmse_ax.set_ylim(0.0, max(float(rmse.max()) * 1.05, 0.5))

    bottom_ax.plot(pixel_size, feature_ratio, color=feature_color, marker='o', linewidth=2.2, markersize=5.5)
    bottom_ax.axvline(train_pixel_size, color='black', linestyle='--', linewidth=1.35)
    bottom_ax.set_xlabel('Assumed pixel size (angstrom / px)', fontsize=AXIS_LABEL_SIZE)
    bottom_ax.set_ylabel('Blob width / RF', fontsize=AXIS_LABEL_SIZE, color=feature_color)
    bottom_ax.tick_params(axis='both', labelsize=AXIS_TICK_SIZE)
    bottom_ax.tick_params(axis='y', colors=feature_color, labelsize=AXIS_TICK_SIZE)
    bottom_ax.set_ylim(0.0, 1.0)
    bottom_ax.grid(alpha=0.25)

    spacing_ax = bottom_ax.twinx()
    spacing_ax.plot(pixel_size, spacing_ratio, color=spacing_color, marker='s', linewidth=2.0, markersize=5.0)
    spacing_ax.set_ylabel('Atom spacing / RF', fontsize=AXIS_LABEL_SIZE, color=spacing_color)
    spacing_ax.tick_params(axis='y', colors=spacing_color, labelsize=AXIS_TICK_SIZE)
    spacing_ax.set_ylim(0.0, 1.0)


def make_figure_3(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    model = _load_blobnet_model(args.checkpoint, device, args.num_filters, args.dropout)
    _generate_missing_sweep_csv(args)
    rows = _read_sweep_rows(args.sweep_csv)
    feature_sigma_range = (float(args.feature_sigma_min), float(args.feature_sigma_max))
    spacing_image = _spacing_variation_image(atom_sigma=float(np.mean(feature_sigma_range)))
    size_image = _size_variation_image(atom_sigma_range=feature_sigma_range)
    spacing_prediction = _predict_array(model, spacing_image, device)
    size_prediction = _predict_array(model, size_image, device)

    fig = plt.figure(figsize=(14.0, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 0.04, 2.55], height_ratios=[1, 1], wspace=0.08, hspace=0.08)

    ax_spacing = fig.add_subplot(grid[0, 0])
    _plot_clean_image(ax_spacing, spacing_image, '')
    ax_size = fig.add_subplot(grid[0, 1])
    _plot_clean_image(ax_size, size_image, '')

    ax_spacing_prediction = fig.add_subplot(grid[1, 0])
    ax_spacing_prediction.imshow(spacing_prediction, cmap=MODEL_CMAPS['random'], vmin=0.0, vmax=max(float(spacing_prediction.max()), 1e-6))
    ax_spacing_prediction.set_xticks([])
    ax_spacing_prediction.set_yticks([])

    ax_size_prediction = fig.add_subplot(grid[1, 1])
    ax_size_prediction.imshow(size_prediction, cmap=MODEL_CMAPS['random'], vmin=0.0, vmax=max(float(size_prediction.max()), 1e-6))
    ax_size_prediction.set_xticks([])
    ax_size_prediction.set_yticks([])

    ax_top = fig.add_subplot(grid[0, 3])
    ax_bottom = fig.add_subplot(grid[1, 3], sharex=ax_top)
    _plot_pixel_size_sweep_stack(ax_top, ax_bottom, rows)

    output_path = output_dir / 'figure3_scale_spacing_robustness.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    return output_path


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/manuscript_figures'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--dpi', type=int, default=300)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--num-filters', type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--dropout', type=float, default=0.2)


def _add_dataset_config_arguments(parser: argparse.ArgumentParser) -> None:
    repo_root = _repo_root()
    parser.add_argument('--square-dataset-config', type=Path, default=repo_root / 'configs/dataset_configs/square.yaml')
    parser.add_argument('--hexagonal-dataset-config', type=Path, default=repo_root / 'configs/dataset_configs/hexagonal.yaml')
    parser.add_argument('--random-dataset-config', type=Path, default=repo_root / 'configs/dataset_configs/random.yaml')


def _add_feature_size_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--feature-sigma-min', type=float, default=2.6)
    parser.add_argument('--feature-sigma-max', type=float, default=3.2)


def _add_edge_localization_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--localization-threshold-rel', type=float, default=0.25)
    parser.add_argument('--localization-match-distance', type=float, default=3.0)
    parser.add_argument('--peak-min-distance', type=int, default=3)
    parser.add_argument('--peak-window-size', type=int, default=5)


def _parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(',') if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description='Build manuscript figures for BlobNet real-STEM utility.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    figure1 = subparsers.add_parser('figure1', help='Synthetic geometry comparison figure.')
    _add_shared_arguments(figure1)
    _add_model_arguments(figure1)
    _add_dataset_config_arguments(figure1)
    figure1.add_argument('--square-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/square/unet_best.pth')
    figure1.add_argument('--hexagonal-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/hexagonal/unet_best.pth')
    figure1.add_argument('--random-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/random/unet_best.pth')
    figure1.add_argument('--seed', type=int, default=0)
    figure1.add_argument('--offset-samples', type=int, default=32)
    figure1.add_argument('--batch-size', type=int, default=4)
    figure1.add_argument('--threshold-rel', type=float, default=0.35)
    figure1.add_argument('--match-distance', type=float, default=3.0)
    figure1.add_argument('--offset-range', type=float, default=2.0)
    figure1.add_argument('--offset-bins', type=int, default=48)
    figure1.set_defaults(func=make_figure_1)

    figure2 = subparsers.add_parser('figure2', help='Experimental HAADF input/output figure.')
    _add_shared_arguments(figure2)
    _add_model_arguments(figure2)
    figure2.add_argument('--checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/random/unet_best.pth')
    figure2.add_argument('--data-dir', type=Path, default=repo_root / 'experimental_data')
    figure2.add_argument('--tile-size', type=int, default=256)
    figure2.add_argument('--tile-overlap', type=int, default=64)
    figure2.add_argument('--batch-size', type=int, default=4)
    figure2.add_argument('--dog-small', type=float, default=1.0)
    figure2.add_argument('--dog-large', type=float, default=20.0)
    figure2.add_argument('--experimental-measurements', type=Path, default=repo_root / 'outputs/experimental_feature_measurements_local/experimental_feature_measurements.json')
    figure2.add_argument('--feature-match-sigma-px', type=float, default=2.9)
    figure2.add_argument('--experimental-crop-size', type=int, default=512)
    figure2.set_defaults(func=make_figure_2)

    figure3 = subparsers.add_parser('figure3', help='Scale and spacing robustness figure.')
    _add_shared_arguments(figure3)
    _add_model_arguments(figure3)
    _add_feature_size_arguments(figure3)
    figure3.add_argument('--checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/random/unet_best.pth')
    figure3.add_argument('--sweep-csv', type=Path, default=repo_root / 'outputs/blobnet_pixel_size_sweep_random_4x/pixel_size_metrics.csv')
    figure3.add_argument('--sweep-dataset-config', type=Path)
    figure3.add_argument('--sweep-samples', type=int, default=64)
    figure3.add_argument('--sweep-pixel-size-factors', type=_parse_float_list)
    figure3.add_argument('--sweep-threshold-grid', type=_parse_float_list)
    figure3.add_argument('--regenerate-sweep', action='store_true')
    figure3.add_argument('--batch-size', type=int, default=8)
    figure3.set_defaults(func=make_figure_3)

    figure4 = subparsers.add_parser('figure4', help='Simulated WS2 monolayer flake edge model comparison.')
    _add_shared_arguments(figure4)
    _add_model_arguments(figure4)
    _add_feature_size_arguments(figure4)
    figure4.add_argument('--square-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/square/unet_best.pth')
    figure4.add_argument('--hexagonal-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/hexagonal/unet_best.pth')
    figure4.add_argument('--random-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/random_dense/unet_best.pth')
    figure4.add_argument('--seed', type=int, default=41)
    _add_edge_localization_arguments(figure4)
    figure4.set_defaults(func=make_figure_4)

    figure5 = subparsers.add_parser('figure5', help='Edge-structure TP/FP/FN diagnostics for WS2, SrTiO3, and graphene.')
    _add_shared_arguments(figure5)
    _add_model_arguments(figure5)
    _add_feature_size_arguments(figure5)
    figure5.add_argument('--square-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/square/unet_best.pth')
    figure5.add_argument('--hexagonal-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/hexagonal/unet_best.pth')
    figure5.add_argument('--random-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/random_dense/unet_best.pth')
    figure5.add_argument('--seed', type=int, default=41)
    _add_edge_localization_arguments(figure5)
    figure5.set_defaults(func=make_figure_5)

    all_parser = subparsers.add_parser('all', help='Build all manuscript figures.')
    _add_shared_arguments(all_parser)
    _add_model_arguments(all_parser)
    _add_dataset_config_arguments(all_parser)
    _add_feature_size_arguments(all_parser)
    all_parser.add_argument('--square-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/square/unet_best.pth')
    all_parser.add_argument('--hexagonal-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/hexagonal/unet_best.pth')
    all_parser.add_argument('--random-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/random/unet_best.pth')
    all_parser.add_argument('--ws2-random-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/random_dense/unet_best.pth')
    all_parser.add_argument('--checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/random/unet_best.pth')
    all_parser.add_argument('--data-dir', type=Path, default=repo_root / 'experimental_data')
    all_parser.add_argument('--sweep-csv', type=Path, default=repo_root / 'outputs/blobnet_pixel_size_sweep_random_4x/pixel_size_metrics.csv')
    all_parser.add_argument('--sweep-dataset-config', type=Path)
    all_parser.add_argument('--sweep-samples', type=int, default=64)
    all_parser.add_argument('--sweep-pixel-size-factors', type=_parse_float_list)
    all_parser.add_argument('--sweep-threshold-grid', type=_parse_float_list)
    all_parser.add_argument('--regenerate-sweep', action='store_true')
    all_parser.add_argument('--seed', type=int, default=0)
    all_parser.add_argument('--offset-samples', type=int, default=32)
    all_parser.add_argument('--batch-size', type=int, default=4)
    all_parser.add_argument('--threshold-rel', type=float, default=0.35)
    all_parser.add_argument('--match-distance', type=float, default=3.0)
    all_parser.add_argument('--offset-range', type=float, default=2.0)
    all_parser.add_argument('--offset-bins', type=int, default=48)
    all_parser.add_argument('--tile-size', type=int, default=256)
    all_parser.add_argument('--tile-overlap', type=int, default=64)
    all_parser.add_argument('--dog-small', type=float, default=1.0)
    all_parser.add_argument('--dog-large', type=float, default=20.0)
    all_parser.add_argument('--experimental-measurements', type=Path, default=repo_root / 'outputs/experimental_feature_measurements_local/experimental_feature_measurements.json')
    all_parser.add_argument('--feature-match-sigma-px', type=float, default=2.9)
    all_parser.add_argument('--experimental-crop-size', type=int, default=512)
    _add_edge_localization_arguments(all_parser)
    all_parser.set_defaults(func=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == 'all':
        original_random_checkpoint = args.random_checkpoint
        paths = [
            make_figure_1(args),
            make_figure_2(args),
            make_figure_3(args),
        ]
        args.random_checkpoint = args.ws2_random_checkpoint
        paths.append(make_figure_4(args))
        paths.append(make_figure_5(args))
        args.random_checkpoint = original_random_checkpoint
    else:
        paths = [args.func(args)]
    for path in paths:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
