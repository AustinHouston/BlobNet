from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, gaussian_laplace, zoom
from scipy.optimize import linear_sum_assignment
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

FIGURE2_FIXED_NOISE_PARAMETERS = {
    'background_range': (0.054, 0.054),
    'gradient_range': (0.0, 0.0),
    'inhomogeneous_background_range': (0.050, 0.050),
    'inhomogeneous_background_sigma_fraction_range': (0.290, 0.290),
    'low_frequency_noise_range': (0.080, 0.080),
    'low_frequency_sigma_fraction_range': (0.095, 0.095),
    'read_noise_std_range': (0.065, 0.065),
    'blur_sigma_range': (0.500, 0.500),
}

FIGURE2_TUNED_THRESHOLDS = {
    'mos2_edge': 0.73,
    'srtio3_edge': 0.68,
    'graphene_rattled_edge': 0.785,
}

FIGURE2_THRESHOLD_NOTE = (
    'Figure 2 thresholds were selected from a count-64 fixed-noise threshold sweep. '
    'For each image row, the same threshold is applied to all three models; the chosen '
    'threshold maximized the random model TP/(FP+FN) margin over the strongest competing '
    'model after excluding predictions and atoms within 10 px of the image border.'
)

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

    def select_haadf(dataset: Any) -> np.ndarray | None:
        candidates = [
            (str(getattr(candidate, 'title', '')), candidate)
            for candidate in dataset.values()
            if len(tuple(dimension for dimension in getattr(candidate, 'shape', ()) if dimension != 1)) == 2
        ]
        for preferred_title in ('HAADF', 'Ref HAADF'):
            for title, candidate in candidates:
                if title == preferred_title:
                    return np.asarray(candidate, dtype=np.float32)
        for title, candidate in candidates:
            if 'HAADF' in title.upper():
                return np.asarray(candidate, dtype=np.float32)
        return None

    try:
        dataset = ft.open_file(str(path))
    except OSError:
        with tempfile.TemporaryDirectory(prefix='blobnet-emd-') as temp_dir:
            copied_path = Path(temp_dir) / path.name
            shutil.copyfile(path, copied_path)
            return select_haadf(ft.open_file(str(copied_path)))
    return select_haadf(dataset)


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
    with h5py.File(path, 'r') as handle:
        if 'image' in handle:
            return _normalize_image(np.asarray(handle['image'], dtype=np.float32).squeeze())
    image = _find_haadf_with_pytemlib(path)
    if image is None:
        image = _find_haadf_with_h5py(path)
    return _normalize_image(np.squeeze(image))


def _decode_velox_json(dataset: h5py.Dataset, index: int = 0) -> dict[str, Any]:
    raw = dataset[:, index] if dataset.ndim == 2 else dataset[index]
    if isinstance(raw, bytes):
        encoded = raw
    elif isinstance(raw, str):
        encoded = raw.encode('utf-8')
    else:
        encoded = np.asarray(raw, dtype=np.uint8).tobytes()
    return json.loads(encoded.split(b'\x00', 1)[0].decode('utf-8'))


def _load_velox_displayed_haadf(path: Path, crop_size: int) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Read the final displayed DCFI(HAADF), or HAADF fallback, from a Velox EMD."""
    with h5py.File(path, 'r') as handle:
        if 'image' in handle and 'pixel_size_nm' in handle['image'].attrs:
            full_image = np.asarray(handle['image'], dtype=np.float32).squeeze()
            image = _center_crop_or_pad(full_image, int(crop_size))
            return _normalize_image(image), float(handle['image'].attrs['pixel_size_nm']), {
                'display_label': str(handle.attrs.get('source_display_label', 'HAADF')),
                'data_path': '/image',
                'series_index': int(handle.attrs.get('source_series_index', 0)),
                'source_shape': list(full_image.shape),
                'selection': str(handle.attrs.get('selection', '')),
            }
        displays = handle.get('Presentation/Displays/ImageDisplay')
        if displays is None:
            raise ValueError(f'No Velox image displays found in {path}.')

        candidates: list[tuple[int, str, int, str]] = []
        for display_dataset in displays.values():
            display = _decode_velox_json(display_dataset)
            label = str(display.get('display', {}).get('label', ''))
            data_path = str(display.get('dataPath', ''))
            if not data_path or data_path.lstrip('/') not in handle:
                continue
            upper_label = label.upper()
            if 'DCFI' in upper_label and 'HAADF' in upper_label:
                priority = 0
            elif 'HAADF' in upper_label:
                priority = 1
            else:
                continue
            candidates.append((priority, data_path, int(display.get('seriesIndex', 0)), label))

        if not candidates:
            raise ValueError(f'No displayed HAADF dataset found in {path}.')
        _priority, data_path, series_index, display_label = min(candidates, key=lambda item: item[0])
        group = handle[data_path.lstrip('/')]
        data = group['Data']
        if data.ndim == 3:
            series_index = int(np.clip(series_index, 0, data.shape[2] - 1))
            height, width = int(data.shape[0]), int(data.shape[1])
            y0 = max((height - int(crop_size)) // 2, 0)
            x0 = max((width - int(crop_size)) // 2, 0)
            image = np.asarray(
                data[y0 : min(y0 + crop_size, height), x0 : min(x0 + crop_size, width), series_index],
                dtype=np.float32,
            )
        else:
            full_image = np.asarray(data, dtype=np.float32).squeeze()
            height, width = int(full_image.shape[0]), int(full_image.shape[1])
            image = _center_crop_or_pad(full_image, int(crop_size))

        metadata_index = min(series_index, group['Metadata'].shape[1] - 1)
        metadata = _decode_velox_json(group['Metadata'], metadata_index)
        binary_result = metadata['BinaryResult']
        pixel_width_nm = float(binary_result['PixelSize']['width']) * 1e9
        pixel_height_nm = float(binary_result['PixelSize']['height']) * 1e9
        if not np.isclose(pixel_width_nm, pixel_height_nm):
            raise ValueError(f'Non-square pixels in {path}: {pixel_width_nm} x {pixel_height_nm} nm.')

    image = _center_crop_or_pad(_normalize_image(image), int(crop_size))
    return image, float((pixel_width_nm + pixel_height_nm) / 2.0), {
        'display_label': display_label,
        'data_path': data_path,
        'series_index': series_index,
        'source_shape': [height, width],
    }


def _read_channel_pixel_size_nm(path: Path, channel: str = 'Channel_000') -> float:
    with h5py.File(path, 'r') as handle:
        if 'image' in handle and 'pixel_size_nm' in handle['image'].attrs:
            return float(handle['image'].attrs['pixel_size_nm'])

    import pyTEMlib.file_tools as ft

    def read_pixel_size(dataset: Any) -> float:
        binary_result = dataset[channel].original_metadata['BinaryResult']
        width_nm = float(binary_result['PixelSize']['width']) * 1e9
        height_nm = float(binary_result['PixelSize']['height']) * 1e9
        if not np.isclose(width_nm, height_nm):
            raise ValueError(f'Non-square pixels in {path}: {width_nm} x {height_nm} nm.')
        return float((width_nm + height_nm) / 2.0)

    try:
        return read_pixel_size(ft.open_file(str(path)))
    except OSError:
        with tempfile.TemporaryDirectory(prefix='blobnet-emd-') as temp_dir:
            copied_path = Path(temp_dir) / path.name
            shutil.copyfile(path, copied_path)
            return read_pixel_size(ft.open_file(str(copied_path)))


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


def _make_fixed_fov_resolution_view(
    image: np.ndarray,
    dog_small: float,
    dog_large: float,
    display_size: int,
    native_pixels: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    processed = gaussian_filter(image, dog_small) - gaussian_filter(image, dog_large)
    processed = _normalize_image(processed)
    display_view = _center_crop_or_pad(processed, int(display_size))
    native_view = _normalize_image(
        _interpolate_image(display_view, (int(native_pixels), int(native_pixels)))
    )
    return display_view, native_view, {
        'display_processing': 'DoG background-subtracted, fixed center FOV, resolution-matched',
        'display_pixels': int(display_size),
        'native_inference_pixels': int(native_pixels),
        'resolution_scale': float(native_pixels) / float(display_size),
    }


def _interpolate_image(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Linearly resample an image to a requested pixel grid with SciPy."""
    image = np.asarray(image, dtype=np.float32)
    source_height, source_width = image.shape
    output_height, output_width = (int(value) for value in output_shape)
    if output_height <= 0 or output_width <= 0:
        raise ValueError(f'Output shape must be positive, received {output_shape}.')

    source_y = np.linspace(0.0, source_height - 1.0, output_height, dtype=np.float64)
    source_x = np.linspace(0.0, source_width - 1.0, output_width, dtype=np.float64)
    query_y, query_x = np.meshgrid(source_y, source_x, indexing='ij')
    query_points = np.column_stack((query_y.ravel(), query_x.ravel()))
    interpolator = RegularGridInterpolator(
        (np.arange(source_height, dtype=np.float64), np.arange(source_width, dtype=np.float64)),
        image,
        method='linear',
        bounds_error=True,
    )
    return interpolator(query_points).reshape(output_height, output_width).astype(np.float32)


def _plot_clean_image(ax: plt.Axes, image: np.ndarray, title: str, cmap: str = 'gray') -> None:
    ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
    if title:
        ax.set_title(title, fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])


def _add_physical_scale_bar(
    ax: plt.Axes,
    image_shape: tuple[int, int],
    pixel_size_nm: float,
    length_nm: float,
    linewidth: float,
) -> None:
    """Add a lower-right scale bar whose displayed length follows the image metadata."""
    height, width = image_shape
    length_pixels = float(length_nm) / float(pixel_size_nm)
    x_right = width * 0.94
    x_left = x_right - length_pixels
    y = height * 0.965
    ax.plot(
        [x_left, x_right],
        [y, y],
        color='white',
        linewidth=linewidth,
        solid_capstyle='butt',
        zorder=10,
    )


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


def match_network_predictions(
    blob_positions_nm: np.ndarray,
    hex_positions_nm: np.ndarray,
    radius_nm: float,
) -> dict[str, np.ndarray]:
    """Maximum-cardinality one-to-one matching, then minimum total distance.

    KD-trees restrict candidates to the physical radius. Dummy assignment columns
    allow unmatched Blob-Net positions. Their cost exceeds the total possible
    distance cost, so gaining a valid pair always takes priority over distance.
    """
    if not np.isfinite(radius_nm) or radius_nm <= 0:
        raise ValueError('Matching radius must be positive and finite.')
    blob = np.asarray(blob_positions_nm, dtype=np.float64).reshape(-1, 2)
    hexnet = np.asarray(hex_positions_nm, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(blob).all() or not np.isfinite(hexnet).all():
        raise ValueError('Prediction coordinates must be finite.')
    pairs = np.empty((0, 2), dtype=np.int64)
    distances = np.empty(0, dtype=np.float64)
    if len(blob) and len(hexnet):
        penalty = float(min(len(blob), len(hexnet)) + 1)
        costs = np.full((len(blob), len(hexnet) + len(blob)), 2 * penalty)
        costs[:, len(hexnet):] = penalty
        neighbors = cKDTree(blob).query_ball_tree(cKDTree(hexnet), r=radius_nm)
        for row, columns in enumerate(neighbors):
            if columns:
                costs[row, columns] = np.linalg.norm(hexnet[columns] - blob[row], axis=1) / radius_nm
        rows, columns = linear_sum_assignment(costs)
        valid = (columns < len(hexnet)) & (costs[rows, columns] <= 1.0 + 1e-12)
        pairs = np.column_stack((rows[valid], columns[valid]))
        distances = np.linalg.norm(blob[pairs[:, 0]] - hexnet[pairs[:, 1]], axis=1)
    return {
        'pairs': pairs,
        'distances_nm': distances,
        'blob_only_indices': np.setdiff1d(np.arange(len(blob)), pairs[:, 0]),
        'hex_only_indices': np.setdiff1d(np.arange(len(hexnet)), pairs[:, 1]),
    }


def make_figure_3(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    model = _load_blobnet_model(args.checkpoint, device, args.num_filters, args.dropout)
    hex_checkpoint = getattr(
        args,
        'figure3_hexagonal_checkpoint',
        getattr(
            args,
            'hexagonal_checkpoint',
            Path(__file__).resolve().parents[1]
            / 'artifacts/manuscript_models/figure3_hexagonal/unet_best.pth',
        ),
    )
    hex_model = _load_blobnet_model(hex_checkpoint, device, args.num_filters, args.dropout)
    match_radius_nm = getattr(args, 'agreement_radius_nm', 0.06)
    category_colors = {
        'Both': getattr(args, 'both_color', MODEL_COLORS['square']),
        'Hex-Net only': getattr(args, 'hex_only_color', '#56B4E9'),
        'Blob-Net only': getattr(args, 'blob_only_color', '#D55E00'),
    }

    fourth_image = getattr(
        args, 'fourth_image',
        args.data_dir / 'high_angle_grain_boundary_monolayer_WS2.h5',
    )
    files = [
        ('MoS$_2$', args.data_dir / 'pristine_monolayer_MoS2.h5'),
        ('Twin boundary', args.data_dir / 'Sigma3_coherent_twin_grain_boundary_FCC_Al.h5'),
        ('WS$_2$ (0063)', fourth_image),
        ('Quasicrystal', args.quasicrystal_image),
    ]
    images: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    coordinates: list[np.ndarray] = []
    category_coordinates: list[dict[str, np.ndarray]] = []
    predictions: list[np.ndarray] = []
    for label, path in files:
        velox_selection = None
        if path == fourth_image:
            image, source_pixel_size_nm, velox_selection = _load_velox_displayed_haadf(
                path, args.experimental_crop_size,
            )
        else:
            image = _load_experimental_image(path)
            source_pixel_size_nm = _read_channel_pixel_size_nm(path)
        field_of_view_nm = float(args.experimental_crop_size) * source_pixel_size_nm
        native_pixels = max(1, int(round(field_of_view_nm / args.target_pixel_size_nm)))
        display_view, native_view, transform = _make_fixed_fov_resolution_view(
            image,
            dog_small=args.dog_small,
            dog_large=args.dog_large,
            display_size=args.experimental_crop_size,
            native_pixels=native_pixels,
        )
        native_prediction = _predict_tiled(
            model,
            native_view,
            device,
            args.tile_size,
            args.tile_overlap,
            args.batch_size,
        )
        native_coordinates = extract_subpixel_peak_positions(
            native_prediction,
            threshold_rel=args.localization_threshold_rel,
            min_distance=args.peak_min_distance,
            window_size=args.peak_window_size,
        )
        display_coordinates = np.asarray(native_coordinates, dtype=np.float32).copy()
        if len(display_coordinates):
            display_coordinates[:, 0] *= (display_view.shape[0] - 1) / max(native_view.shape[0] - 1, 1)
            display_coordinates[:, 1] *= (display_view.shape[1] - 1) / max(native_view.shape[1] - 1, 1)
        hexagonal_pixel_size_factor = (
            float(args.ws2_hexagonal_pixel_size_factor) if path.name == 'pristine_monolayer_MoS2.h5' else 1.0
        )
        hexagonal_target_pixel_size_nm = float(args.target_pixel_size_nm) * hexagonal_pixel_size_factor
        hexagonal_native_pixels = max(1, int(round(field_of_view_nm / hexagonal_target_pixel_size_nm)))
        hexagonal_native_view = (
            native_view if hexagonal_native_pixels == native_pixels
            else _normalize_image(_interpolate_image(
                display_view, (hexagonal_native_pixels, hexagonal_native_pixels),
            ))
        )
        hex_prediction = _predict_tiled(
            hex_model, hexagonal_native_view, device, args.tile_size, args.tile_overlap, args.batch_size,
        )
        hexagonal_threshold_rel = (
            float(args.ws2_hexagonal_threshold_rel)
            if path.name == 'pristine_monolayer_MoS2.h5' else float(args.localization_threshold_rel)
        )
        hex_coordinates = np.asarray(extract_subpixel_peak_positions(
            hex_prediction, threshold_rel=hexagonal_threshold_rel,
            min_distance=args.peak_min_distance, window_size=args.peak_window_size,
        ), dtype=np.float32).reshape(-1, 2)
        if len(hex_coordinates):
            hex_coordinates[:, 0] *= (display_view.shape[0] - 1) / max(hexagonal_native_view.shape[0] - 1, 1)
            hex_coordinates[:, 1] *= (display_view.shape[1] - 1) / max(hexagonal_native_view.shape[1] - 1, 1)
        blob_nm = display_coordinates.astype(np.float64) * source_pixel_size_nm
        hex_nm = hex_coordinates.astype(np.float64) * source_pixel_size_nm
        matching = match_network_predictions(blob_nm, hex_nm, match_radius_nm)
        pairs = matching['pairs']
        groups = {
            'Both': (display_coordinates[pairs[:, 0]] + hex_coordinates[pairs[:, 1]]) / 2,
            'Hex-Net only': hex_coordinates[matching['hex_only_indices']],
            'Blob-Net only': display_coordinates[matching['blob_only_indices']],
        }
        category_coordinates.append(groups)
        matching_summary = {
            'radius_nm': float(match_radius_nm),
            'method': 'KD-tree candidates; maximum-cardinality, minimum-total-distance one-to-one assignment.',
            'shared_marker_position': 'Pair midpoint; agreement is not ground-truth correctness.',
            'counts': {key: len(value) for key, value in groups.items()},
            'category_colors': category_colors,
            'category_markers': {
                'Both': 'open circle',
                'Hex-Net only': getattr(args, 'disagreement_marker', 'x'),
                'Blob-Net only': getattr(args, 'disagreement_marker', 'x'),
            },
            'pairs_blob_hex_indices': pairs.tolist(),
            'pair_distances_nm': matching['distances_nm'].tolist(),
            'category_coordinates_yx_display_pixels': {key: value.tolist() for key, value in groups.items()},
            'radius_sensitivity': [
                {'radius_nm': radius, 'both_count': len(match_network_predictions(blob_nm, hex_nm, radius)['pairs'])}
                for radius in (0.04, 0.06, 0.08)
            ],
        }
        prediction = _interpolate_image(native_prediction, display_view.shape)
        transform.update(
            {
                'source_pixel_size_nm': source_pixel_size_nm,
                'target_pixel_size_nm': float(args.target_pixel_size_nm),
                'field_of_view_nm': field_of_view_nm,
                'predicted_atom_count': int(len(display_coordinates)),
                'peak_coordinate_system': 'Peaks found on native NN output and mapped to the 512 px display FOV.',
                'source_image': str(path),
                'velox_selection': velox_selection,
                'localization_threshold_rel': float(args.localization_threshold_rel),
                'checkpoint': str(args.checkpoint),
                'hexagonal_checkpoint': str(hex_checkpoint),
                'hexagonal_pixel_size_factor': hexagonal_pixel_size_factor,
                'hexagonal_target_pixel_size_nm': hexagonal_target_pixel_size_nm,
                'hexagonal_native_inference_pixels': hexagonal_native_pixels,
                'hexagonal_localization_threshold_rel': hexagonal_threshold_rel,
                'hexagonal_predicted_atom_count': int(len(hex_coordinates)),
                'matching': matching_summary,
            }
        )
        images.append((label, display_view, transform))
        coordinates.append(display_coordinates)
        predictions.append(prediction)

    fig, axes = plt.subplots(2, len(images), figsize=(4 * len(images), 7.4), constrained_layout=True)
    for col, ((label, image, _transform), groups) in enumerate(zip(images, category_coordinates)):
        _plot_clean_image(axes[0, col], image, label, cmap='gray')
        _plot_clean_image(axes[1, col], image, '', cmap='gray')
        disagreement_marker = getattr(args, 'disagreement_marker', 'x')
        for category, atom_coordinates in groups.items():
            if len(atom_coordinates):
                if category == 'Both':
                    style = {'marker': 'o', 'facecolors': 'none', 'edgecolors': category_colors[category]}
                elif disagreement_marker == 's':
                    style = {'marker': 's', 'facecolors': 'none', 'edgecolors': category_colors[category]}
                else:
                    style = {'marker': disagreement_marker, 'color': category_colors[category]}
                marker_size = args.marker_size
                if category != 'Both':
                    marker_size *= getattr(args, 'disagreement_marker_size_scale', 1.2)
                axes[1, col].scatter(
                    atom_coordinates[:, 1], atom_coordinates[:, 0], s=marker_size,
                    linewidths=args.marker_linewidth, alpha=0.96, **style,
                )
        for row in range(2):
            _add_physical_scale_bar(
                axes[row, col],
                image.shape,
                pixel_size_nm=float(_transform['source_pixel_size_nm']),
                length_nm=float(args.scale_bar_length_nm),
                linewidth=float(args.scale_bar_linewidth),
            )

    fig.legend(
        handles=[Line2D([0], [0], linestyle='none',
                        marker='o' if category == 'Both' else disagreement_marker,
                        markerfacecolor='none', markeredgecolor=color, markeredgewidth=args.marker_linewidth,
                        markersize=8 if category == 'Both' else 8.8,
                        label=category) for category, color in category_colors.items()],
        loc='upper center', bbox_to_anchor=(0.5, 1.065), ncol=3, frameon=False, fontsize=AXIS_LABEL_SIZE,
    )
    output_path = output_dir / getattr(args, 'output_name', 'figure3_experimental_haadf_outputs.png')
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    if args.save_pdf:
        figure_pdf_path = output_dir / 'figure3_experimental_haadf_outputs.pdf'
        fig.savefig(figure_pdf_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    palette_options = [
        ('A - Current', {'Both': '#2f69bf', 'Hex-Net only': '#56B4E9', 'Blob-Net only': '#D55E00'}),
        ('B - Green / cyan / vermillion', {'Both': '#009E73', 'Hex-Net only': '#56B4E9', 'Blob-Net only': '#D55E00'}),
        ('C - Yellow / blue / vermillion', {'Both': '#F0E442', 'Hex-Net only': '#0072B2', 'Blob-Net only': '#D55E00'}),
        ('D - Manuscript model palette', {'Both': '#2f8f4e', 'Hex-Net only': '#dd7a1f', 'Blob-Net only': '#2f69bf'}),
    ]
    preview_index = next(i for i, (label, _image, _transform) in enumerate(images) if '0063' in label)
    preview_image = images[preview_index][1]
    preview_groups = category_coordinates[preview_index]
    palette_fig, palette_axes = plt.subplots(2, 2, figsize=(11, 11), constrained_layout=True)
    for axis, (palette_label, palette) in zip(palette_axes.reshape(-1), palette_options):
        _plot_clean_image(axis, preview_image, palette_label, cmap='gray')
        for category, atom_coordinates in preview_groups.items():
            if not len(atom_coordinates):
                continue
            style = (
                {'marker': 'o', 'facecolors': 'none', 'edgecolors': palette[category]}
                if category == 'Both' else {'marker': 'x', 'color': palette[category]}
            )
            axis.scatter(
                atom_coordinates[:, 1], atom_coordinates[:, 0], s=args.marker_size,
                linewidths=args.marker_linewidth, alpha=0.96, **style,
            )
        _add_physical_scale_bar(
            axis, preview_image.shape,
            pixel_size_nm=float(images[preview_index][2]['source_pixel_size_nm']),
            length_nm=float(args.scale_bar_length_nm), linewidth=float(args.scale_bar_linewidth),
        )
        axis.legend(
            handles=[Line2D([0], [0], linestyle='none', marker='o' if category == 'Both' else 'x',
                            markerfacecolor='none', markeredgecolor=color,
                            markeredgewidth=args.marker_linewidth, markersize=7, label=category)
                     for category, color in palette.items()],
            loc='lower left', ncol=1, frameon=True, facecolor='black', edgecolor='none',
            framealpha=0.55, labelcolor='white', fontsize=8,
        )
    palette_path = output_dir / 'figure3_experimental_palette_options.png'
    palette_fig.savefig(palette_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(palette_fig)
    (output_dir / 'figure3_experimental_palette_options.json').write_text(json.dumps({
        'preview_image': images[preview_index][0],
        'marker_linewidth_points': float(args.marker_linewidth),
        'options': [{'label': label, 'colors': colors} for label, colors in palette_options],
    }, indent=2))

    summary = [
        {
            'label': label,
            'shape_y': int(image.shape[0]),
            'shape_x': int(image.shape[1]),
            **transform,
            'output_mean': float(prediction.mean()),
            'output_max': float(prediction.max()),
            'scale_bar_length_nm': float(args.scale_bar_length_nm),
        }
        for (label, image, transform), prediction in zip(images, predictions)
    ]
    (output_dir / 'figure3_experimental_haadf_outputs.json').write_text(json.dumps(summary, indent=2))
    return output_path


def make_figure_3b(args: argparse.Namespace) -> Path:
    """Compare Blob-Net and hexagonal-model atom localizations on Figure 3 images."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    models = [
        (
            'Blob-Net',
            _load_blobnet_model(args.blobnet_checkpoint, device, args.num_filters, args.dropout),
            args.blobnet_marker_color,
        ),
        (
            'Hexagonal model',
            _load_blobnet_model(args.hexagonal_checkpoint, device, args.num_filters, args.dropout),
            args.hexagonal_marker_color,
        ),
    ]
    files = [
        ('MoS$_2$', args.data_dir / 'pristine_monolayer_MoS2.h5'),
        ('Quasicrystal', args.quasicrystal_image),
        ('Twin boundary', args.data_dir / 'Sigma3_coherent_twin_grain_boundary_FCC_Al.h5'),
    ]

    images: list[tuple[str, np.ndarray, np.ndarray, dict[str, Any]]] = []
    for label, path in files:
        image = _load_experimental_image(path)
        source_pixel_size_nm = _read_channel_pixel_size_nm(path)
        field_of_view_nm = float(args.experimental_crop_size) * source_pixel_size_nm
        native_pixels = max(1, int(round(field_of_view_nm / args.target_pixel_size_nm)))
        display_view, native_view, transform = _make_fixed_fov_resolution_view(
            image,
            dog_small=args.dog_small,
            dog_large=args.dog_large,
            display_size=args.experimental_crop_size,
            native_pixels=native_pixels,
        )
        transform.update(
            {
                'source_image': str(path),
                'source_pixel_size_nm': source_pixel_size_nm,
                'target_pixel_size_nm': float(args.target_pixel_size_nm),
                'field_of_view_nm': field_of_view_nm,
            }
        )
        images.append((label, display_view, native_view, transform))

    model_coordinates: dict[str, list[np.ndarray]] = {}
    model_records: dict[str, list[dict[str, Any]]] = {}
    for model_label, model, marker_color in models:
        coordinates_for_model: list[np.ndarray] = []
        records_for_model: list[dict[str, Any]] = []
        for (_label, display_view, native_view, transform) in images:
            native_prediction = _predict_tiled(
                model,
                native_view,
                device,
                args.tile_size,
                args.tile_overlap,
                args.batch_size,
            )
            native_coordinates = extract_subpixel_peak_positions(
                native_prediction,
                threshold_rel=args.localization_threshold_rel,
                min_distance=args.peak_min_distance,
                window_size=args.peak_window_size,
            )
            display_coordinates = np.asarray(native_coordinates, dtype=np.float32).copy()
            if len(display_coordinates):
                display_coordinates[:, 0] *= (display_view.shape[0] - 1) / max(native_view.shape[0] - 1, 1)
                display_coordinates[:, 1] *= (display_view.shape[1] - 1) / max(native_view.shape[1] - 1, 1)
            coordinates_for_model.append(display_coordinates)
            records_for_model.append(
                {
                    'source_image': transform['source_image'],
                    'predicted_atom_count': int(len(display_coordinates)),
                    'prediction_mean': float(native_prediction.mean()),
                    'prediction_max': float(native_prediction.max()),
                }
            )
        model_coordinates[model_label] = coordinates_for_model
        model_records[model_label] = records_for_model

    fig, axes = plt.subplots(3, 3, figsize=(12, 10.8), constrained_layout=True)
    for col, (label, image, _native_view, transform) in enumerate(images):
        _plot_clean_image(axes[0, col], image, label, cmap='gray')
        _add_physical_scale_bar(
            axes[0, col],
            image.shape,
            pixel_size_nm=float(transform['source_pixel_size_nm']),
            length_nm=float(args.scale_bar_length_nm),
            linewidth=float(args.scale_bar_linewidth),
        )
        for row, (model_label, _model, marker_color) in enumerate(models, start=1):
            _plot_clean_image(axes[row, col], image, '', cmap='gray')
            atom_coordinates = model_coordinates[model_label][col]
            if len(atom_coordinates):
                axes[row, col].scatter(
                    atom_coordinates[:, 1],
                    atom_coordinates[:, 0],
                    s=args.marker_size,
                    facecolors=marker_color,
                    edgecolors=args.marker_edge_color,
                    linewidths=args.marker_linewidth,
                    alpha=0.96,
                )
            _add_physical_scale_bar(
                axes[row, col],
                image.shape,
                pixel_size_nm=float(transform['source_pixel_size_nm']),
                length_nm=float(args.scale_bar_length_nm),
                linewidth=float(args.scale_bar_linewidth),
            )

    axes[1, 0].set_ylabel('Blob-Net', fontsize=AXIS_LABEL_SIZE)
    axes[2, 0].set_ylabel('Hexagonal model', fontsize=AXIS_LABEL_SIZE)

    output_path = output_dir / 'fig-Blob-Net-3b.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'figure': str(output_path),
        'field_of_view': 'Preserved independently for each image.',
        'target_pixel_size_nm': float(args.target_pixel_size_nm),
        'scale_bar_length_nm': float(args.scale_bar_length_nm),
        'marker_size_points_squared': float(args.marker_size),
        'models': {
            'Blob-Net': {
                'checkpoint': str(args.blobnet_checkpoint),
                'marker_color': args.blobnet_marker_color,
                'records': model_records['Blob-Net'],
            },
            'Hexagonal model': {
                'checkpoint': str(args.hexagonal_checkpoint),
                'marker_color': args.hexagonal_marker_color,
                'records': model_records['Hexagonal model'],
            },
        },
        'images': [
            {
                'label': label,
                **transform,
            }
            for label, _image, _native_view, transform in images
        ],
    }
    (output_dir / 'fig-Blob-Net-3b.json').write_text(json.dumps(summary, indent=2))
    return output_path


def make_figure_3c(args: argparse.Namespace) -> Path:
    """Run the Figure 3b comparison on displayed multi-frame Velox HAADF images."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    models = [
        (
            'Blob-Net',
            _load_blobnet_model(args.blobnet_checkpoint, device, args.num_filters, args.dropout),
            args.blobnet_marker_color,
        ),
        (
            'Hexagonal model',
            _load_blobnet_model(args.hexagonal_checkpoint, device, args.num_filters, args.dropout),
            args.hexagonal_marker_color,
        ),
    ]

    images: list[tuple[str, np.ndarray, np.ndarray, dict[str, Any]]] = []
    for path in args.images:
        image, source_pixel_size_nm, velox_selection = _load_velox_displayed_haadf(
            path,
            args.experimental_crop_size,
        )
        field_of_view_nm = float(args.experimental_crop_size) * source_pixel_size_nm
        native_pixels = max(1, int(round(field_of_view_nm / args.target_pixel_size_nm)))
        display_view, native_view, transform = _make_fixed_fov_resolution_view(
            image,
            dog_small=args.dog_small,
            dog_large=args.dog_large,
            display_size=args.experimental_crop_size,
            native_pixels=native_pixels,
        )
        transform.update(
            {
                'source_image': str(path),
                'source_pixel_size_nm': source_pixel_size_nm,
                'target_pixel_size_nm': float(args.target_pixel_size_nm),
                'field_of_view_nm': field_of_view_nm,
                'velox_selection': velox_selection,
            }
        )
        label = path.name.split(' - ')[0]
        np.savez_compressed(output_dir / f'{label}_processed_inputs.npz', display=display_view, inference=native_view)
        transform['effective_inference_pixel_size_nm'] = source_pixel_size_nm * (display_view.shape[0] - 1) / max(native_pixels - 1, 1)
        images.append((label, display_view, native_view, transform))

    model_coordinates: dict[str, list[np.ndarray]] = {}
    model_records: dict[str, list[dict[str, Any]]] = {}
    for model_label, model, _marker_color in models:
        coordinates_for_model: list[np.ndarray] = []
        records_for_model: list[dict[str, Any]] = []
        for _label, display_view, native_view, transform in images:
            native_prediction = _predict_tiled(
                model,
                native_view,
                device,
                args.tile_size,
                args.tile_overlap,
                args.batch_size,
            )
            native_coordinates = extract_subpixel_peak_positions(
                native_prediction,
                threshold_rel=args.localization_threshold_rel,
                min_distance=args.peak_min_distance,
                window_size=args.peak_window_size,
            )
            display_coordinates = np.asarray(native_coordinates, dtype=np.float32).copy()
            if len(display_coordinates):
                display_coordinates[:, 0] *= (display_view.shape[0] - 1) / max(native_view.shape[0] - 1, 1)
                display_coordinates[:, 1] *= (display_view.shape[1] - 1) / max(native_view.shape[1] - 1, 1)
            np.savez_compressed(output_dir / f'{_label}_{model_label.replace(" ", "_")}_predictions.npz', prediction=native_prediction, coordinates_yx_inference=native_coordinates, coordinates_yx_display=display_coordinates, coordinates_yx_nm=display_coordinates * transform['source_pixel_size_nm'])
            coordinates_for_model.append(display_coordinates)
            records_for_model.append(
                {
                    'source_image': transform['source_image'],
                    'predicted_atom_count': int(len(display_coordinates)),
                    'prediction_mean': float(native_prediction.mean()),
                    'prediction_max': float(native_prediction.max()),
                }
            )
        model_coordinates[model_label] = coordinates_for_model
        model_records[model_label] = records_for_model

    fig, axes = plt.subplots(3, len(images), figsize=(3.95 * len(images), 10.8), constrained_layout=True, squeeze=False)
    for col, (label, image, _native_view, transform) in enumerate(images):
        _plot_clean_image(axes[0, col], image, label, cmap='gray')
        _add_physical_scale_bar(
            axes[0, col],
            image.shape,
            pixel_size_nm=float(transform['source_pixel_size_nm']),
            length_nm=float(args.scale_bar_length_nm),
            linewidth=float(args.scale_bar_linewidth),
        )
        for row, (model_label, _model, marker_color) in enumerate(models, start=1):
            _plot_clean_image(axes[row, col], image, '', cmap='gray')
            atom_coordinates = model_coordinates[model_label][col]
            if len(atom_coordinates):
                axes[row, col].scatter(
                    atom_coordinates[:, 1],
                    atom_coordinates[:, 0],
                    s=args.marker_size,
                    facecolors=marker_color,
                    edgecolors=args.marker_edge_color,
                    linewidths=args.marker_linewidth,
                    alpha=0.96,
                )
            _add_physical_scale_bar(
                axes[row, col],
                image.shape,
                pixel_size_nm=float(transform['source_pixel_size_nm']),
                length_nm=float(args.scale_bar_length_nm),
                linewidth=float(args.scale_bar_linewidth),
            )

    axes[1, 0].set_ylabel('Blob-Net', fontsize=AXIS_LABEL_SIZE)
    axes[2, 0].set_ylabel('Hexagonal model', fontsize=AXIS_LABEL_SIZE)

    output_path = output_dir / 'fig-Blob-Net-3c.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'figure': str(output_path),
        'image_selection': 'Final displayed DCFI(HAADF) series frame, with HAADF fallback.',
        'field_of_view': 'Centered 512 x 512 source-pixel FOV preserved independently for each image.',
        'target_pixel_size_nm': float(args.target_pixel_size_nm),
        'scale_bar_length_nm': float(args.scale_bar_length_nm),
        'marker_size_points_squared': float(args.marker_size),
        'models': {
            'Blob-Net': {
                'checkpoint': str(args.blobnet_checkpoint),
                'marker_color': args.blobnet_marker_color,
                'records': model_records['Blob-Net'],
            },
            'Hexagonal model': {
                'checkpoint': str(args.hexagonal_checkpoint),
                'marker_color': args.hexagonal_marker_color,
                'records': model_records['Hexagonal model'],
            },
        },
        'images': [
            {
                'label': label,
                **transform,
            }
            for label, _image, _native_view, transform in images
        ],
    }
    (output_dir / 'fig-Blob-Net-3c.json').write_text(json.dumps(summary, indent=2))
    return output_path


def make_figure_3c_cutoff_sweep(args: argparse.Namespace) -> Path:
    """Sweep peak cutoffs for both Figure 3C models on one Velox HAADF image."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    models = [
        (
            'Blob-Net',
            _load_blobnet_model(args.blobnet_checkpoint, device, args.num_filters, args.dropout),
            args.blobnet_marker_color,
        ),
        (
            'Hexagonal model',
            _load_blobnet_model(args.hexagonal_checkpoint, device, args.num_filters, args.dropout),
            args.hexagonal_marker_color,
        ),
    ]

    if getattr(args, 'standard_emd', False):
        image = _load_experimental_image(args.image)
        source_pixel_size_nm = _read_channel_pixel_size_nm(args.image)
        velox_selection = None
    else:
        image, source_pixel_size_nm, velox_selection = _load_velox_displayed_haadf(
            args.image,
            args.experimental_crop_size,
        )
    image_label = args.image.stem.split(' - ')[0].replace(' ', '-').replace('_', '-')
    field_of_view_nm = float(args.experimental_crop_size) * source_pixel_size_nm
    native_pixels = max(1, int(round(field_of_view_nm / args.target_pixel_size_nm)))
    display_view, native_view, transform = _make_fixed_fov_resolution_view(
        image,
        dog_small=args.dog_small,
        dog_large=args.dog_large,
        display_size=args.experimental_crop_size,
        native_pixels=native_pixels,
    )
    cutoffs = np.linspace(args.cutoff_min, args.cutoff_max, args.steps, dtype=np.float64)

    coordinates: dict[str, list[np.ndarray]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    for model_label, model, _marker_color in models:
        native_prediction = _predict_tiled(
            model,
            native_view,
            device,
            args.tile_size,
            args.tile_overlap,
            args.batch_size,
        )
        model_coordinates: list[np.ndarray] = []
        model_records: list[dict[str, Any]] = []
        for cutoff in cutoffs:
            native_coordinates = extract_subpixel_peak_positions(
                native_prediction,
                threshold_rel=float(cutoff),
                min_distance=args.peak_min_distance,
                window_size=args.peak_window_size,
            )
            display_coordinates = np.asarray(native_coordinates, dtype=np.float32).copy()
            if len(display_coordinates):
                display_coordinates[:, 0] *= (display_view.shape[0] - 1) / max(native_view.shape[0] - 1, 1)
                display_coordinates[:, 1] *= (display_view.shape[1] - 1) / max(native_view.shape[1] - 1, 1)
            model_coordinates.append(display_coordinates)
            model_records.append(
                {
                    'cutoff': float(cutoff),
                    'predicted_atom_count': int(len(display_coordinates)),
                }
            )
        coordinates[model_label] = model_coordinates
        records[model_label] = model_records

    fig = plt.figure(figsize=(3.0 + 2.25 * len(cutoffs), 5.5), constrained_layout=True)
    grid = fig.add_gridspec(2, len(cutoffs) + 1, width_ratios=[1.14] + [1.0] * len(cutoffs))
    input_axis = fig.add_subplot(grid[:, 0])
    _plot_clean_image(input_axis, display_view, args.image.name.split(' - ')[0], cmap='gray')
    _add_physical_scale_bar(
        input_axis,
        display_view.shape,
        pixel_size_nm=source_pixel_size_nm,
        length_nm=float(args.scale_bar_length_nm),
        linewidth=float(args.scale_bar_linewidth),
    )

    for row, (model_label, _model, marker_color) in enumerate(models):
        for col, cutoff in enumerate(cutoffs, start=1):
            axis = fig.add_subplot(grid[row, col])
            _plot_clean_image(axis, display_view, '', cmap='gray')
            atom_coordinates = coordinates[model_label][col - 1]
            if len(atom_coordinates):
                axis.scatter(
                    atom_coordinates[:, 1],
                    atom_coordinates[:, 0],
                    s=args.marker_size,
                    facecolors=marker_color,
                    edgecolors=args.marker_edge_color,
                    linewidths=args.marker_linewidth,
                    alpha=0.96,
                )
            _add_physical_scale_bar(
                axis,
                display_view.shape,
                pixel_size_nm=source_pixel_size_nm,
                length_nm=float(args.scale_bar_length_nm),
                linewidth=float(args.scale_bar_linewidth),
            )
            if row == 0:
                axis.set_title(f'Cutoff {cutoff:.2f}', fontsize=11)
            axis.text(
                0.04,
                0.04,
                f'n={len(atom_coordinates)}',
                transform=axis.transAxes,
                color='white',
                fontsize=8,
                ha='left',
                va='bottom',
                bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': 0.45, 'pad': 1.2},
            )
            if col == 1:
                axis.set_ylabel(model_label, fontsize=AXIS_LABEL_SIZE)

    output_path = output_dir / 'fig-Blob-Net-3c-0063-cutoff-sweep.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'figure': str(output_path),
        'source_image': str(args.image),
        'image_selection': 'Final displayed DCFI(HAADF) series frame.',
        'velox_selection': velox_selection,
        'source_pixel_size_nm': source_pixel_size_nm,
        'field_of_view_nm': field_of_view_nm,
        'target_pixel_size_nm': float(args.target_pixel_size_nm),
        'native_inference_pixels': native_pixels,
        'cutoff_range': [float(args.cutoff_min), float(args.cutoff_max)],
        'steps': int(args.steps),
        'models': {
            'Blob-Net': {
                'checkpoint': str(args.blobnet_checkpoint),
                'marker_color': args.blobnet_marker_color,
                'records': records['Blob-Net'],
            },
            'Hexagonal model': {
                'checkpoint': str(args.hexagonal_checkpoint),
                'marker_color': args.hexagonal_marker_color,
                'records': records['Hexagonal model'],
            },
        },
        'processing': transform,
    }
    (output_dir / 'fig-Blob-Net-3c-0063-cutoff-sweep.json').write_text(json.dumps(summary, indent=2))
    return output_path


def make_figure_3c_agreement_cutoff_sweep(args: argparse.Namespace) -> Path:
    """Sweep both model cutoffs and classify their one-to-one agreement."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    blob_model = _load_blobnet_model(args.blobnet_checkpoint, device, args.num_filters, args.dropout)
    hex_model = _load_blobnet_model(args.hexagonal_checkpoint, device, args.num_filters, args.dropout)
    if getattr(args, 'standard_emd', False):
        image = _load_experimental_image(args.image)
        source_pixel_size_nm = _read_channel_pixel_size_nm(args.image)
        velox_selection = None
    else:
        image, source_pixel_size_nm, velox_selection = _load_velox_displayed_haadf(
            args.image, args.experimental_crop_size,
        )
    field_of_view_nm = float(args.experimental_crop_size) * source_pixel_size_nm
    native_pixels = max(1, int(round(field_of_view_nm / args.target_pixel_size_nm)))
    display_view, native_view, transform = _make_fixed_fov_resolution_view(
        image, dog_small=args.dog_small, dog_large=args.dog_large,
        display_size=args.experimental_crop_size, native_pixels=native_pixels,
    )
    blob_prediction = _predict_tiled(
        blob_model, native_view, device, args.tile_size, args.tile_overlap, args.batch_size,
    )
    hex_prediction = _predict_tiled(
        hex_model, native_view, device, args.tile_size, args.tile_overlap, args.batch_size,
    )
    cutoffs = (
        np.asarray(args.cutoffs, dtype=np.float64)
        if args.cutoffs is not None
        else np.linspace(args.cutoff_min, args.cutoff_max, args.steps, dtype=np.float64)
    )
    category_colors = {
        'Both': args.both_color,
        'Hex-Net only': args.hex_only_color,
        'Blob-Net only': args.blob_only_color,
    }
    records: list[dict[str, Any]] = []
    category_coordinates: list[dict[str, np.ndarray]] = []
    for cutoff in cutoffs:
        model_coordinates: list[np.ndarray] = []
        for model_index, prediction in enumerate((blob_prediction, hex_prediction)):
            model_cutoff = (
                float(args.fixed_blobnet_cutoff)
                if model_index == 0 and args.fixed_blobnet_cutoff is not None
                else float(cutoff)
            )
            native_coordinates = np.asarray(extract_subpixel_peak_positions(
                prediction, threshold_rel=model_cutoff, min_distance=args.peak_min_distance,
                window_size=args.peak_window_size,
            ), dtype=np.float32).reshape(-1, 2)
            display_coordinates = native_coordinates.copy()
            if len(display_coordinates):
                display_coordinates[:, 0] *= (display_view.shape[0] - 1) / max(native_view.shape[0] - 1, 1)
                display_coordinates[:, 1] *= (display_view.shape[1] - 1) / max(native_view.shape[1] - 1, 1)
            model_coordinates.append(display_coordinates)
        blob_coordinates, hex_coordinates = model_coordinates
        matching = match_network_predictions(
            blob_coordinates.astype(np.float64) * source_pixel_size_nm,
            hex_coordinates.astype(np.float64) * source_pixel_size_nm,
            args.agreement_radius_nm,
        )
        pairs = matching['pairs']
        groups = {
            'Both': (blob_coordinates[pairs[:, 0]] + hex_coordinates[pairs[:, 1]]) / 2,
            'Hex-Net only': hex_coordinates[matching['hex_only_indices']],
            'Blob-Net only': blob_coordinates[matching['blob_only_indices']],
        }
        category_coordinates.append(groups)
        records.append({
            'cutoff': float(cutoff),
            'blobnet_cutoff': (
                float(cutoff) if args.fixed_blobnet_cutoff is None else float(args.fixed_blobnet_cutoff)
            ),
            'hexnet_cutoff': float(cutoff),
            'blob_count': int(len(blob_coordinates)),
            'hex_count': int(len(hex_coordinates)),
            'category_counts': {key: int(len(value)) for key, value in groups.items()},
            'pair_distances_nm': matching['distances_nm'].tolist(),
        })

    panel_count = len(cutoffs) + 1
    columns = 4
    rows = int(math.ceil(panel_count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.35 * columns, 3.35 * rows), constrained_layout=True)
    flat_axes = np.asarray(axes).reshape(-1)
    _plot_clean_image(flat_axes[0], display_view, 'Input', cmap='gray')
    _add_physical_scale_bar(
        flat_axes[0], display_view.shape, pixel_size_nm=source_pixel_size_nm,
        length_nm=float(args.scale_bar_length_nm), linewidth=float(args.scale_bar_linewidth),
    )
    for axis, cutoff, groups, record in zip(flat_axes[1:], cutoffs, category_coordinates, records):
        title = f'Cutoff {cutoff:.2f}'
        if args.fixed_blobnet_cutoff is not None:
            title = f'Hex cutoff {cutoff:.2f}\nBlob cutoff {args.fixed_blobnet_cutoff:.2f}'
        _plot_clean_image(axis, display_view, title, cmap='gray')
        for category, atom_coordinates in groups.items():
            if not len(atom_coordinates):
                continue
            style = (
                {'marker': 'o', 'facecolors': 'none', 'edgecolors': category_colors[category]}
                if category == 'Both' else {'marker': 'x', 'color': category_colors[category]}
            )
            axis.scatter(
                atom_coordinates[:, 1], atom_coordinates[:, 0], s=args.marker_size,
                linewidths=args.marker_linewidth, alpha=0.96, **style,
            )
        _add_physical_scale_bar(
            axis, display_view.shape, pixel_size_nm=source_pixel_size_nm,
            length_nm=float(args.scale_bar_length_nm), linewidth=float(args.scale_bar_linewidth),
        )
        counts = record['category_counts']
        axis.text(
            0.03, 0.035,
            f"Both {counts['Both']} | H {counts['Hex-Net only']} | B {counts['Blob-Net only']}",
            transform=axis.transAxes, color='white', fontsize=7.5, ha='left', va='bottom',
            bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': 0.48, 'pad': 1.2},
        )
    for axis in flat_axes[panel_count:]:
        axis.axis('off')
    fig.legend(
        handles=[Line2D([0], [0], linestyle='none', marker='o' if category == 'Both' else 'x',
                        markerfacecolor='none', markeredgecolor=color,
                        markeredgewidth=args.marker_linewidth, markersize=8, label=category)
                 for category, color in category_colors.items()],
        loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False, fontsize=AXIS_LABEL_SIZE,
    )
    image_label = args.image.stem.split(' - ')[0].replace(' ', '-').replace('_', '-')
    sweep_label = 'hex-only-threshold-sweep' if args.fixed_blobnet_cutoff is not None else 'cutoff-sweep'
    output_path = output_dir / f'fig-Blob-Net-3-{image_label}-agreement-{sweep_label}.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    summary = {
        'figure': str(output_path), 'source_image': str(args.image),
        'velox_selection': velox_selection, 'source_pixel_size_nm': source_pixel_size_nm,
        'field_of_view_nm': field_of_view_nm, 'target_pixel_size_nm': float(args.target_pixel_size_nm),
        'native_inference_pixels': native_pixels, 'agreement_radius_nm': float(args.agreement_radius_nm),
        'blobnet_fixed_cutoff': (
            None if args.fixed_blobnet_cutoff is None else float(args.fixed_blobnet_cutoff)
        ),
        'cutoffs': [float(value) for value in cutoffs], 'category_colors': category_colors,
        'category_markers': {'Both': 'open circle', 'Hex-Net only': 'x', 'Blob-Net only': 'x'},
        'records': records, 'processing': transform,
        'blobnet_checkpoint': str(args.blobnet_checkpoint),
        'hexagonal_checkpoint': str(args.hexagonal_checkpoint),
    }
    output_path.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    return output_path


def make_figure_3c_normalization_sweep(args: argparse.Namespace) -> Path:
    """Sweep robust upper-percentile clipping for the Figure 3C 0063 crop."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    blob_model = _load_blobnet_model(args.blobnet_checkpoint, device, args.num_filters, args.dropout)
    hex_model = _load_blobnet_model(args.hexagonal_checkpoint, device, args.num_filters, args.dropout)
    image, source_pixel_size_nm, velox_selection = _load_velox_displayed_haadf(
        args.image, args.experimental_crop_size,
    )
    field_of_view_nm = float(args.experimental_crop_size) * source_pixel_size_nm
    native_pixels = max(1, int(round(field_of_view_nm / args.target_pixel_size_nm)))
    processed = gaussian_filter(image, args.dog_small) - gaussian_filter(image, args.dog_large)
    upper_percentiles = np.asarray(args.upper_percentiles, dtype=np.float64)
    gammas = (
        np.ones(len(upper_percentiles), dtype=np.float64)
        if args.gammas is None else np.asarray(args.gammas, dtype=np.float64)
    )
    if len(gammas) != len(upper_percentiles):
        raise ValueError('--gammas must contain one value per --upper-percentiles value.')
    records: list[dict[str, Any]] = []
    display_views: list[np.ndarray] = []
    category_coordinates: list[dict[str, np.ndarray]] = []
    category_colors = {
        'Both': args.both_color,
        'Hex-Net only': args.hex_only_color,
        'Blob-Net only': args.blob_only_color,
    }
    for upper_percentile, gamma in zip(upper_percentiles, gammas):
        lower_value, upper_value = np.percentile(
            processed, [args.lower_percentile, float(upper_percentile)],
        )
        normalized = _normalize_image(
            processed, low=args.lower_percentile, high=float(upper_percentile),
        )
        display_view = _center_crop_or_pad(normalized, args.experimental_crop_size)
        native_view = _normalize_image(
            _interpolate_image(display_view, (native_pixels, native_pixels)),
            low=args.lower_percentile, high=float(upper_percentile),
        )
        display_view = np.power(np.clip(display_view, 0.0, 1.0), float(gamma)).astype(np.float32)
        native_view = np.power(np.clip(native_view, 0.0, 1.0), float(gamma)).astype(np.float32)
        display_views.append(display_view)
        model_coordinates: list[np.ndarray] = []
        for model in (blob_model, hex_model):
            prediction = _predict_tiled(
                model, native_view, device, args.tile_size, args.tile_overlap, args.batch_size,
            )
            native_coordinates = np.asarray(extract_subpixel_peak_positions(
                prediction, threshold_rel=args.localization_threshold_rel,
                min_distance=args.peak_min_distance, window_size=args.peak_window_size,
            ), dtype=np.float32).reshape(-1, 2)
            display_coordinates = native_coordinates.copy()
            if len(display_coordinates):
                display_coordinates[:, 0] *= (display_view.shape[0] - 1) / max(native_pixels - 1, 1)
                display_coordinates[:, 1] *= (display_view.shape[1] - 1) / max(native_pixels - 1, 1)
            model_coordinates.append(display_coordinates)
        blob_coordinates, hex_coordinates = model_coordinates
        matching = match_network_predictions(
            blob_coordinates.astype(np.float64) * source_pixel_size_nm,
            hex_coordinates.astype(np.float64) * source_pixel_size_nm,
            args.agreement_radius_nm,
        )
        pairs = matching['pairs']
        groups = {
            'Both': (blob_coordinates[pairs[:, 0]] + hex_coordinates[pairs[:, 1]]) / 2,
            'Hex-Net only': hex_coordinates[matching['hex_only_indices']],
            'Blob-Net only': blob_coordinates[matching['blob_only_indices']],
        }
        category_coordinates.append(groups)
        records.append({
            'lower_percentile': float(args.lower_percentile),
            'upper_percentile': float(upper_percentile),
            'gamma': float(gamma),
            'lower_clip_value': float(lower_value),
            'upper_clip_value': float(upper_value),
            'blob_count': int(len(blob_coordinates)),
            'hex_count': int(len(hex_coordinates)),
            'category_counts': {key: int(len(value)) for key, value in groups.items()},
            'pair_distances_nm': matching['distances_nm'].tolist(),
        })

    columns = len(upper_percentiles)
    fig, axes = plt.subplots(2, columns, figsize=(3.0 * columns, 6.0), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, columns)
    for column, (upper_percentile, gamma, display_view, groups, record) in enumerate(zip(
        upper_percentiles, gammas, display_views, category_coordinates, records,
    )):
        title = f'Upper clip {upper_percentile:g}%'
        if not np.allclose(gammas, 1.0):
            title += f'\nGamma {gamma:g}'
        _plot_clean_image(axes[0, column], display_view, title, cmap='gray')
        _add_physical_scale_bar(
            axes[0, column], display_view.shape, pixel_size_nm=source_pixel_size_nm,
            length_nm=float(args.scale_bar_length_nm), linewidth=float(args.scale_bar_linewidth),
        )
        _plot_clean_image(axes[1, column], display_view, None, cmap='gray')
        for category, atom_coordinates in groups.items():
            if not len(atom_coordinates):
                continue
            style = (
                {'marker': 'o', 'facecolors': 'none', 'edgecolors': category_colors[category]}
                if category == 'Both' else {'marker': 'x', 'color': category_colors[category]}
            )
            axes[1, column].scatter(
                atom_coordinates[:, 1], atom_coordinates[:, 0], s=args.marker_size,
                linewidths=args.marker_linewidth, alpha=0.96, **style,
            )
        _add_physical_scale_bar(
            axes[1, column], display_view.shape, pixel_size_nm=source_pixel_size_nm,
            length_nm=float(args.scale_bar_length_nm), linewidth=float(args.scale_bar_linewidth),
        )
        counts = record['category_counts']
        axes[1, column].text(
            0.03, 0.035,
            f"Both {counts['Both']} | H {counts['Hex-Net only']} | B {counts['Blob-Net only']}",
            transform=axes[1, column].transAxes, color='white', fontsize=7.5,
            ha='left', va='bottom',
            bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': 0.48, 'pad': 1.2},
        )
    fig.legend(
        handles=[Line2D([0], [0], linestyle='none', marker='o' if category == 'Both' else 'x',
                        markerfacecolor='none', markeredgecolor=color,
                        markeredgewidth=args.marker_linewidth, markersize=8, label=category)
                 for category, color in category_colors.items()],
        loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3,
        frameon=False, fontsize=AXIS_LABEL_SIZE,
    )
    output_path = output_dir / 'fig-Blob-Net-3c-0063-normalization-sweep.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    summary = {
        'figure': str(output_path), 'source_image': str(args.image),
        'velox_selection': velox_selection, 'source_pixel_size_nm': source_pixel_size_nm,
        'field_of_view_nm': field_of_view_nm, 'target_pixel_size_nm': float(args.target_pixel_size_nm),
        'native_inference_pixels': native_pixels,
        'localization_threshold_rel': float(args.localization_threshold_rel),
        'agreement_radius_nm': float(args.agreement_radius_nm),
        'normalization': 'DoG followed by lower/upper percentile clipping and linear scaling to [0, 1]',
        'upper_percentiles': [float(value) for value in upper_percentiles],
        'gammas': [float(value) for value in gammas],
        'category_colors': category_colors,
        'category_markers': {'Both': 'open circle', 'Hex-Net only': 'x', 'Blob-Net only': 'x'},
        'records': records,
        'blobnet_checkpoint': str(args.blobnet_checkpoint),
        'hexagonal_checkpoint': str(args.hexagonal_checkpoint),
    }
    output_path.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    return output_path


def make_figure_3c_pixel_size_sweep(args: argparse.Namespace) -> Path:
    """Sweep physical pixel size for both Figure 3C models on one Velox HAADF image."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    models = [
        (
            'Blob-Net',
            _load_blobnet_model(args.blobnet_checkpoint, device, args.num_filters, args.dropout),
            args.blobnet_marker_color,
        ),
        (
            'Hexagonal model',
            _load_blobnet_model(args.hexagonal_checkpoint, device, args.num_filters, args.dropout),
            args.hexagonal_marker_color,
        ),
    ]

    if getattr(args, 'standard_emd', False):
        image = _load_experimental_image(args.image)
        source_pixel_size_nm = _read_channel_pixel_size_nm(args.image)
        velox_selection = None
    else:
        image, source_pixel_size_nm, velox_selection = _load_velox_displayed_haadf(
            args.image,
            args.experimental_crop_size,
        )
    image_label = args.image.stem.split(' - ')[0].replace(' ', '-').replace('_', '-')
    processed = gaussian_filter(image, args.dog_small) - gaussian_filter(image, args.dog_large)
    display_view = _center_crop_or_pad(_normalize_image(processed), args.experimental_crop_size)
    field_of_view_nm = float(args.experimental_crop_size) * source_pixel_size_nm
    factors = np.asarray(args.pixel_size_factors, dtype=np.float64)

    native_views: list[np.ndarray] = []
    scale_records: list[dict[str, Any]] = []
    for factor in factors:
        target_pixel_size_nm = float(args.base_pixel_size_nm) * float(factor)
        native_pixels = max(1, int(round(field_of_view_nm / target_pixel_size_nm)))
        native_views.append(
            _normalize_image(_interpolate_image(display_view, (native_pixels, native_pixels)))
        )
        scale_records.append(
            {
                'factor': float(factor),
                'target_pixel_size_nm': target_pixel_size_nm,
                'native_inference_pixels': native_pixels,
            }
        )

    fixed_blobnet_view: np.ndarray | None = None
    fixed_blobnet_record: dict[str, Any] | None = None
    if args.fixed_blobnet_factor is not None:
        fixed_target_pixel_size_nm = float(args.base_pixel_size_nm) * float(args.fixed_blobnet_factor)
        fixed_native_pixels = max(1, int(round(field_of_view_nm / fixed_target_pixel_size_nm)))
        fixed_blobnet_view = _normalize_image(
            _interpolate_image(display_view, (fixed_native_pixels, fixed_native_pixels))
        )
        fixed_blobnet_record = {
            'factor': float(args.fixed_blobnet_factor),
            'target_pixel_size_nm': fixed_target_pixel_size_nm,
            'native_inference_pixels': fixed_native_pixels,
        }

    coordinates: dict[str, list[np.ndarray]] = {}
    model_records: dict[str, list[dict[str, Any]]] = {}
    for model_label, model, _marker_color in models:
        coordinates_for_model: list[np.ndarray] = []
        records_for_model: list[dict[str, Any]] = []
        for native_view, scale_record in zip(native_views, scale_records):
            inference_view = (
                fixed_blobnet_view
                if model_label == 'Blob-Net' and fixed_blobnet_view is not None
                else native_view
            )
            effective_scale_record = (
                fixed_blobnet_record
                if model_label == 'Blob-Net' and fixed_blobnet_record is not None
                else scale_record
            )
            native_prediction = _predict_tiled(
                model,
                inference_view,
                device,
                args.tile_size,
                args.tile_overlap,
                args.batch_size,
            )
            native_coordinates = extract_subpixel_peak_positions(
                native_prediction,
                threshold_rel=args.localization_threshold_rel,
                min_distance=args.peak_min_distance,
                window_size=args.peak_window_size,
            )
            display_coordinates = np.asarray(native_coordinates, dtype=np.float32).copy()
            if len(display_coordinates):
                display_coordinates[:, 0] *= (display_view.shape[0] - 1) / max(inference_view.shape[0] - 1, 1)
                display_coordinates[:, 1] *= (display_view.shape[1] - 1) / max(inference_view.shape[1] - 1, 1)
            coordinates_for_model.append(display_coordinates)
            records_for_model.append(
                {
                    **effective_scale_record,
                    'comparison_factor': float(scale_record['factor']),
                    'predicted_atom_count': int(len(display_coordinates)),
                    'prediction_mean': float(native_prediction.mean()),
                    'prediction_max': float(native_prediction.max()),
                }
            )
        coordinates[model_label] = coordinates_for_model
        model_records[model_label] = records_for_model

    fig = plt.figure(figsize=(3.2 + 4.0 * len(factors), 7.1), constrained_layout=True)
    grid = fig.add_gridspec(2, len(factors) + 1, width_ratios=[1.08] + [1.0] * len(factors))
    input_axis = fig.add_subplot(grid[:, 0])
    _plot_clean_image(input_axis, display_view, args.image.name.split(' - ')[0], cmap='gray')
    _add_physical_scale_bar(
        input_axis,
        display_view.shape,
        pixel_size_nm=source_pixel_size_nm,
        length_nm=float(args.scale_bar_length_nm),
        linewidth=float(args.scale_bar_linewidth),
    )

    for row, (model_label, _model, marker_color) in enumerate(models):
        for col, scale_record in enumerate(scale_records, start=1):
            axis = fig.add_subplot(grid[row, col])
            _plot_clean_image(axis, display_view, '', cmap='gray')
            atom_coordinates = coordinates[model_label][col - 1]
            if len(atom_coordinates):
                axis.scatter(
                    atom_coordinates[:, 1],
                    atom_coordinates[:, 0],
                    s=args.marker_size,
                    facecolors=marker_color,
                    edgecolors=args.marker_edge_color,
                    linewidths=args.marker_linewidth,
                    alpha=0.96,
                )
            _add_physical_scale_bar(
                axis,
                display_view.shape,
                pixel_size_nm=source_pixel_size_nm,
                length_nm=float(args.scale_bar_length_nm),
                linewidth=float(args.scale_bar_linewidth),
            )
            if row == 0:
                axis.set_title(
                    f"{scale_record['factor']:.2f}x | {scale_record['target_pixel_size_nm']:.5f} nm/px\n"
                    f"{scale_record['native_inference_pixels']} px inference",
                    fontsize=11,
                )
            axis.text(
                0.04,
                0.04,
                f'n={len(atom_coordinates)}',
                transform=axis.transAxes,
                color='white',
                fontsize=9,
                ha='left',
                va='bottom',
                bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': 0.45, 'pad': 1.2},
            )
            if col == 1:
                axis.set_ylabel(model_label, fontsize=AXIS_LABEL_SIZE)

    output_path = output_dir / 'fig-Blob-Net-3c-0063-pixel-size-sweep.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    if args.combined_overlay:
        fig, axes = plt.subplots(1, len(factors), figsize=(6.4 * len(factors), 6.4), constrained_layout=True)
        for col, (axis, scale_record) in enumerate(zip(np.asarray(axes).reshape(-1), scale_records)):
            _plot_clean_image(
                axis, display_view,
                f"{scale_record['factor']:.2f}x | {scale_record['target_pixel_size_nm']:.5f} nm/px\n"
                f"{scale_record['native_inference_pixels']} x {scale_record['native_inference_pixels']} inference grid",
                cmap='gray',
            )
            blob_coordinates = coordinates['Blob-Net'][col]
            hex_coordinates = coordinates['Hexagonal model'][col]
            if len(blob_coordinates):
                axis.scatter(
                    blob_coordinates[:, 1], blob_coordinates[:, 0], s=args.marker_size,
                    marker='o', facecolors='none', edgecolors=args.blobnet_marker_color,
                    linewidths=1.05, zorder=3,
                )
            if len(hex_coordinates):
                axis.scatter(
                    hex_coordinates[:, 1], hex_coordinates[:, 0], s=float(args.marker_size) * 0.72,
                    marker='x', color=args.comparison_hexagonal_marker_color, linewidths=1.05, zorder=4,
                )
            _add_physical_scale_bar(
                axis, display_view.shape, pixel_size_nm=source_pixel_size_nm,
                length_nm=float(args.scale_bar_length_nm), linewidth=float(args.scale_bar_linewidth),
            )
            axis.text(
                0.04, 0.04, f'B={len(blob_coordinates)}  H={len(hex_coordinates)}',
                transform=axis.transAxes, color='white', fontsize=9, ha='left', va='bottom',
                bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': 0.45, 'pad': 1.2},
            )
        fig.legend(
            handles=[
                Line2D([0], [0], linestyle='none', marker='o', markerfacecolor='none',
                       markeredgecolor=args.blobnet_marker_color, markeredgewidth=1.05,
                       markersize=8, label='Blob-Net'),
                Line2D([0], [0], linestyle='none', marker='x', color=args.comparison_hexagonal_marker_color,
                       markeredgewidth=1.4, markersize=8, label='Hexagonal model'),
            ],
            loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, fontsize=AXIS_LABEL_SIZE,
        )
        output_path = output_dir / f'fig-Blob-Net-3c-{image_label}-combined-pixel-size-sweep.png'
        fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)

    agreement_records: list[dict[str, Any]] = []
    if args.agreement_overlay:
        category_colors = {
            'Both': args.both_color,
            'Hex-Net only': args.hex_only_color,
            'Blob-Net only': args.blob_only_color,
        }
        category_coordinates: list[dict[str, np.ndarray]] = []
        for col, scale_record in enumerate(scale_records):
            blob_coordinates = coordinates['Blob-Net'][col]
            hex_coordinates = coordinates['Hexagonal model'][col]
            matching = match_network_predictions(
                blob_coordinates.astype(np.float64) * source_pixel_size_nm,
                hex_coordinates.astype(np.float64) * source_pixel_size_nm,
                args.agreement_radius_nm,
            )
            pairs = matching['pairs']
            groups = {
                'Both': (blob_coordinates[pairs[:, 0]] + hex_coordinates[pairs[:, 1]]) / 2,
                'Hex-Net only': hex_coordinates[matching['hex_only_indices']],
                'Blob-Net only': blob_coordinates[matching['blob_only_indices']],
            }
            category_coordinates.append(groups)
            agreement_records.append({
                **scale_record,
                'blob_count': int(len(blob_coordinates)),
                'hex_count': int(len(hex_coordinates)),
                'category_counts': {key: int(len(value)) for key, value in groups.items()},
                'pair_distances_nm': matching['distances_nm'].tolist(),
            })

        panel_count = len(factors) + 1
        columns = 4
        rows = int(math.ceil(panel_count / columns))
        fig, axes = plt.subplots(
            rows, columns, figsize=(3.35 * columns, 3.35 * rows), constrained_layout=True,
        )
        flat_axes = np.asarray(axes).reshape(-1)
        _plot_clean_image(flat_axes[0], display_view, 'Input', cmap='gray')
        _add_physical_scale_bar(
            flat_axes[0], display_view.shape, pixel_size_nm=source_pixel_size_nm,
            length_nm=float(args.scale_bar_length_nm), linewidth=float(args.scale_bar_linewidth),
        )
        for axis, groups, record in zip(flat_axes[1:], category_coordinates, agreement_records):
            if args.fixed_blobnet_factor is None:
                title = (
                    f"{record['factor']:.2f}x current | {record['target_pixel_size_nm']:.5f} nm/px\n"
                    f"{record['native_inference_pixels']} px inference"
                )
            else:
                title = (
                    f"Hex {record['factor']:.2f}x | Blob {args.fixed_blobnet_factor:.2f}x\n"
                    f"Hex {record['native_inference_pixels']} px inference"
                )
            _plot_clean_image(axis, display_view, title, cmap='gray')
            for category, atom_coordinates in groups.items():
                if not len(atom_coordinates):
                    continue
                style = (
                    {'marker': 'o', 'facecolors': 'none', 'edgecolors': category_colors[category]}
                    if category == 'Both' else {'marker': 'x', 'color': category_colors[category]}
                )
                axis.scatter(
                    atom_coordinates[:, 1], atom_coordinates[:, 0], s=args.marker_size,
                    linewidths=args.agreement_marker_linewidth, alpha=0.96, **style,
                )
            _add_physical_scale_bar(
                axis, display_view.shape, pixel_size_nm=source_pixel_size_nm,
                length_nm=float(args.scale_bar_length_nm), linewidth=float(args.scale_bar_linewidth),
            )
            counts = record['category_counts']
            axis.text(
                0.03, 0.035,
                f"Both {counts['Both']} | H {counts['Hex-Net only']} | B {counts['Blob-Net only']}",
                transform=axis.transAxes, color='white', fontsize=7.5, ha='left', va='bottom',
                bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': 0.48, 'pad': 1.2},
            )
        for axis in flat_axes[panel_count:]:
            axis.axis('off')
        fig.legend(
            handles=[Line2D([0], [0], linestyle='none', marker='o' if category == 'Both' else 'x',
                            markerfacecolor='none', markeredgecolor=color,
                            markeredgewidth=args.agreement_marker_linewidth, markersize=8, label=category)
                     for category, color in category_colors.items()],
            loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3,
            frameon=False, fontsize=AXIS_LABEL_SIZE,
        )
        sweep_label = 'hex-only-pixel-size-sweep' if args.fixed_blobnet_factor is not None else 'pixel-size-sweep'
        output_path = output_dir / f'fig-Blob-Net-3-{image_label}-agreement-{sweep_label}.png'
        fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)

    summary = {
        'figure': str(output_path),
        'source_image': str(args.image),
        'image_selection': 'Final displayed DCFI(HAADF) series frame.',
        'velox_selection': velox_selection,
        'source_pixel_size_nm': source_pixel_size_nm,
        'field_of_view_nm': field_of_view_nm,
        'base_pixel_size_nm': float(args.base_pixel_size_nm),
        'blobnet_fixed_factor': (
            None if args.fixed_blobnet_factor is None else float(args.fixed_blobnet_factor)
        ),
        'pixel_size_factors': [float(value) for value in factors],
        'localization_threshold_rel': float(args.localization_threshold_rel),
        'models': {
            'Blob-Net': {
                'checkpoint': str(args.blobnet_checkpoint),
                'marker_color': args.blobnet_marker_color,
                'records': model_records['Blob-Net'],
            },
            'Hexagonal model': {
                'checkpoint': str(args.hexagonal_checkpoint),
                'marker_color': args.hexagonal_marker_color,
                'records': model_records['Hexagonal model'],
            },
        },
    }
    if args.combined_overlay:
        summary['combined_overlay_style'] = {
            'Blob-Net': {'marker': 'o', 'fill': 'none', 'color': args.blobnet_marker_color},
            'Hexagonal model': {'marker': 'x', 'color': args.comparison_hexagonal_marker_color},
        }
    if args.agreement_overlay:
        summary['agreement_radius_nm'] = float(args.agreement_radius_nm)
        summary['agreement_records'] = agreement_records
        summary['agreement_overlay_style'] = {
            'Both': {'marker': 'o', 'fill': 'none', 'color': args.both_color},
            'Hex-Net only': {'marker': 'x', 'color': args.hex_only_color},
            'Blob-Net only': {'marker': 'x', 'color': args.blob_only_color},
        }
    output_path.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    return output_path


def make_figure_3_ws2_hexnet_pixel_cutoff_grid(args: argparse.Namespace) -> Path:
    """Plot Hex-Net detections across a physical-pixel-size by cutoff grid."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    model = _load_blobnet_model(
        args.hexagonal_checkpoint, device, args.num_filters, args.dropout,
    )
    image = _load_experimental_image(args.image)
    source_pixel_size_nm = _read_channel_pixel_size_nm(args.image)
    processed = gaussian_filter(image, args.dog_small) - gaussian_filter(image, args.dog_large)
    display_view = _center_crop_or_pad(_normalize_image(processed), args.experimental_crop_size)
    field_of_view_nm = float(args.experimental_crop_size) * source_pixel_size_nm
    factors = np.asarray(args.pixel_size_factors, dtype=np.float64)
    cutoffs = np.asarray(args.cutoffs, dtype=np.float64)

    predictions: list[np.ndarray] = []
    scale_records: list[dict[str, Any]] = []
    for factor in factors:
        target_pixel_size_nm = float(args.base_pixel_size_nm) * float(factor)
        native_pixels = max(1, int(round(field_of_view_nm / target_pixel_size_nm)))
        native_view = _normalize_image(
            _interpolate_image(display_view, (native_pixels, native_pixels))
        )
        predictions.append(_predict_tiled(
            model, native_view, device, args.tile_size, args.tile_overlap, args.batch_size,
        ))
        scale_records.append({
            'factor': float(factor),
            'target_pixel_size_nm': target_pixel_size_nm,
            'native_inference_pixels': native_pixels,
        })

    coordinate_grid: list[list[np.ndarray]] = []
    records: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        row_coordinates: list[np.ndarray] = []
        for prediction, scale_record in zip(predictions, scale_records):
            native_coordinates = np.asarray(extract_subpixel_peak_positions(
                prediction, threshold_rel=float(cutoff), min_distance=args.peak_min_distance,
                window_size=args.peak_window_size,
            ), dtype=np.float32).reshape(-1, 2)
            display_coordinates = native_coordinates.copy()
            if len(display_coordinates):
                native_pixels = int(scale_record['native_inference_pixels'])
                display_coordinates[:, 0] *= (display_view.shape[0] - 1) / max(native_pixels - 1, 1)
                display_coordinates[:, 1] *= (display_view.shape[1] - 1) / max(native_pixels - 1, 1)
            row_coordinates.append(display_coordinates)
            records.append({
                **scale_record,
                'cutoff': float(cutoff),
                'predicted_atom_count': int(len(display_coordinates)),
            })
        coordinate_grid.append(row_coordinates)

    rows = len(cutoffs)
    columns = len(factors)
    fig, axes = plt.subplots(
        rows, columns, figsize=(2.7 * columns, 2.7 * rows), constrained_layout=True,
        squeeze=False,
    )
    for row, cutoff in enumerate(cutoffs):
        for column, (scale_record, atom_coordinates) in enumerate(zip(
            scale_records, coordinate_grid[row],
        )):
            axis = axes[row, column]
            title = ''
            if row == 0:
                title = (
                    f"{scale_record['factor']:.2f}x | "
                    f"{scale_record['target_pixel_size_nm']:.5f} nm/px\n"
                    f"{scale_record['native_inference_pixels']} px"
                )
            _plot_clean_image(axis, display_view, title, cmap='gray')
            if len(atom_coordinates):
                axis.scatter(
                    atom_coordinates[:, 1], atom_coordinates[:, 0],
                    s=args.marker_size, marker='o', facecolors='none',
                    edgecolors=args.hexnet_color, linewidths=args.marker_linewidth,
                    alpha=0.96,
                )
            _add_physical_scale_bar(
                axis, display_view.shape, pixel_size_nm=source_pixel_size_nm,
                length_nm=float(args.scale_bar_length_nm),
                linewidth=float(args.scale_bar_linewidth),
            )
            axis.text(
                0.03, 0.035, f'n={len(atom_coordinates)}', transform=axis.transAxes,
                color='white', fontsize=7.5, ha='left', va='bottom',
                bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': 0.48, 'pad': 1.2},
            )
            if column == 0:
                axis.set_ylabel(f'Cutoff {cutoff:g}', fontsize=AXIS_LABEL_SIZE)

    output_path = output_dir / 'fig-Blob-Net-3-WS2-hexnet-pixel-cutoff-grid.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    summary = {
        'figure': str(output_path),
        'source_image': str(args.image),
        'model': 'Hex-Net',
        'checkpoint': str(args.hexagonal_checkpoint),
        'source_pixel_size_nm': source_pixel_size_nm,
        'field_of_view_nm': field_of_view_nm,
        'base_pixel_size_nm': float(args.base_pixel_size_nm),
        'pixel_size_factors': [float(value) for value in factors],
        'cutoffs': [float(value) for value in cutoffs],
        'marker': {'shape': 'open circle', 'color': args.hexnet_color},
        'records': records,
        'processing': 'DoG background subtraction, 1st-99.8th percentile normalization, fixed FOV.',
    }
    output_path.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    return output_path


def make_figure_3c_fov_sweep(args: argparse.Namespace) -> Path:
    """Compare centered fields of view at one physical inference pixel size."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    models = [
        (
            'Blob-Net',
            _load_blobnet_model(args.blobnet_checkpoint, device, args.num_filters, args.dropout),
            args.blobnet_marker_color,
        ),
        (
            'Hexagonal model',
            _load_blobnet_model(args.hexagonal_checkpoint, device, args.num_filters, args.dropout),
            args.hexagonal_marker_color,
        ),
    ]

    largest_crop = max(args.crop_sizes)
    image, source_pixel_size_nm, velox_selection = _load_velox_displayed_haadf(
        args.image,
        largest_crop,
    )
    views: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    for crop_size in args.crop_sizes:
        field_of_view_nm = float(crop_size) * source_pixel_size_nm
        native_pixels = max(1, int(round(field_of_view_nm / args.target_pixel_size_nm)))
        display_view, native_view, transform = _make_fixed_fov_resolution_view(
            image,
            dog_small=args.dog_small,
            dog_large=args.dog_large,
            display_size=crop_size,
            native_pixels=native_pixels,
        )
        transform.update(
            {
                'source_crop_pixels': int(crop_size),
                'field_of_view_nm': field_of_view_nm,
                'target_pixel_size_nm': float(args.target_pixel_size_nm),
            }
        )
        views.append((display_view, native_view, transform))

    coordinates: dict[str, list[np.ndarray]] = {}
    model_records: dict[str, list[dict[str, Any]]] = {}
    for model_label, model, _marker_color in models:
        model_coordinates: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        for display_view, native_view, transform in views:
            native_prediction = _predict_tiled(
                model,
                native_view,
                device,
                args.tile_size,
                args.tile_overlap,
                args.batch_size,
            )
            native_coordinates = extract_subpixel_peak_positions(
                native_prediction,
                threshold_rel=args.localization_threshold_rel,
                min_distance=args.peak_min_distance,
                window_size=args.peak_window_size,
            )
            display_coordinates = np.asarray(native_coordinates, dtype=np.float32).copy()
            if len(display_coordinates):
                display_coordinates[:, 0] *= (display_view.shape[0] - 1) / max(native_view.shape[0] - 1, 1)
                display_coordinates[:, 1] *= (display_view.shape[1] - 1) / max(native_view.shape[1] - 1, 1)
            model_coordinates.append(display_coordinates)
            records.append(
                {
                    **transform,
                    'predicted_atom_count': int(len(display_coordinates)),
                    'prediction_mean': float(native_prediction.mean()),
                    'prediction_max': float(native_prediction.max()),
                }
            )
        coordinates[model_label] = model_coordinates
        model_records[model_label] = records

    fig, axes = plt.subplots(3, len(views), figsize=(4.0 * len(views), 11.2), constrained_layout=True)
    for col, (display_view, _native_view, transform) in enumerate(views):
        title = (
            f"{transform['source_crop_pixels']} px crop\n"
            f"{transform['field_of_view_nm']:.2f} nm FOV"
        )
        _plot_clean_image(axes[0, col], display_view, title, cmap='gray')
        _add_physical_scale_bar(
            axes[0, col],
            display_view.shape,
            pixel_size_nm=source_pixel_size_nm,
            length_nm=float(args.scale_bar_length_nm),
            linewidth=float(args.scale_bar_linewidth),
        )
        for row, (model_label, _model, marker_color) in enumerate(models, start=1):
            _plot_clean_image(axes[row, col], display_view, '', cmap='gray')
            atom_coordinates = coordinates[model_label][col]
            marker_size = float(args.marker_size) * (float(args.crop_sizes[0]) / display_view.shape[0]) ** 2
            if len(atom_coordinates):
                axes[row, col].scatter(
                    atom_coordinates[:, 1],
                    atom_coordinates[:, 0],
                    s=marker_size,
                    facecolors=marker_color,
                    edgecolors=args.marker_edge_color,
                    linewidths=float(args.marker_linewidth) * float(args.crop_sizes[0]) / display_view.shape[0],
                    alpha=0.96,
                )
            _add_physical_scale_bar(
                axes[row, col],
                display_view.shape,
                pixel_size_nm=source_pixel_size_nm,
                length_nm=float(args.scale_bar_length_nm),
                linewidth=float(args.scale_bar_linewidth),
            )
            axes[row, col].text(
                0.04,
                0.04,
                f'n={len(atom_coordinates)}',
                transform=axes[row, col].transAxes,
                color='white',
                fontsize=9,
                ha='left',
                va='bottom',
                bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': 0.45, 'pad': 1.2},
            )
    axes[0, 0].set_ylabel('Input', fontsize=AXIS_LABEL_SIZE)
    axes[1, 0].set_ylabel('Blob-Net', fontsize=AXIS_LABEL_SIZE)
    axes[2, 0].set_ylabel('Hexagonal model', fontsize=AXIS_LABEL_SIZE)

    image_label = args.image.name.split(' - ')[0]
    output_path = output_dir / f'fig-Blob-Net-3c-{image_label}-fov-sweep.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'figure': str(output_path),
        'source_image': str(args.image),
        'image_selection': 'Final displayed DCFI(HAADF) series frame, with HAADF fallback.',
        'velox_selection': velox_selection,
        'source_pixel_size_nm': source_pixel_size_nm,
        'source_shape': velox_selection['source_shape'],
        'crop_sizes': [int(value) for value in args.crop_sizes],
        'target_pixel_size_nm': float(args.target_pixel_size_nm),
        'localization_threshold_rel': float(args.localization_threshold_rel),
        'models': {
            model_label: {
                'checkpoint': str(checkpoint),
                'marker_color': marker_color,
                'records': model_records[model_label],
            }
            for model_label, checkpoint, marker_color in [
                ('Blob-Net', args.blobnet_checkpoint, args.blobnet_marker_color),
                ('Hexagonal model', args.hexagonal_checkpoint, args.hexagonal_marker_color),
            ]
        },
    }
    (output_dir / f'fig-Blob-Net-3c-{image_label}-fov-sweep.json').write_text(json.dumps(summary, indent=2))
    return output_path


def make_figure_3c_region_atlas(args: argparse.Namespace) -> Path:
    """Tile a full Velox frame into regional localization atlases for both models."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    models = [
        (
            'Blob-Net',
            _load_blobnet_model(args.blobnet_checkpoint, device, args.num_filters, args.dropout),
            args.blobnet_marker_color,
            'blobnet',
        ),
        (
            'Hexagonal model',
            _load_blobnet_model(args.hexagonal_checkpoint, device, args.num_filters, args.dropout),
            args.hexagonal_marker_color,
            'hexagonal',
        ),
    ]

    full_size = int(args.region_size * args.grid_size)
    image, source_pixel_size_nm, velox_selection = _load_velox_displayed_haadf(args.image, full_size)
    source_height, source_width = velox_selection['source_shape']
    if full_size > min(source_height, source_width):
        raise ValueError('Requested region grid exceeds the recorded image dimensions.')
    source_y_offset = (source_height - full_size) // 2
    source_x_offset = (source_width - full_size) // 2
    field_of_view_nm = float(args.region_size) * source_pixel_size_nm
    native_pixels = max(1, int(round(field_of_view_nm / args.target_pixel_size_nm)))
    region_views: list[tuple[str, np.ndarray, np.ndarray, dict[str, Any]]] = []
    region_index = 0
    for row in range(args.grid_size):
        for col in range(args.grid_size):
            region_index += 1
            y0 = row * args.region_size
            x0 = col * args.region_size
            region = image[y0 : y0 + args.region_size, x0 : x0 + args.region_size]
            display_view, native_view, transform = _make_fixed_fov_resolution_view(
                region,
                dog_small=args.dog_small,
                dog_large=args.dog_large,
                display_size=args.region_size,
                native_pixels=native_pixels,
            )
            transform.update(
                {
                    'region': f'R{region_index}',
                    'grid_row': row + 1,
                    'grid_column': col + 1,
                    'source_bounds_pixels': [
                        source_y_offset + y0,
                        source_y_offset + y0 + args.region_size,
                        source_x_offset + x0,
                        source_x_offset + x0 + args.region_size,
                    ],
                    'field_of_view_nm': field_of_view_nm,
                }
            )
            region_views.append((f'R{region_index}', display_view, native_view, transform))

    model_records: dict[str, list[dict[str, Any]]] = {}
    model_coordinates: dict[str, list[np.ndarray]] = {}
    output_paths: dict[str, Path] = {}
    image_label = args.image.name.split(' - ')[0]
    for model_label, model, marker_color, file_label in models:
        records: list[dict[str, Any]] = []
        coordinates: list[np.ndarray] = []
        for region_label, display_view, native_view, transform in region_views:
            native_prediction = _predict_tiled(
                model,
                native_view,
                device,
                args.tile_size,
                args.tile_overlap,
                args.batch_size,
            )
            native_coordinates = extract_subpixel_peak_positions(
                native_prediction,
                threshold_rel=args.localization_threshold_rel,
                min_distance=args.peak_min_distance,
                window_size=args.peak_window_size,
            )
            display_coordinates = np.asarray(native_coordinates, dtype=np.float32).copy()
            if len(display_coordinates):
                display_coordinates[:, 0] *= (display_view.shape[0] - 1) / max(native_view.shape[0] - 1, 1)
                display_coordinates[:, 1] *= (display_view.shape[1] - 1) / max(native_view.shape[1] - 1, 1)
            coordinates.append(display_coordinates)
            records.append(
                {
                    **transform,
                    'predicted_atom_count': int(len(display_coordinates)),
                    'prediction_mean': float(native_prediction.mean()),
                    'prediction_max': float(native_prediction.max()),
                }
            )

        fig, axes = plt.subplots(
            args.grid_size,
            args.grid_size,
            figsize=(3.2 * args.grid_size, 3.2 * args.grid_size),
            constrained_layout=True,
        )
        for axis, (region_label, display_view, _native_view, _transform), atom_coordinates in zip(
            np.asarray(axes).reshape(-1),
            region_views,
            coordinates,
        ):
            _plot_clean_image(axis, display_view, region_label, cmap='gray')
            if len(atom_coordinates):
                axis.scatter(
                    atom_coordinates[:, 1],
                    atom_coordinates[:, 0],
                    s=args.marker_size,
                    facecolors=marker_color,
                    edgecolors=args.marker_edge_color,
                    linewidths=args.marker_linewidth,
                    alpha=0.96,
                )
            _add_physical_scale_bar(
                axis,
                display_view.shape,
                pixel_size_nm=source_pixel_size_nm,
                length_nm=float(args.scale_bar_length_nm),
                linewidth=float(args.scale_bar_linewidth),
            )
            axis.text(
                0.04,
                0.04,
                f'n={len(atom_coordinates)}',
                transform=axis.transAxes,
                color='white',
                fontsize=9,
                ha='left',
                va='bottom',
                bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': 0.45, 'pad': 1.2},
            )
        fig.suptitle(model_label, fontsize=AXIS_LABEL_SIZE + 2)
        output_path = output_dir / f'fig-Blob-Net-3c-{image_label}-512-region-atlas-{file_label}.png'
        fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        output_paths[model_label] = output_path
        model_records[model_label] = records
        model_coordinates[model_label] = coordinates

    fig, axes = plt.subplots(
        args.grid_size,
        args.grid_size,
        figsize=(max(6.4, 3.2 * args.grid_size), max(6.4, 3.2 * args.grid_size)),
        constrained_layout=True,
    )
    for region_number, (axis, (_region_label, display_view, _native_view, _transform)) in enumerate(
        zip(np.asarray(axes).reshape(-1), region_views),
        start=1,
    ):
        region_title = f'{image_label}: center' if args.grid_size == 1 else f'R{region_number}'
        _plot_clean_image(axis, display_view, region_title, cmap='gray')
        blob_coordinates = model_coordinates['Blob-Net'][region_number - 1]
        hex_coordinates = model_coordinates['Hexagonal model'][region_number - 1]
        if len(blob_coordinates):
            axis.scatter(
                blob_coordinates[:, 1],
                blob_coordinates[:, 0],
                s=args.marker_size,
                marker='o',
                facecolors='none',
                edgecolors=args.blobnet_marker_color,
                linewidths=1.05,
                alpha=1.0,
                zorder=3,
            )
        if len(hex_coordinates):
            axis.scatter(
                hex_coordinates[:, 1],
                hex_coordinates[:, 0],
                s=float(args.marker_size) * 0.72,
                marker='x',
                color=args.comparison_hexagonal_marker_color,
                linewidths=1.05,
                alpha=1.0,
                zorder=4,
            )
        _add_physical_scale_bar(
            axis,
            display_view.shape,
            pixel_size_nm=source_pixel_size_nm,
            length_nm=float(args.scale_bar_length_nm),
            linewidth=float(args.scale_bar_linewidth),
        )
        axis.text(
            0.04,
            0.04,
            f'B={len(blob_coordinates)}  H={len(hex_coordinates)}',
            transform=axis.transAxes,
            color='white',
            fontsize=8,
            ha='left',
            va='bottom',
            bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': 0.45, 'pad': 1.2},
        )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker='o',
            color='none',
            markerfacecolor='none',
            markeredgecolor=args.blobnet_marker_color,
            markeredgewidth=1.05,
            markersize=8,
            label='Blob-Net',
        ),
        Line2D(
            [0],
            [0],
            marker='x',
            color=args.comparison_hexagonal_marker_color,
            markeredgewidth=1.4,
            markersize=8,
            label='Hexagonal model',
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08 if args.grid_size == 1 else 1.035),
        ncol=2,
        frameon=False,
        fontsize=AXIS_LABEL_SIZE,
    )
    combined_path = output_dir / f'fig-Blob-Net-3c-{image_label}-512-region-atlas-combined.png'
    fig.savefig(combined_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    output_paths['Combined comparison'] = combined_path

    summary = {
        'figures': {label: str(path) for label, path in output_paths.items()},
        'combined_overlay_style': {
            'Blob-Net': {'marker': 'o', 'color': args.blobnet_marker_color, 'fill': 'none', 'linewidth': 1.05},
            'Hexagonal model': {'marker': 'x', 'color': args.comparison_hexagonal_marker_color},
        },
        'source_image': str(args.image),
        'image_selection': 'Full displayed DCFI(HAADF) series frame, with HAADF fallback.',
        'velox_selection': velox_selection,
        'source_pixel_size_nm': source_pixel_size_nm,
        'full_frame_pixels': full_size,
        'recorded_source_shape': velox_selection['source_shape'],
        'grid_origin_source_pixels': [source_y_offset, source_x_offset],
        'grid_size': int(args.grid_size),
        'region_size_pixels': int(args.region_size),
        'region_field_of_view_nm': field_of_view_nm,
        'target_pixel_size_nm': float(args.target_pixel_size_nm),
        'native_inference_pixels_per_region': native_pixels,
        'localization_threshold_rel': float(args.localization_threshold_rel),
        'models': {
            'Blob-Net': {
                'checkpoint': str(args.blobnet_checkpoint),
                'marker_color': args.blobnet_marker_color,
                'records': model_records['Blob-Net'],
            },
            'Hexagonal model': {
                'checkpoint': str(args.hexagonal_checkpoint),
                'marker_color': args.hexagonal_marker_color,
                'records': model_records['Hexagonal model'],
            },
        },
    }
    metadata_path = output_dir / f'fig-Blob-Net-3c-{image_label}-512-region-atlas.json'
    metadata_path.write_text(json.dumps(summary, indent=2))
    return output_paths['Combined comparison']


def make_figure_3c_sqrt_input(args: argparse.Namespace) -> Path:
    """Compare standard and square-root-transformed inputs for Blob-Net on one image."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    model = _load_blobnet_model(args.blobnet_checkpoint, device, args.num_filters, args.dropout)

    image, source_pixel_size_nm, velox_selection = _load_velox_displayed_haadf(
        args.image,
        args.experimental_crop_size,
    )
    field_of_view_nm = float(args.experimental_crop_size) * source_pixel_size_nm
    native_pixels = max(1, int(round(field_of_view_nm / args.target_pixel_size_nm)))
    display_view, native_view, transform = _make_fixed_fov_resolution_view(
        image,
        dog_small=args.dog_small,
        dog_large=args.dog_large,
        display_size=args.experimental_crop_size,
        native_pixels=native_pixels,
    )
    variants = [
        ('Standard input', display_view, native_view, 1.0),
        ('Square-root input', np.sqrt(np.clip(display_view, 0.0, 1.0)), np.sqrt(np.clip(native_view, 0.0, 1.0)), 0.5),
    ]

    coordinates: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for label, _display_input, inference_input, power in variants:
        native_prediction = _predict_tiled(
            model,
            inference_input,
            device,
            args.tile_size,
            args.tile_overlap,
            args.batch_size,
        )
        native_coordinates = extract_subpixel_peak_positions(
            native_prediction,
            threshold_rel=args.localization_threshold_rel,
            min_distance=args.peak_min_distance,
            window_size=args.peak_window_size,
        )
        display_coordinates = np.asarray(native_coordinates, dtype=np.float32).copy()
        if len(display_coordinates):
            display_coordinates[:, 0] *= (display_view.shape[0] - 1) / max(inference_input.shape[0] - 1, 1)
            display_coordinates[:, 1] *= (display_view.shape[1] - 1) / max(inference_input.shape[1] - 1, 1)
        coordinates.append(display_coordinates)
        records.append(
            {
                'label': label,
                'input_power': power,
                'predicted_atom_count': int(len(display_coordinates)),
                'prediction_mean': float(native_prediction.mean()),
                'prediction_max': float(native_prediction.max()),
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.6), constrained_layout=True)
    for col, (label, display_input, _inference_input, _power) in enumerate(variants):
        _plot_clean_image(axes[0, col], display_input, label, cmap='gray')
        _add_physical_scale_bar(
            axes[0, col],
            display_input.shape,
            pixel_size_nm=source_pixel_size_nm,
            length_nm=float(args.scale_bar_length_nm),
            linewidth=float(args.scale_bar_linewidth),
        )
        _plot_clean_image(axes[1, col], display_view, '', cmap='gray')
        atom_coordinates = coordinates[col]
        if len(atom_coordinates):
            axes[1, col].scatter(
                atom_coordinates[:, 1],
                atom_coordinates[:, 0],
                s=args.marker_size,
                facecolors=args.blobnet_marker_color,
                edgecolors=args.marker_edge_color,
                linewidths=args.marker_linewidth,
                alpha=0.96,
            )
        _add_physical_scale_bar(
            axes[1, col],
            display_view.shape,
            pixel_size_nm=source_pixel_size_nm,
            length_nm=float(args.scale_bar_length_nm),
            linewidth=float(args.scale_bar_linewidth),
        )
        axes[1, col].text(
            0.04,
            0.04,
            f'n={len(atom_coordinates)}',
            transform=axes[1, col].transAxes,
            color='white',
            fontsize=9,
            ha='left',
            va='bottom',
            bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': 0.45, 'pad': 1.2},
        )
    axes[0, 0].set_ylabel('Model input', fontsize=AXIS_LABEL_SIZE)
    axes[1, 0].set_ylabel('Blob-Net detections', fontsize=AXIS_LABEL_SIZE)

    output_path = output_dir / 'fig-Blob-Net-3c-0063-sqrt-input.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'figure': str(output_path),
        'source_image': str(args.image),
        'image_selection': 'Final displayed DCFI(HAADF) series frame.',
        'velox_selection': velox_selection,
        'source_pixel_size_nm': source_pixel_size_nm,
        'field_of_view_nm': field_of_view_nm,
        'target_pixel_size_nm': float(args.target_pixel_size_nm),
        'native_inference_pixels': native_pixels,
        'localization_threshold_rel': float(args.localization_threshold_rel),
        'checkpoint': str(args.blobnet_checkpoint),
        'records': records,
        'processing': {
            **transform,
            'sqrt_stage': 'After DoG processing and [0, 1] normalization; immediately before inference.',
        },
    }
    (output_dir / 'fig-Blob-Net-3c-0063-sqrt-input.json').write_text(json.dumps(summary, indent=2))
    return output_path


def make_figure_3c_blobnet_cutoff_gallery(args: argparse.Namespace) -> Path:
    """Show a compact low-cutoff gallery for Blob-Net on one Velox HAADF image."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    model = _load_blobnet_model(args.blobnet_checkpoint, device, args.num_filters, args.dropout)

    image, source_pixel_size_nm, velox_selection = _load_velox_displayed_haadf(
        args.image,
        args.experimental_crop_size,
    )
    field_of_view_nm = float(args.experimental_crop_size) * source_pixel_size_nm
    native_pixels = max(1, int(round(field_of_view_nm / args.target_pixel_size_nm)))
    display_view, native_view, transform = _make_fixed_fov_resolution_view(
        image,
        dog_small=args.dog_small,
        dog_large=args.dog_large,
        display_size=args.experimental_crop_size,
        native_pixels=native_pixels,
    )
    native_prediction = _predict_tiled(
        model,
        native_view,
        device,
        args.tile_size,
        args.tile_overlap,
        args.batch_size,
    )
    cutoffs = np.linspace(args.cutoff_min, args.cutoff_max, args.steps, dtype=np.float64)

    coordinates: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        native_coordinates = extract_subpixel_peak_positions(
            native_prediction,
            threshold_rel=float(cutoff),
            min_distance=args.peak_min_distance,
            window_size=args.peak_window_size,
        )
        display_coordinates = np.asarray(native_coordinates, dtype=np.float32).copy()
        if len(display_coordinates):
            display_coordinates[:, 0] *= (display_view.shape[0] - 1) / max(native_view.shape[0] - 1, 1)
            display_coordinates[:, 1] *= (display_view.shape[1] - 1) / max(native_view.shape[1] - 1, 1)
        coordinates.append(display_coordinates)
        records.append(
            {
                'cutoff': float(cutoff),
                'predicted_atom_count': int(len(display_coordinates)),
            }
        )

    panel_count = len(cutoffs) + 1
    columns = 4
    rows = int(math.ceil(panel_count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows), constrained_layout=True)
    flat_axes = np.asarray(axes).reshape(-1)
    _plot_clean_image(flat_axes[0], display_view, 'Input', cmap='gray')
    _add_physical_scale_bar(
        flat_axes[0],
        display_view.shape,
        pixel_size_nm=source_pixel_size_nm,
        length_nm=float(args.scale_bar_length_nm),
        linewidth=float(args.scale_bar_linewidth),
    )

    for axis, cutoff, atom_coordinates in zip(flat_axes[1:], cutoffs, coordinates):
        _plot_clean_image(axis, display_view, f'Cutoff {cutoff:.2f}', cmap='gray')
        if len(atom_coordinates):
            axis.scatter(
                atom_coordinates[:, 1],
                atom_coordinates[:, 0],
                s=args.marker_size,
                facecolors=args.blobnet_marker_color,
                edgecolors=args.marker_edge_color,
                linewidths=args.marker_linewidth,
                alpha=0.96,
            )
        _add_physical_scale_bar(
            axis,
            display_view.shape,
            pixel_size_nm=source_pixel_size_nm,
            length_nm=float(args.scale_bar_length_nm),
            linewidth=float(args.scale_bar_linewidth),
        )
        axis.text(
            0.04,
            0.04,
            f'n={len(atom_coordinates)}',
            transform=axis.transAxes,
            color='white',
            fontsize=9,
            ha='left',
            va='bottom',
            bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': 0.45, 'pad': 1.2},
        )
    for axis in flat_axes[panel_count:]:
        axis.axis('off')

    output_path = output_dir / 'fig-Blob-Net-3c-0063-low-cutoff-gallery.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'figure': str(output_path),
        'source_image': str(args.image),
        'image_selection': 'Final displayed DCFI(HAADF) series frame.',
        'velox_selection': velox_selection,
        'source_pixel_size_nm': source_pixel_size_nm,
        'field_of_view_nm': field_of_view_nm,
        'target_pixel_size_nm': float(args.target_pixel_size_nm),
        'native_inference_pixels': native_pixels,
        'cutoff_range': [float(args.cutoff_min), float(args.cutoff_max)],
        'steps': int(args.steps),
        'checkpoint': str(args.blobnet_checkpoint),
        'records': records,
        'processing': transform,
    }
    (output_dir / 'fig-Blob-Net-3c-0063-low-cutoff-gallery.json').write_text(json.dumps(summary, indent=2))
    return output_path


def make_quasicrystal_scale_sweep(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    model = _load_blobnet_model(args.checkpoint, device, args.num_filters, args.dropout)

    source_image = _load_experimental_image(args.image)
    processed_image = gaussian_filter(source_image, args.dog_small) - gaussian_filter(source_image, args.dog_large)
    processed_image = _normalize_image(processed_image)
    fixed_fov = _center_crop_or_pad(processed_image, args.crop_size)
    scales = np.linspace(args.scale_min, args.scale_max, args.steps, dtype=np.float64)

    records: list[dict[str, Any]] = []
    native_views: list[np.ndarray] = []
    native_predictions: list[np.ndarray] = []
    display_predictions: list[np.ndarray] = []
    for scale in scales:
        n_pixels = max(1, int(round(args.crop_size * float(scale))))
        native_view = _normalize_image(_interpolate_image(fixed_fov, (n_pixels, n_pixels)))
        native_prediction = _predict_tiled(
            model,
            native_view,
            device,
            args.tile_size,
            args.tile_overlap,
            args.batch_size,
        )
        display_prediction = _interpolate_image(native_prediction, fixed_fov.shape)
        coordinates = extract_subpixel_peak_positions(
            display_prediction,
            threshold_rel=args.threshold_rel,
            min_distance=args.peak_min_distance,
            window_size=args.peak_window_size,
        )
        native_views.append(native_view)
        native_predictions.append(native_prediction)
        display_predictions.append(display_prediction)
        records.append(
            {
                'scale': float(scale),
                'native_height': int(native_view.shape[0]),
                'native_width': int(native_view.shape[1]),
                'display_height': int(display_prediction.shape[0]),
                'display_width': int(display_prediction.shape[1]),
                'predicted_peak_count': int(len(coordinates)),
                'prediction_mean': float(native_prediction.mean()),
                'prediction_max': float(native_prediction.max()),
            }
        )

    columns = 5
    rows = int(np.ceil(len(scales) / columns))
    global_prediction_max = max(float(prediction.max()) for prediction in display_predictions)

    overlay_figure, overlay_axes = plt.subplots(rows, columns, figsize=(15, 3.05 * rows), constrained_layout=True)
    heatmap_figure, heatmap_axes = plt.subplots(rows, columns, figsize=(15, 3.05 * rows), constrained_layout=True)
    overlay_axes = np.asarray(overlay_axes).reshape(-1)
    heatmap_axes = np.asarray(heatmap_axes).reshape(-1)
    for index, (record, prediction) in enumerate(zip(records, display_predictions)):
        title = f"{record['scale']:.3f}x | {record['native_width']} x {record['native_height']} px"
        overlay_axes[index].imshow(fixed_fov, cmap='gray', vmin=0.0, vmax=1.0)
        overlay_alpha = np.clip(prediction / max(global_prediction_max, 1e-8), 0.0, 1.0) * 0.90
        overlay_axes[index].imshow(
            prediction,
            cmap=MODEL_CMAPS['random'],
            vmin=0.0,
            vmax=global_prediction_max,
            alpha=overlay_alpha,
        )
        overlay_axes[index].set_title(title, fontsize=10)
        overlay_axes[index].set_xticks([])
        overlay_axes[index].set_yticks([])

        heatmap_axes[index].imshow(
            prediction,
            cmap=MODEL_CMAPS['random'],
            vmin=0.0,
            vmax=global_prediction_max,
        )
        heatmap_axes[index].set_title(title, fontsize=10)
        heatmap_axes[index].set_xticks([])
        heatmap_axes[index].set_yticks([])

    for axis in overlay_axes[len(scales) :]:
        axis.set_visible(False)
    for axis in heatmap_axes[len(scales) :]:
        axis.set_visible(False)

    overlay_path = output_dir / 'quasicrystal_scale_sweep_overlay.png'
    heatmap_path = output_dir / 'quasicrystal_scale_sweep_heatmaps.png'
    overlay_figure.savefig(overlay_path, dpi=args.dpi, bbox_inches='tight')
    heatmap_figure.savefig(heatmap_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(overlay_figure)
    plt.close(heatmap_figure)

    array_payload: dict[str, np.ndarray] = {
        'scales': scales.astype(np.float32),
        'fixed_fov': fixed_fov.astype(np.float32),
        'display_predictions': np.stack(display_predictions).astype(np.float32),
    }
    for index, (native_view, native_prediction) in enumerate(zip(native_views, native_predictions)):
        array_payload[f'model_input_{index:02d}'] = native_view.astype(np.float32)
        array_payload[f'native_prediction_{index:02d}'] = native_prediction.astype(np.float32)
    np.savez_compressed(output_dir / 'quasicrystal_scale_sweep_predictions.npz', **array_payload)
    summary = {
        'source_image': str(args.image),
        'checkpoint': str(args.checkpoint),
        'interpolation': 'scipy.interpolate.RegularGridInterpolator(method=linear)',
        'preprocessing': 'DoG background subtraction, percentile normalization, fixed center FOV, interpolation',
        'field_of_view': 'Constant across all scales; only the native inference pixel grid changes.',
        'dog_small': float(args.dog_small),
        'dog_large': float(args.dog_large),
        'crop_size': int(args.crop_size),
        'threshold_rel': float(args.threshold_rel),
        'overlay_path': str(overlay_path),
        'heatmap_path': str(heatmap_path),
        'records': records,
    }
    (output_dir / 'quasicrystal_scale_sweep.json').write_text(json.dumps(summary, indent=2))
    return overlay_path


def make_quasicrystal_common_pixel_size(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    model = _load_blobnet_model(args.checkpoint, device, args.num_filters, args.dropout)

    pixel_sizes_nm = [_read_channel_pixel_size_nm(path) for path in args.images]
    target_pixel_size_nm = (
        float(args.target_pixel_size_nm)
        if args.target_pixel_size_nm is not None
        else max(pixel_sizes_nm)
    )

    records: list[dict[str, Any]] = []
    display_views: list[np.ndarray] = []
    display_predictions: list[np.ndarray] = []
    native_views: list[np.ndarray] = []
    native_predictions: list[np.ndarray] = []
    for path, source_pixel_size_nm in zip(args.images, pixel_sizes_nm):
        source_image = _load_experimental_image(path)
        processed_image = gaussian_filter(source_image, args.dog_small) - gaussian_filter(source_image, args.dog_large)
        processed_image = _normalize_image(processed_image)
        display_view = _center_crop_or_pad(processed_image, args.crop_size)
        field_of_view_nm = float(args.crop_size) * source_pixel_size_nm
        native_pixels = max(1, int(round(field_of_view_nm / target_pixel_size_nm)))
        native_view = _normalize_image(_interpolate_image(display_view, (native_pixels, native_pixels)))
        native_prediction = _predict_tiled(
            model,
            native_view,
            device,
            args.tile_size,
            args.tile_overlap,
            args.batch_size,
        )
        display_prediction = _interpolate_image(native_prediction, display_view.shape)

        display_views.append(display_view)
        display_predictions.append(display_prediction)
        native_views.append(native_view)
        native_predictions.append(native_prediction)
        records.append(
            {
                'source_image': str(path),
                'source_pixel_size_nm': source_pixel_size_nm,
                'target_pixel_size_nm': target_pixel_size_nm,
                'field_of_view_nm': field_of_view_nm,
                'display_pixels': int(args.crop_size),
                'native_inference_pixels': native_pixels,
                'resolution_scale': float(native_pixels) / float(args.crop_size),
                'prediction_mean': float(native_prediction.mean()),
                'prediction_max': float(native_prediction.max()),
            }
        )

    figure, axes = plt.subplots(2, len(records), figsize=(4.2 * len(records), 7.2), constrained_layout=True)
    for column, (record, display_view, display_prediction) in enumerate(
        zip(records, display_views, display_predictions)
    ):
        axes[0, column].imshow(display_view, cmap='gray', vmin=0.0, vmax=1.0)
        axes[0, column].set_title(
            f"{Path(record['source_image']).name.split(' - ')[0]}\n"
            f"{record['source_pixel_size_nm']:.6f} nm/px native",
            fontsize=11,
        )
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        axes[1, column].imshow(
            display_prediction,
            cmap=MODEL_CMAPS['random'],
            vmin=0.0,
            vmax=max(float(display_prediction.max()), 1e-6),
        )
        axes[1, column].set_title(
            f"{record['native_inference_pixels']} x {record['native_inference_pixels']} px inference\n"
            f"{target_pixel_size_nm:.6f} nm/px common",
            fontsize=11,
        )
        axes[1, column].set_xticks([])
        axes[1, column].set_yticks([])

    output_path = output_dir / 'quasicrystal_common_pixel_size.png'
    figure.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(figure)

    array_payload: dict[str, np.ndarray] = {
        'source_pixel_sizes_nm': np.asarray(pixel_sizes_nm, dtype=np.float64),
        'target_pixel_size_nm': np.asarray(target_pixel_size_nm, dtype=np.float64),
        'display_inputs': np.stack(display_views).astype(np.float32),
        'display_predictions': np.stack(display_predictions).astype(np.float32),
    }
    for index, (native_view, native_prediction) in enumerate(zip(native_views, native_predictions)):
        array_payload[f'model_input_{index:02d}'] = native_view.astype(np.float32)
        array_payload[f'native_prediction_{index:02d}'] = native_prediction.astype(np.float32)
    np.savez_compressed(output_dir / 'quasicrystal_common_pixel_size.npz', **array_payload)

    summary = {
        'figure': str(output_path),
        'pixel_size_source': "pyTEMlib Channel_000 original_metadata['BinaryResult']['PixelSize']",
        'target_selection': 'Largest native pixel size, avoiding interpolation-based upsampling.',
        'field_of_view': 'Preserved independently for each 512 x 512 source crop.',
        'records': records,
    }
    (output_dir / 'quasicrystal_common_pixel_size.json').write_text(json.dumps(summary, indent=2))
    return output_path


def make_figure3_pixel_size_sweep(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    model = _load_blobnet_model(args.checkpoint, device, args.num_filters, args.dropout)

    source_pixel_sizes_nm = [_read_channel_pixel_size_nm(path) for path in args.images]
    display_views: list[np.ndarray] = []
    field_of_views_nm: list[float] = []
    for path, source_pixel_size_nm in zip(args.images, source_pixel_sizes_nm):
        source_image = _load_experimental_image(path)
        processed_image = gaussian_filter(source_image, args.dog_small) - gaussian_filter(source_image, args.dog_large)
        processed_image = _normalize_image(processed_image)
        display_views.append(_center_crop_or_pad(processed_image, args.crop_size))
        field_of_views_nm.append(float(args.crop_size) * source_pixel_size_nm)

    factors = np.linspace(args.factor_min, args.factor_max, args.steps, dtype=np.float64)
    sweep_predictions: list[list[np.ndarray]] = []
    native_inputs: list[list[np.ndarray]] = []
    native_predictions: list[list[np.ndarray]] = []
    records: list[dict[str, Any]] = []
    for factor in factors:
        target_pixel_size_nm = float(args.base_pixel_size_nm) * float(factor)
        factor_display_predictions: list[np.ndarray] = []
        factor_native_inputs: list[np.ndarray] = []
        factor_native_predictions: list[np.ndarray] = []
        native_pixel_counts: list[int] = []
        for display_view, field_of_view_nm in zip(display_views, field_of_views_nm):
            native_pixels = max(1, int(round(field_of_view_nm / target_pixel_size_nm)))
            native_view = _normalize_image(_interpolate_image(display_view, (native_pixels, native_pixels)))
            native_prediction = _predict_tiled(
                model,
                native_view,
                device,
                args.tile_size,
                args.tile_overlap,
                args.batch_size,
            )
            display_prediction = _interpolate_image(native_prediction, display_view.shape)
            factor_native_inputs.append(native_view)
            factor_native_predictions.append(native_prediction)
            factor_display_predictions.append(display_prediction)
            native_pixel_counts.append(native_pixels)

        native_inputs.append(factor_native_inputs)
        native_predictions.append(factor_native_predictions)
        sweep_predictions.append(factor_display_predictions)
        records.append(
            {
                'factor': float(factor),
                'target_pixel_size_nm': target_pixel_size_nm,
                'native_inference_pixels': native_pixel_counts,
            }
        )

    rows = int(np.ceil(args.steps / 5))
    columns = min(5, args.steps)
    overlay_figure, overlay_axes = plt.subplots(rows, columns, figsize=(15, 2.35 * rows), constrained_layout=True)
    heatmap_figure, heatmap_axes = plt.subplots(rows, columns, figsize=(15, 2.35 * rows), constrained_layout=True)
    overlay_axes = np.asarray(overlay_axes).reshape(-1)
    heatmap_axes = np.asarray(heatmap_axes).reshape(-1)
    combined_inputs = np.concatenate(display_views, axis=1)
    global_prediction_max = max(
        float(prediction.max())
        for factor_predictions in sweep_predictions
        for prediction in factor_predictions
    )

    for index, (record, factor_predictions) in enumerate(zip(records, sweep_predictions)):
        combined_prediction = np.concatenate(factor_predictions, axis=1)
        counts = '/'.join(str(value) for value in record['native_inference_pixels'])
        title = (
            f"{record['factor']:.3f}x | {record['target_pixel_size_nm']:.5f} nm/px\n"
            f'W/Q/T: {counts} px'
        )
        overlay_axes[index].imshow(combined_inputs, cmap='gray', vmin=0.0, vmax=1.0)
        overlay_alpha = np.clip(combined_prediction / max(global_prediction_max, 1e-8), 0.0, 1.0) * 0.90
        overlay_axes[index].imshow(
            combined_prediction,
            cmap=MODEL_CMAPS['random'],
            vmin=0.0,
            vmax=global_prediction_max,
            alpha=overlay_alpha,
        )
        heatmap_axes[index].imshow(
            combined_prediction,
            cmap=MODEL_CMAPS['random'],
            vmin=0.0,
            vmax=global_prediction_max,
        )
        for axis in (overlay_axes[index], heatmap_axes[index]):
            axis.set_title(title, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.axvline(args.crop_size - 0.5, color='#777777', linewidth=0.5)
            axis.axvline(2 * args.crop_size - 0.5, color='#777777', linewidth=0.5)
            for panel_index, label in enumerate(('W', 'Q', 'T')):
                axis.text(
                    (panel_index + 0.03) / 3.0,
                    0.95,
                    label,
                    transform=axis.transAxes,
                    ha='left',
                    va='top',
                    fontsize=7,
                    color='black',
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.7, 'pad': 0.8},
                )

    for axis in overlay_axes[len(records) :]:
        axis.set_visible(False)
    for axis in heatmap_axes[len(records) :]:
        axis.set_visible(False)

    overlay_path = output_dir / 'figure3_pixel_size_sweep_overlay.png'
    heatmap_path = output_dir / 'figure3_pixel_size_sweep_heatmaps.png'
    overlay_figure.savefig(overlay_path, dpi=args.dpi, bbox_inches='tight')
    heatmap_figure.savefig(heatmap_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(overlay_figure)
    plt.close(heatmap_figure)

    array_payload: dict[str, np.ndarray] = {
        'factors': factors.astype(np.float32),
        'source_pixel_sizes_nm': np.asarray(source_pixel_sizes_nm, dtype=np.float64),
        'base_pixel_size_nm': np.asarray(args.base_pixel_size_nm, dtype=np.float64),
        'display_inputs': np.stack(display_views).astype(np.float32),
        'display_predictions': np.asarray(sweep_predictions, dtype=np.float32),
    }
    for factor_index, (factor_inputs, factor_predictions) in enumerate(zip(native_inputs, native_predictions)):
        for image_index, (native_input, native_prediction) in enumerate(zip(factor_inputs, factor_predictions)):
            array_payload[f'model_input_{factor_index:02d}_{image_index:02d}'] = native_input.astype(np.float32)
            array_payload[f'native_prediction_{factor_index:02d}_{image_index:02d}'] = native_prediction.astype(np.float32)
    np.savez_compressed(output_dir / 'figure3_pixel_size_sweep.npz', **array_payload)

    summary = {
        'images': [str(path) for path in args.images],
        'image_order': ['Pristine monolayer MoS2', 'Al72Ni11Co17 quasicrystal', 'Sigma3 coherent twin grain boundary in FCC-Al'],
        'source_pixel_sizes_nm': source_pixel_sizes_nm,
        'field_of_views_nm': field_of_views_nm,
        'base_pixel_size_nm': float(args.base_pixel_size_nm),
        'factor_range': [float(args.factor_min), float(args.factor_max)],
        'steps': int(args.steps),
        'field_of_view': 'Preserved independently for every image throughout the sweep.',
        'records': records,
    }
    (output_dir / 'figure3_pixel_size_sweep.json').write_text(json.dumps(summary, indent=2))
    return overlay_path


def _experimental_figure_files(args: argparse.Namespace) -> list[tuple[str, Path]]:
    return [
        ('WS2 grain boundary', args.data_dir / 'high_angle_grain_boundary_monolayer_WS2.h5'),
        ('Quasicrystal', args.data_dir / 'Al72Ni11Co17_quasicrystal.h5'),
        ('Twin boundary', args.data_dir / 'Sigma3_coherent_twin_grain_boundary_FCC_Al.h5'),
    ]


def _poisson_noisy_image(image: np.ndarray, total_counts: float, rng: np.random.Generator) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    scaled = np.clip(image, 0.0, None)
    scaled = scaled / max(float(scaled.max()), 1e-8)
    counts = rng.poisson(scaled * float(total_counts)).astype(np.float32)
    noisy = counts / max(float(total_counts), 1e-8)
    return _normalize_image(noisy, low=0.0, high=99.8)


def _log_blob_positions(
    image: np.ndarray,
    sigma_px: float,
    threshold_rel: float,
    min_distance: int,
    peak_window_size: int,
    max_peaks: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    response = -gaussian_laplace(np.asarray(image, dtype=np.float32), sigma=float(sigma_px))
    response = _normalize_image(np.clip(response, 0.0, None), low=1.0, high=99.8)
    coordinates = extract_subpixel_peak_positions(
        response,
        threshold_rel=threshold_rel,
        min_distance=min_distance,
        window_size=peak_window_size,
        max_peaks=max_peaks,
    )
    return coordinates, response


def _prediction_positions(
    prediction: np.ndarray,
    threshold_rel: float,
    min_distance: int,
    peak_window_size: int,
    max_peaks: int | None,
) -> np.ndarray:
    return extract_subpixel_peak_positions(
        prediction,
        threshold_rel=threshold_rel,
        min_distance=min_distance,
        window_size=peak_window_size,
        max_peaks=max_peaks,
    )


def _plot_experimental_localization_overlay(
    ax: plt.Axes,
    image: np.ndarray,
    blobnet_coordinates: np.ndarray,
    log_coordinates: np.ndarray,
    title: str,
    show_legend: bool = False,
) -> None:
    ax.imshow(image, cmap='gray', vmin=0.0, vmax=1.0)
    blobnet_coordinates = np.asarray(blobnet_coordinates, dtype=np.float32).reshape(-1, 2)
    log_coordinates = np.asarray(log_coordinates, dtype=np.float32).reshape(-1, 2)
    if len(log_coordinates):
        ax.scatter(
            log_coordinates[:, 1],
            log_coordinates[:, 0],
            s=26,
            facecolors='none',
            edgecolors='#f28e2b',
            marker='o',
            linewidths=0.85,
            alpha=0.82,
            label='LoG',
        )
    if len(blobnet_coordinates):
        ax.scatter(
            blobnet_coordinates[:, 1],
            blobnet_coordinates[:, 0],
            s=10,
            c='#3fb950',
            marker='.',
            linewidths=0,
            alpha=0.9,
            label='BlobNet',
        )
    ax.set_title(title, fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_legend:
        handles = [
            Line2D([0], [0], marker='.', color='#3fb950', markersize=8, linestyle='None', label='BlobNet'),
            Line2D([0], [0], marker='o', color='#f28e2b', markerfacecolor='none', markersize=6, linestyle='None', label='LoG'),
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=9, frameon=True, borderpad=0.25, labelspacing=0.25)


def make_figure_3_localizations(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    model = _load_blobnet_model(args.checkpoint, device, args.num_filters, args.dropout)
    measurements = _read_experimental_feature_measurements(args.experimental_measurements)
    rng = np.random.default_rng(args.seed)

    panels: list[dict[str, Any]] = []
    for label, path in _experimental_figure_files(args):
        image = _load_experimental_image(path)
        model_input, transform = _make_feature_matched_experimental_view(
            image,
            measurements[path.stem],
            dog_small=args.dog_small,
            dog_large=args.dog_large,
            target_sigma_px=args.feature_match_sigma_px,
            crop_size=args.experimental_crop_size,
        )
        variants = [
            ('clean', model_input, None),
            (f'poisson_{int(args.poisson_counts)}', _poisson_noisy_image(model_input, args.poisson_counts, rng), args.poisson_counts),
            (
                f'poisson_{int(args.heavy_poisson_counts)}',
                _poisson_noisy_image(model_input, args.heavy_poisson_counts, rng),
                args.heavy_poisson_counts,
            ),
        ]
        predictions: dict[str, np.ndarray] = {}
        blobnet_coordinates: dict[str, np.ndarray] = {}
        log_coordinates: dict[str, np.ndarray] = {}
        for key, variant_image, _counts in variants:
            prediction = _predict_tiled(model, variant_image, device, args.tile_size, args.tile_overlap, args.batch_size)
            predictions[key] = prediction
            blobnet_coordinates[key] = _prediction_positions(
                prediction,
                threshold_rel=args.localization_threshold_rel,
                min_distance=args.peak_min_distance,
                peak_window_size=args.peak_window_size,
                max_peaks=args.max_peaks,
            )
            log_coordinates[key], _response = _log_blob_positions(
                variant_image,
                sigma_px=args.log_sigma_px,
                threshold_rel=args.log_threshold_rel,
                min_distance=args.log_min_distance,
                peak_window_size=args.peak_window_size,
                max_peaks=args.max_peaks,
            )

        panels.append(
            {
                'label': label,
                'path': str(path),
                'input': model_input,
                'transform': transform,
                'variants': variants,
                'predictions': predictions,
                'blobnet_coordinates': blobnet_coordinates,
                'log_coordinates': log_coordinates,
            }
        )

    fig, axes = plt.subplots(4, len(panels), figsize=(16, 14.5), constrained_layout=True)
    row_labels = [
        'Figure 3 input',
        'Clean localizations',
        f'Poisson localizations ({args.poisson_counts:g} counts)',
        f'Poisson localizations ({args.heavy_poisson_counts:g} counts)',
    ]
    variant_keys = ['clean', f'poisson_{int(args.poisson_counts)}', f'poisson_{int(args.heavy_poisson_counts)}']
    for col, panel in enumerate(panels):
        _plot_clean_image(axes[0, col], panel['input'], panel['label'], cmap='gray')
        if col == 0:
            axes[0, col].set_ylabel(row_labels[0], fontsize=AXIS_LABEL_SIZE)
        for row, variant_key in enumerate(variant_keys, start=1):
            variant_image = dict((key, image) for key, image, _counts in panel['variants'])[variant_key]
            _plot_experimental_localization_overlay(
                axes[row, col],
                variant_image,
                panel['blobnet_coordinates'][variant_key],
                panel['log_coordinates'][variant_key],
                '',
                show_legend=(row == 1 and col == len(panels) - 1),
            )
            if col == 0:
                axes[row, col].set_ylabel(row_labels[row], fontsize=AXIS_LABEL_SIZE)

    output_path = output_dir / 'figure3_experimental_localization_comparison.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    summary = []
    for panel in panels:
        panel_summary = {
            'label': panel['label'],
            'path': panel['path'],
            'shape_y': int(panel['input'].shape[0]),
            'shape_x': int(panel['input'].shape[1]),
            'display_processing': 'DoG background-subtracted, feature-sigma matched, center-cropped',
            'target_sigma_px': float(args.feature_match_sigma_px),
            **panel['transform'],
            'variants': {},
        }
        for variant_key in variant_keys:
            prediction = panel['predictions'][variant_key]
            panel_summary['variants'][variant_key] = {
                'blobnet_peaks': int(len(panel['blobnet_coordinates'][variant_key])),
                'log_peaks': int(len(panel['log_coordinates'][variant_key])),
                'prediction_mean': float(prediction.mean()),
                'prediction_max': float(prediction.max()),
            }
        summary.append(panel_summary)
    (output_dir / 'figure3_experimental_localization_comparison.json').write_text(json.dumps(summary, indent=2))
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
        background_range = FIGURE2_FIXED_NOISE_PARAMETERS['background_range']
        gradient_range = FIGURE2_FIXED_NOISE_PARAMETERS['gradient_range']
        inhomogeneous_background_range = FIGURE2_FIXED_NOISE_PARAMETERS['inhomogeneous_background_range']
        inhomogeneous_background_sigma_fraction_range = FIGURE2_FIXED_NOISE_PARAMETERS[
            'inhomogeneous_background_sigma_fraction_range'
        ]
        low_frequency_noise_range = FIGURE2_FIXED_NOISE_PARAMETERS['low_frequency_noise_range']
        low_frequency_sigma_fraction_range = FIGURE2_FIXED_NOISE_PARAMETERS['low_frequency_sigma_fraction_range']
        read_noise_std_range = FIGURE2_FIXED_NOISE_PARAMETERS['read_noise_std_range']
        blur_sigma_range = FIGURE2_FIXED_NOISE_PARAMETERS['blur_sigma_range']
    else:
        background_range = (0.04, 0.32)
        gradient_range = (-0.08, 0.08)
        inhomogeneous_background_range = (0.07, 0.18)
        inhomogeneous_background_sigma_fraction_range = (0.16, 0.42)
        low_frequency_noise_range = (0.05, 0.20)
        low_frequency_sigma_fraction_range = (0.05, 0.14)
        read_noise_std_range = (0.08, 0.22)
        blur_sigma_range = (0.15, 0.85)
    return ImageFormationConfig(
        image_shape=shape,
        sigma_range=(sigma_min, sigma_max),
        intensity_range=(0.2, 1.0),
        target_sigma=2.0,
        background_range=background_range,
        gradient_range=gradient_range,
        inhomogeneous_background_range=inhomogeneous_background_range,
        inhomogeneous_background_sigma_fraction_range=inhomogeneous_background_sigma_fraction_range,
        low_frequency_noise_range=low_frequency_noise_range,
        low_frequency_sigma_fraction_range=low_frequency_sigma_fraction_range,
        read_noise_std_range=read_noise_std_range,
        total_counts_range=total_counts_range,
        counts_per_pixel_range=None,
        blur_sigma_range=blur_sigma_range,
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
    carbon_sigma = 0.5 * (sigma_min + sigma_max)
    sigmas = np.full((len(coordinates_yx),), carbon_sigma, dtype=np.float32)
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
            'carbon_intensity': 0.78,
            'carbon_sigma_px': float(carbon_sigma),
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


def _image_border_mask_for_figure(coordinates: np.ndarray, shape: tuple[int, int], border_px: float) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
    if len(coordinates) == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (coordinates[:, 0] >= float(border_px))
        & (coordinates[:, 0] < float(shape[0]) - float(border_px))
        & (coordinates[:, 1] >= float(border_px))
        & (coordinates[:, 1] < float(shape[1]) - float(border_px))
    )


def _localization_classes_for_figure_with_image_border_exclusion(
    prediction: np.ndarray,
    true_coordinates: np.ndarray,
    threshold_rel: float,
    min_distance: int,
    peak_window_size: int,
    match_distance: float,
    shape: tuple[int, int],
    border_px: float,
) -> dict[str, Any]:
    predicted_coordinates = extract_subpixel_peak_positions(
        prediction,
        threshold_rel=threshold_rel,
        min_distance=min_distance,
        window_size=peak_window_size,
    )
    predicted_coordinates = predicted_coordinates[_image_border_mask_for_figure(predicted_coordinates, shape, border_px)]
    true_coordinates = np.asarray(true_coordinates, dtype=np.float32).reshape(-1, 2)
    true_coordinates = true_coordinates[_image_border_mask_for_figure(true_coordinates, shape, border_px)]
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


def make_ws2_edge_comparison(args: argparse.Namespace) -> Path:
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

    output_path = output_dir / 'ws2_edge_edge_model_comparison.png'
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
    (output_dir / 'ws2_edge_edge_model_comparison.json').write_text(json.dumps(summary, indent=2))
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


def make_figure_2(args: argparse.Namespace) -> Path:
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
    figure2_counts = (64.0, 64.0)
    figure2_image_border_exclusion_px = 10.0
    cases = [
        ('mos2_edge', _make_mos2_edge_record(shape, args.seed, sigma_range, total_counts_range=figure2_counts, quiet_background=True)),
        ('srtio3_edge', _make_sto_edge_record(shape, args.seed + 101, sigma_range, total_counts_range=figure2_counts, quiet_background=True)),
        ('graphene_rattled_edge', _make_graphene_rattled_edge_record(shape, args.seed + 202, sigma_range, total_counts_range=figure2_counts, quiet_background=True)),
    ]

    with_offsets = getattr(args, 'with_offset_diagnostics', False)
    if with_offsets:
        fig = plt.figure(figsize=(27, 8.8))
        grid = fig.add_gridspec(len(cases), 8, left=0.025, right=0.99, top=0.985,
                                bottom=0.075, wspace=0.38, hspace=0.28)
    else:
        fig = plt.figure(figsize=(20.0, 11.4), constrained_layout=True)
        grid = fig.add_gridspec(len(cases), 5, width_ratios=[1.05] + [1.0] * 4,
                                wspace=0.05, hspace=0.05)
    summary: dict[str, Any] = {
        'output_path': str(output_dir / 'figure2_edge_lattice_model_diagnostics.png'),
        'seed': int(args.seed),
        'shape': [int(args.height), int(args.width)],
        'feature_sigma_range_px': [float(args.feature_sigma_min), float(args.feature_sigma_max)],
        'poisson_total_counts_range': [float(figure2_counts[0]), float(figure2_counts[1])],
        'background_profile': 'fixed_noise_poisson_count_64_lowfreq_0.08_no_gradient',
        'noise_parameters': {
            key: [float(value[0]), float(value[1])]
            for key, value in FIGURE2_FIXED_NOISE_PARAMETERS.items()
        },
        'threshold_selection_note': FIGURE2_THRESHOLD_NOTE,
        'localization_settings': {
            'threshold_rel': float(args.localization_threshold_rel),
            'match_distance_px': float(args.localization_match_distance),
            'peak_min_distance_px': int(args.peak_min_distance),
            'peak_window_size_px': int(args.peak_window_size),
            'image_border_exclusion_px': float(figure2_image_border_exclusion_px),
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
        case_threshold = FIGURE2_TUNED_THRESHOLDS[case_key]
        case_thresholds = {spec.key: case_threshold for spec in models}
        case_localization = {
            spec.key: _localization_classes_for_figure_with_image_border_exclusion(
                case_predictions[spec.key],
                coordinates,
                threshold_rel=case_thresholds[spec.key],
                min_distance=args.peak_min_distance,
                peak_window_size=args.peak_window_size,
                match_distance=args.localization_match_distance,
                shape=shape,
                border_px=figure2_image_border_exclusion_px,
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

        if with_offsets:
            for column, spec in enumerate(models, start=5):
                classes = case_localization[spec.key]
                offsets = (classes['matched_predicted'] - classes['true_positives'])[:, ::-1]
                ax = fig.add_subplot(grid[row, column])
                if len(offsets):
                    ax.scatter(
                        offsets[:, 0], offsets[:, 1],
                        s=9, c=MODEL_COLORS[spec.key], alpha=0.78,
                        linewidths=0, rasterized=True,
                    )
                ax.axhline(0, color='white', linewidth=0.7, alpha=0.65)
                ax.axvline(0, color='white', linewidth=0.7, alpha=0.65)
                ax.set(xlim=(-args.offset_range, args.offset_range),
                       ylim=(-args.offset_range, args.offset_range), aspect='equal', facecolor='#17121f',
                       xticks=[-1, 0, 1], yticks=[-1, 0, 1])
                ax.tick_params(labelsize=AXIS_TICK_SIZE)
                metrics = _localization_metric_summary(classes)
                rmse_label = 'N/A' if metrics['rmse'] is None else f"{metrics['rmse']:.2f}px"
                ax.text(0.04, 0.96, f"F1={metrics['f1']:.3f}\nRMSE={rmse_label}",
                        transform=ax.transAxes, ha='left', va='top', color='white', fontsize=ANNOTATION_SIZE)
                if row == 0:
                    ax.set_title({'square': 'Square-Net', 'hexagonal': 'Hex-Net', 'random': 'Blob-Net'}[spec.key],
                                 fontsize=AXIS_LABEL_SIZE)
                if row == len(cases) - 1:
                    ax.set_xlabel('x offset (px)')
                if column == 5:
                    ax.set_ylabel('y offset (px)')

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

    output_path = output_dir / 'figure2_edge_lattice_model_diagnostics.png'
    if with_offsets:
        output_path = output_dir / 'fig-Blob-Net-2-with-diagnostics-draft.png'
        summary['diagnostics_note'] = ('Offsets and F1/RMSE use the displayed example in each row, '
                                       'rendered as model-colored scatter without a histogram underlay, '
                                       'with the same thresholds and 10 px border exclusion as its TP/FP/FN panels. '
                                       'Offsets are predicted minus ground truth; RMSE uses all matched points, '
                                       'including offsets outside the displayed +/-2 px window.')
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    summary['output_path'] = str(output_path)
    output_path.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    (output_dir / 'figure2_threshold_note.txt').write_text(FIGURE2_THRESHOLD_NOTE + '\n')
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


def make_figure_4(args: argparse.Namespace) -> Path:
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

    output_path = output_dir / 'figure4_scale_spacing_robustness.png'
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
    figure1.add_argument('--square-checkpoint', type=Path, default=repo_root / 'artifacts/manuscript_models/square/unet_best.pth')
    figure1.add_argument('--hexagonal-checkpoint', type=Path, default=repo_root / 'artifacts/manuscript_models/hexagonal/unet_best.pth')
    figure1.add_argument('--random-checkpoint', type=Path, default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth')
    figure1.add_argument('--seed', type=int, default=0)
    figure1.add_argument('--offset-samples', type=int, default=32)
    figure1.add_argument('--batch-size', type=int, default=4)
    figure1.add_argument('--threshold-rel', type=float, default=0.35)
    figure1.add_argument('--match-distance', type=float, default=3.0)
    figure1.add_argument('--offset-range', type=float, default=2.0)
    figure1.add_argument('--offset-bins', type=int, default=48)
    figure1.set_defaults(func=make_figure_1)

    figure3 = subparsers.add_parser('figure3', help='Experimental HAADF input/output figure.')
    _add_shared_arguments(figure3)
    _add_model_arguments(figure3)
    figure3.add_argument(
        '--checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/figure3_random/unet_best.pth',
    )
    figure3.add_argument('--data-dir', type=Path, default=repo_root / 'experimental_data')
    figure3.add_argument('--tile-size', type=int, default=256)
    figure3.add_argument('--tile-overlap', type=int, default=64)
    figure3.add_argument('--batch-size', type=int, default=4)
    figure3.add_argument('--dog-small', type=float, default=1.0)
    figure3.add_argument('--dog-large', type=float, default=20.0)
    figure3.add_argument('--experimental-measurements', type=Path, default=repo_root / 'outputs/experimental_feature_measurements_local/experimental_feature_measurements.json')
    figure3.add_argument('--feature-match-sigma-px', type=float, default=2.9)
    figure3.add_argument('--experimental-crop-size', type=int, default=512)
    figure3.add_argument(
        '--quasicrystal-image',
        type=Path,
        default=repo_root / 'experimental_data/Al72Ni11Co17_quasicrystal.h5',
    )
    figure3.add_argument('--target-pixel-size-nm', type=float, default=0.027193128874910695)
    figure3.add_argument(
        '--fourth-image', type=Path,
        default=repo_root / 'experimental_data/high_angle_grain_boundary_monolayer_WS2.h5',
    )
    figure3.add_argument('--localization-threshold-rel', type=float, default=0.35)
    figure3.add_argument('--peak-min-distance', type=int, default=3)
    figure3.add_argument('--peak-window-size', type=int, default=5)
    figure3.add_argument('--marker-size', type=float, default=40.0)
    figure3.add_argument('--marker-linewidth', type=float, default=1.8)
    figure3.add_argument('--marker-color', default='#D55E00')
    figure3.add_argument('--marker-edge-color', default='white')
    figure3.add_argument('--hexagonal-checkpoint', type=Path,
                         default=repo_root / 'artifacts/manuscript_models/figure3_hexagonal/unet_best.pth')
    figure3.add_argument(
        '--ws2-hexagonal-pixel-size-factor', type=float, default=0.70,
        help='Hex-Net pixel-size factor for the first-column WS2 image; Blob-Net remains at 1.0.',
    )
    figure3.add_argument(
        '--ws2-hexagonal-threshold-rel', type=float, default=0.30,
        help='Hex-Net localization cutoff for the first-column WS2 image.',
    )
    figure3.add_argument('--save-pdf', action='store_true')
    figure3.add_argument('--disagreement-marker', choices=['x', '+', 's'], default='+')
    figure3.add_argument('--disagreement-marker-size-scale', type=float, default=1.2)
    figure3.add_argument('--output-name', default='figure3_experimental_haadf_outputs.png')
    figure3.add_argument('--agreement-radius-nm', type=float, default=0.06)
    figure3.add_argument('--both-color', default=MODEL_COLORS['random'])
    figure3.add_argument('--hex-only-color', default=MODEL_COLORS['hexagonal'])
    figure3.add_argument('--blob-only-color', default=MODEL_COLORS['square'])
    figure3.add_argument('--scale-bar-length-nm', type=float, default=1.0)
    figure3.add_argument('--scale-bar-linewidth', type=float, default=4.0)
    figure3.set_defaults(func=make_figure_3)

    figure3b = subparsers.add_parser(
        'figure3b',
        help='Experimental Figure 3 comparison with Blob-Net and hexagonal-model localization rows.',
    )
    _add_shared_arguments(figure3b)
    _add_model_arguments(figure3b)
    figure3b.add_argument(
        '--blobnet-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth',
    )
    figure3b.add_argument(
        '--hexagonal-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/hexagonal/unet_best.pth',
    )
    figure3b.add_argument('--data-dir', type=Path, default=repo_root / 'experimental_data')
    figure3b.add_argument('--tile-size', type=int, default=256)
    figure3b.add_argument('--tile-overlap', type=int, default=64)
    figure3b.add_argument('--batch-size', type=int, default=4)
    figure3b.add_argument('--dog-small', type=float, default=1.0)
    figure3b.add_argument('--dog-large', type=float, default=20.0)
    figure3b.add_argument('--experimental-crop-size', type=int, default=512)
    figure3b.add_argument(
        '--quasicrystal-image',
        type=Path,
        default=repo_root / 'experimental_data/Al72Ni11Co17_quasicrystal.h5',
    )
    figure3b.add_argument('--target-pixel-size-nm', type=float, default=0.027193128874910695)
    figure3b.add_argument('--localization-threshold-rel', type=float, default=0.35)
    figure3b.add_argument('--peak-min-distance', type=int, default=3)
    figure3b.add_argument('--peak-window-size', type=int, default=5)
    figure3b.add_argument('--marker-size', type=float, default=40.0)
    figure3b.add_argument('--marker-linewidth', type=float, default=0.2)
    figure3b.add_argument('--marker-edge-color', default='white')
    figure3b.add_argument('--blobnet-marker-color', default='#D55E00')
    figure3b.add_argument('--hexagonal-marker-color', default=MODEL_COLORS['hexagonal'])
    figure3b.add_argument('--scale-bar-length-nm', type=float, default=1.0)
    figure3b.add_argument('--scale-bar-linewidth', type=float, default=4.0)
    figure3b.set_defaults(func=make_figure_3b)

    figure3c = subparsers.add_parser(
        'figure3c',
        help='Figure 3b-style comparison for multi-frame Velox HAADF images.',
    )
    _add_shared_arguments(figure3c)
    _add_model_arguments(figure3c)
    figure3c.add_argument(
        '--blobnet-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth',
    )
    figure3c.add_argument(
        '--hexagonal-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/hexagonal/unet_best.pth',
    )
    figure3c.add_argument('--images', type=Path, nargs='+', required=True)
    figure3c.add_argument('--tile-size', type=int, default=256)
    figure3c.add_argument('--tile-overlap', type=int, default=64)
    figure3c.add_argument('--batch-size', type=int, default=4)
    figure3c.add_argument('--dog-small', type=float, default=1.0)
    figure3c.add_argument('--dog-large', type=float, default=20.0)
    figure3c.add_argument('--experimental-crop-size', type=int, default=512)
    figure3c.add_argument('--target-pixel-size-nm', type=float, default=0.027193128874910695)
    figure3c.add_argument('--localization-threshold-rel', type=float, default=0.35)
    figure3c.add_argument('--peak-min-distance', type=int, default=3)
    figure3c.add_argument('--peak-window-size', type=int, default=5)
    figure3c.add_argument('--marker-size', type=float, default=40.0)
    figure3c.add_argument('--marker-linewidth', type=float, default=0.2)
    figure3c.add_argument('--marker-edge-color', default='white')
    figure3c.add_argument('--blobnet-marker-color', default='#D55E00')
    figure3c.add_argument('--hexagonal-marker-color', default=MODEL_COLORS['hexagonal'])
    figure3c.add_argument('--scale-bar-length-nm', type=float, default=1.0)
    figure3c.add_argument('--scale-bar-linewidth', type=float, default=4.0)
    figure3c.set_defaults(func=make_figure_3c)

    figure3c_cutoffs = subparsers.add_parser(
        'figure3c-cutoff-sweep',
        help='Sweep localization cutoffs for both Figure 3C models on one Velox HAADF image.',
    )
    _add_shared_arguments(figure3c_cutoffs)
    _add_model_arguments(figure3c_cutoffs)
    figure3c_cutoffs.add_argument(
        '--blobnet-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth',
    )
    figure3c_cutoffs.add_argument(
        '--hexagonal-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/hexagonal/unet_best.pth',
    )
    figure3c_cutoffs.add_argument('--image', type=Path, required=True)
    figure3c_cutoffs.add_argument('--tile-size', type=int, default=256)
    figure3c_cutoffs.add_argument('--tile-overlap', type=int, default=64)
    figure3c_cutoffs.add_argument('--batch-size', type=int, default=4)
    figure3c_cutoffs.add_argument('--dog-small', type=float, default=1.0)
    figure3c_cutoffs.add_argument('--dog-large', type=float, default=20.0)
    figure3c_cutoffs.add_argument('--experimental-crop-size', type=int, default=512)
    figure3c_cutoffs.add_argument('--target-pixel-size-nm', type=float, default=0.027193128874910695)
    figure3c_cutoffs.add_argument('--cutoff-min', type=float, default=0.10)
    figure3c_cutoffs.add_argument('--cutoff-max', type=float, default=0.40)
    figure3c_cutoffs.add_argument('--steps', type=int, default=7)
    figure3c_cutoffs.add_argument('--peak-min-distance', type=int, default=3)
    figure3c_cutoffs.add_argument('--peak-window-size', type=int, default=5)
    figure3c_cutoffs.add_argument('--marker-size', type=float, default=40.0)
    figure3c_cutoffs.add_argument('--marker-linewidth', type=float, default=0.2)
    figure3c_cutoffs.add_argument('--marker-edge-color', default='white')
    figure3c_cutoffs.add_argument('--blobnet-marker-color', default='#D55E00')
    figure3c_cutoffs.add_argument('--hexagonal-marker-color', default=MODEL_COLORS['hexagonal'])
    figure3c_cutoffs.add_argument('--scale-bar-length-nm', type=float, default=1.0)
    figure3c_cutoffs.add_argument('--scale-bar-linewidth', type=float, default=4.0)
    figure3c_cutoffs.set_defaults(func=make_figure_3c_cutoff_sweep)

    figure3c_agreement_cutoffs = subparsers.add_parser(
        'figure3c-agreement-cutoff-sweep',
        help='Sweep cutoffs and compare matched Blob-Net and Hex-Net predictions on the 0063 crop.',
    )
    _add_shared_arguments(figure3c_agreement_cutoffs)
    _add_model_arguments(figure3c_agreement_cutoffs)
    figure3c_agreement_cutoffs.add_argument(
        '--blobnet-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth',
    )
    figure3c_agreement_cutoffs.add_argument(
        '--hexagonal-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/hexagonal/unet_best.pth',
    )
    figure3c_agreement_cutoffs.add_argument(
        '--image',
        type=Path,
        default=repo_root / 'experimental_data/high_angle_grain_boundary_monolayer_WS2.h5',
    )
    figure3c_agreement_cutoffs.add_argument(
        '--standard-emd', action='store_true',
        help='Read the standard experimental Channel_000 image instead of a Velox displayed-series frame.',
    )
    figure3c_agreement_cutoffs.add_argument('--tile-size', type=int, default=256)
    figure3c_agreement_cutoffs.add_argument('--tile-overlap', type=int, default=64)
    figure3c_agreement_cutoffs.add_argument('--batch-size', type=int, default=4)
    figure3c_agreement_cutoffs.add_argument('--dog-small', type=float, default=1.0)
    figure3c_agreement_cutoffs.add_argument('--dog-large', type=float, default=20.0)
    figure3c_agreement_cutoffs.add_argument('--experimental-crop-size', type=int, default=512)
    figure3c_agreement_cutoffs.add_argument('--target-pixel-size-nm', type=float, default=0.027193128874910695)
    figure3c_agreement_cutoffs.add_argument('--cutoff-min', type=float, default=0.10)
    figure3c_agreement_cutoffs.add_argument('--cutoff-max', type=float, default=0.40)
    figure3c_agreement_cutoffs.add_argument('--steps', type=int, default=7)
    figure3c_agreement_cutoffs.add_argument(
        '--cutoffs', type=_parse_float_list, default=None,
        help='Optional explicit cutoff list, overriding cutoff-min, cutoff-max, and steps.',
    )
    figure3c_agreement_cutoffs.add_argument(
        '--fixed-blobnet-cutoff', type=float, default=None,
        help='Keep Blob-Net at this cutoff while sweeping the listed cutoffs only for Hex-Net.',
    )
    figure3c_agreement_cutoffs.add_argument('--agreement-radius-nm', type=float, default=0.06)
    figure3c_agreement_cutoffs.add_argument('--peak-min-distance', type=int, default=3)
    figure3c_agreement_cutoffs.add_argument('--peak-window-size', type=int, default=5)
    figure3c_agreement_cutoffs.add_argument('--marker-size', type=float, default=40.0)
    figure3c_agreement_cutoffs.add_argument('--marker-linewidth', type=float, default=1.8)
    figure3c_agreement_cutoffs.add_argument('--both-color', default=MODEL_COLORS['random'])
    figure3c_agreement_cutoffs.add_argument('--hex-only-color', default=MODEL_COLORS['hexagonal'])
    figure3c_agreement_cutoffs.add_argument('--blob-only-color', default=MODEL_COLORS['square'])
    figure3c_agreement_cutoffs.add_argument('--scale-bar-length-nm', type=float, default=1.0)
    figure3c_agreement_cutoffs.add_argument('--scale-bar-linewidth', type=float, default=4.0)
    figure3c_agreement_cutoffs.set_defaults(func=make_figure_3c_agreement_cutoff_sweep)

    figure3c_normalization = subparsers.add_parser(
        'figure3c-normalization-sweep',
        help='Sweep robust upper-percentile clipping for both models on the 0063 crop.',
    )
    _add_shared_arguments(figure3c_normalization)
    _add_model_arguments(figure3c_normalization)
    figure3c_normalization.add_argument(
        '--blobnet-checkpoint', type=Path,
        default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth',
    )
    figure3c_normalization.add_argument(
        '--hexagonal-checkpoint', type=Path,
        default=repo_root / 'artifacts/manuscript_models/hexagonal/unet_best.pth',
    )
    figure3c_normalization.add_argument(
        '--image', type=Path,
        default=repo_root / 'experimental_data/high_angle_grain_boundary_monolayer_WS2.h5',
    )
    figure3c_normalization.add_argument('--tile-size', type=int, default=256)
    figure3c_normalization.add_argument('--tile-overlap', type=int, default=64)
    figure3c_normalization.add_argument('--batch-size', type=int, default=4)
    figure3c_normalization.add_argument('--dog-small', type=float, default=1.0)
    figure3c_normalization.add_argument('--dog-large', type=float, default=20.0)
    figure3c_normalization.add_argument('--experimental-crop-size', type=int, default=512)
    figure3c_normalization.add_argument('--target-pixel-size-nm', type=float, default=0.027193128874910695)
    figure3c_normalization.add_argument('--lower-percentile', type=float, default=1.0)
    figure3c_normalization.add_argument(
        '--upper-percentiles', type=_parse_float_list,
        default=[99.8, 99.5, 99.0, 98.0, 97.0],
    )
    figure3c_normalization.add_argument(
        '--gammas', type=_parse_float_list, default=None,
        help='Optional intensity powers, one per upper percentile (1.0 is linear; lower values brighten dim sites).',
    )
    figure3c_normalization.add_argument('--localization-threshold-rel', type=float, default=0.35)
    figure3c_normalization.add_argument('--agreement-radius-nm', type=float, default=0.06)
    figure3c_normalization.add_argument('--peak-min-distance', type=int, default=3)
    figure3c_normalization.add_argument('--peak-window-size', type=int, default=5)
    figure3c_normalization.add_argument('--marker-size', type=float, default=40.0)
    figure3c_normalization.add_argument('--marker-linewidth', type=float, default=1.8)
    figure3c_normalization.add_argument('--both-color', default=MODEL_COLORS['random'])
    figure3c_normalization.add_argument('--hex-only-color', default=MODEL_COLORS['hexagonal'])
    figure3c_normalization.add_argument('--blob-only-color', default=MODEL_COLORS['square'])
    figure3c_normalization.add_argument('--scale-bar-length-nm', type=float, default=1.0)
    figure3c_normalization.add_argument('--scale-bar-linewidth', type=float, default=4.0)
    figure3c_normalization.set_defaults(func=make_figure_3c_normalization_sweep)

    figure3c_pixel_sizes = subparsers.add_parser(
        'figure3c-pixel-size-sweep',
        help='Sweep physical pixel sizes for both Figure 3C models while keeping the field of view fixed.',
    )
    _add_shared_arguments(figure3c_pixel_sizes)
    _add_model_arguments(figure3c_pixel_sizes)
    figure3c_pixel_sizes.add_argument(
        '--blobnet-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth',
    )
    figure3c_pixel_sizes.add_argument(
        '--hexagonal-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/hexagonal/unet_best.pth',
    )
    figure3c_pixel_sizes.add_argument('--image', type=Path, required=True)
    figure3c_pixel_sizes.add_argument(
        '--standard-emd', action='store_true',
        help='Read the standard experimental Channel_000 image instead of a Velox displayed-series frame.',
    )
    figure3c_pixel_sizes.add_argument('--tile-size', type=int, default=256)
    figure3c_pixel_sizes.add_argument('--tile-overlap', type=int, default=64)
    figure3c_pixel_sizes.add_argument('--batch-size', type=int, default=4)
    figure3c_pixel_sizes.add_argument('--dog-small', type=float, default=1.0)
    figure3c_pixel_sizes.add_argument('--dog-large', type=float, default=20.0)
    figure3c_pixel_sizes.add_argument('--experimental-crop-size', type=int, default=512)
    figure3c_pixel_sizes.add_argument('--base-pixel-size-nm', type=float, default=0.05438625774982139)
    figure3c_pixel_sizes.add_argument(
        '--pixel-size-factors',
        type=_parse_float_list,
        default=[0.25, 0.5, 1.0],
    )
    figure3c_pixel_sizes.add_argument(
        '--fixed-blobnet-factor', type=float, default=None,
        help='Keep Blob-Net at this factor while sweeping the listed factors only for Hex-Net.',
    )
    figure3c_pixel_sizes.add_argument('--localization-threshold-rel', type=float, default=0.35)
    figure3c_pixel_sizes.add_argument('--peak-min-distance', type=int, default=3)
    figure3c_pixel_sizes.add_argument('--peak-window-size', type=int, default=5)
    figure3c_pixel_sizes.add_argument('--marker-size', type=float, default=40.0)
    figure3c_pixel_sizes.add_argument('--marker-linewidth', type=float, default=0.2)
    figure3c_pixel_sizes.add_argument('--marker-edge-color', default='white')
    figure3c_pixel_sizes.add_argument('--blobnet-marker-color', default='#D55E00')
    figure3c_pixel_sizes.add_argument('--hexagonal-marker-color', default=MODEL_COLORS['hexagonal'])
    figure3c_pixel_sizes.add_argument('--combined-overlay', action='store_true')
    figure3c_pixel_sizes.add_argument('--comparison-hexagonal-marker-color', default='#56B4E9')
    figure3c_pixel_sizes.add_argument('--agreement-overlay', action='store_true')
    figure3c_pixel_sizes.add_argument('--agreement-radius-nm', type=float, default=0.06)
    figure3c_pixel_sizes.add_argument('--agreement-marker-linewidth', type=float, default=1.8)
    figure3c_pixel_sizes.add_argument('--both-color', default=MODEL_COLORS['random'])
    figure3c_pixel_sizes.add_argument('--hex-only-color', default=MODEL_COLORS['hexagonal'])
    figure3c_pixel_sizes.add_argument('--blob-only-color', default=MODEL_COLORS['square'])
    figure3c_pixel_sizes.add_argument('--scale-bar-length-nm', type=float, default=1.0)
    figure3c_pixel_sizes.add_argument('--scale-bar-linewidth', type=float, default=4.0)
    figure3c_pixel_sizes.set_defaults(func=make_figure_3c_pixel_size_sweep)

    ws2_edge_hex_grid = subparsers.add_parser(
        'figure3-ws2-hexnet-pixel-cutoff-grid',
        help='Grid Hex-Net WS2 overlays by physical pixel-size factor and localization cutoff.',
    )
    _add_shared_arguments(ws2_edge_hex_grid)
    _add_model_arguments(ws2_edge_hex_grid)
    ws2_edge_hex_grid.add_argument(
        '--hexagonal-checkpoint', type=Path,
        default=repo_root / 'artifacts/manuscript_models/hexagonal/unet_best.pth',
    )
    ws2_edge_hex_grid.add_argument(
        '--image', type=Path, default=repo_root / 'experimental_data/pristine_monolayer_MoS2.h5',
    )
    ws2_edge_hex_grid.add_argument('--tile-size', type=int, default=256)
    ws2_edge_hex_grid.add_argument('--tile-overlap', type=int, default=64)
    ws2_edge_hex_grid.add_argument('--batch-size', type=int, default=4)
    ws2_edge_hex_grid.add_argument('--dog-small', type=float, default=1.0)
    ws2_edge_hex_grid.add_argument('--dog-large', type=float, default=20.0)
    ws2_edge_hex_grid.add_argument('--experimental-crop-size', type=int, default=512)
    ws2_edge_hex_grid.add_argument('--base-pixel-size-nm', type=float, default=0.027193128874910695)
    ws2_edge_hex_grid.add_argument(
        '--pixel-size-factors', type=_parse_float_list,
        default=[0.70, 0.75, 0.80, 0.85, 0.90, 1.00],
    )
    ws2_edge_hex_grid.add_argument(
        '--cutoffs', type=_parse_float_list, default=[0.01, 0.05, 0.10, 0.20, 0.35],
    )
    ws2_edge_hex_grid.add_argument('--peak-min-distance', type=int, default=3)
    ws2_edge_hex_grid.add_argument('--peak-window-size', type=int, default=5)
    ws2_edge_hex_grid.add_argument('--marker-size', type=float, default=30.0)
    ws2_edge_hex_grid.add_argument('--marker-linewidth', type=float, default=1.4)
    ws2_edge_hex_grid.add_argument('--hexnet-color', default=MODEL_COLORS['hexagonal'])
    ws2_edge_hex_grid.add_argument('--scale-bar-length-nm', type=float, default=1.0)
    ws2_edge_hex_grid.add_argument('--scale-bar-linewidth', type=float, default=4.0)
    ws2_edge_hex_grid.set_defaults(func=make_figure_3_ws2_hexnet_pixel_cutoff_grid)

    figure3c_fov_sweep = subparsers.add_parser(
        'figure3c-fov-sweep',
        help='Compare centered fields of view at one physical inference pixel size.',
    )
    _add_shared_arguments(figure3c_fov_sweep)
    _add_model_arguments(figure3c_fov_sweep)
    figure3c_fov_sweep.add_argument(
        '--blobnet-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth',
    )
    figure3c_fov_sweep.add_argument(
        '--hexagonal-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/hexagonal/unet_best.pth',
    )
    figure3c_fov_sweep.add_argument('--image', type=Path, required=True)
    figure3c_fov_sweep.add_argument('--crop-sizes', type=int, nargs='+', default=[512, 1024, 2048])
    figure3c_fov_sweep.add_argument('--tile-size', type=int, default=256)
    figure3c_fov_sweep.add_argument('--tile-overlap', type=int, default=64)
    figure3c_fov_sweep.add_argument('--batch-size', type=int, default=4)
    figure3c_fov_sweep.add_argument('--dog-small', type=float, default=1.0)
    figure3c_fov_sweep.add_argument('--dog-large', type=float, default=20.0)
    figure3c_fov_sweep.add_argument('--target-pixel-size-nm', type=float, default=0.013596564437455348)
    figure3c_fov_sweep.add_argument('--localization-threshold-rel', type=float, default=0.35)
    figure3c_fov_sweep.add_argument('--peak-min-distance', type=int, default=3)
    figure3c_fov_sweep.add_argument('--peak-window-size', type=int, default=5)
    figure3c_fov_sweep.add_argument('--marker-size', type=float, default=40.0)
    figure3c_fov_sweep.add_argument('--marker-linewidth', type=float, default=0.2)
    figure3c_fov_sweep.add_argument('--marker-edge-color', default='white')
    figure3c_fov_sweep.add_argument('--blobnet-marker-color', default='#D55E00')
    figure3c_fov_sweep.add_argument('--hexagonal-marker-color', default=MODEL_COLORS['hexagonal'])
    figure3c_fov_sweep.add_argument('--scale-bar-length-nm', type=float, default=1.0)
    figure3c_fov_sweep.add_argument('--scale-bar-linewidth', type=float, default=4.0)
    figure3c_fov_sweep.set_defaults(func=make_figure_3c_fov_sweep)

    figure3c_region_atlas = subparsers.add_parser(
        'figure3c-region-atlas',
        help='Tile a full Velox frame into regional localization atlases for both models.',
    )
    _add_shared_arguments(figure3c_region_atlas)
    _add_model_arguments(figure3c_region_atlas)
    figure3c_region_atlas.add_argument(
        '--blobnet-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth',
    )
    figure3c_region_atlas.add_argument(
        '--hexagonal-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/hexagonal/unet_best.pth',
    )
    figure3c_region_atlas.add_argument('--image', type=Path, required=True)
    figure3c_region_atlas.add_argument('--grid-size', type=int, default=4)
    figure3c_region_atlas.add_argument('--region-size', type=int, default=512)
    figure3c_region_atlas.add_argument('--tile-size', type=int, default=256)
    figure3c_region_atlas.add_argument('--tile-overlap', type=int, default=64)
    figure3c_region_atlas.add_argument('--batch-size', type=int, default=4)
    figure3c_region_atlas.add_argument('--dog-small', type=float, default=1.0)
    figure3c_region_atlas.add_argument('--dog-large', type=float, default=20.0)
    figure3c_region_atlas.add_argument('--target-pixel-size-nm', type=float, default=0.013596564437455348)
    figure3c_region_atlas.add_argument('--localization-threshold-rel', type=float, default=0.35)
    figure3c_region_atlas.add_argument('--peak-min-distance', type=int, default=3)
    figure3c_region_atlas.add_argument('--peak-window-size', type=int, default=5)
    figure3c_region_atlas.add_argument('--marker-size', type=float, default=40.0)
    figure3c_region_atlas.add_argument('--marker-linewidth', type=float, default=0.2)
    figure3c_region_atlas.add_argument('--marker-edge-color', default='white')
    figure3c_region_atlas.add_argument('--blobnet-marker-color', default='#D55E00')
    figure3c_region_atlas.add_argument('--hexagonal-marker-color', default=MODEL_COLORS['hexagonal'])
    figure3c_region_atlas.add_argument('--comparison-hexagonal-marker-color', default='#56B4E9')
    figure3c_region_atlas.add_argument('--scale-bar-length-nm', type=float, default=1.0)
    figure3c_region_atlas.add_argument('--scale-bar-linewidth', type=float, default=4.0)
    figure3c_region_atlas.set_defaults(func=make_figure_3c_region_atlas)

    figure3c_sqrt_input = subparsers.add_parser(
        'figure3c-sqrt-input',
        help='Compare standard and square-root-transformed inputs for Blob-Net on one Velox HAADF image.',
    )
    _add_shared_arguments(figure3c_sqrt_input)
    _add_model_arguments(figure3c_sqrt_input)
    figure3c_sqrt_input.add_argument(
        '--blobnet-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth',
    )
    figure3c_sqrt_input.add_argument('--image', type=Path, required=True)
    figure3c_sqrt_input.add_argument('--tile-size', type=int, default=256)
    figure3c_sqrt_input.add_argument('--tile-overlap', type=int, default=64)
    figure3c_sqrt_input.add_argument('--batch-size', type=int, default=4)
    figure3c_sqrt_input.add_argument('--dog-small', type=float, default=1.0)
    figure3c_sqrt_input.add_argument('--dog-large', type=float, default=20.0)
    figure3c_sqrt_input.add_argument('--experimental-crop-size', type=int, default=512)
    figure3c_sqrt_input.add_argument('--target-pixel-size-nm', type=float, default=0.013596564437455348)
    figure3c_sqrt_input.add_argument('--localization-threshold-rel', type=float, default=0.35)
    figure3c_sqrt_input.add_argument('--peak-min-distance', type=int, default=3)
    figure3c_sqrt_input.add_argument('--peak-window-size', type=int, default=5)
    figure3c_sqrt_input.add_argument('--marker-size', type=float, default=40.0)
    figure3c_sqrt_input.add_argument('--marker-linewidth', type=float, default=0.2)
    figure3c_sqrt_input.add_argument('--marker-edge-color', default='white')
    figure3c_sqrt_input.add_argument('--blobnet-marker-color', default='#D55E00')
    figure3c_sqrt_input.add_argument('--scale-bar-length-nm', type=float, default=1.0)
    figure3c_sqrt_input.add_argument('--scale-bar-linewidth', type=float, default=4.0)
    figure3c_sqrt_input.set_defaults(func=make_figure_3c_sqrt_input)

    figure3c_low_cutoffs = subparsers.add_parser(
        'figure3c-blobnet-cutoff-gallery',
        help='Show a compact low-cutoff gallery for Blob-Net on one Velox HAADF image.',
    )
    _add_shared_arguments(figure3c_low_cutoffs)
    _add_model_arguments(figure3c_low_cutoffs)
    figure3c_low_cutoffs.add_argument(
        '--blobnet-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth',
    )
    figure3c_low_cutoffs.add_argument('--image', type=Path, required=True)
    figure3c_low_cutoffs.add_argument('--tile-size', type=int, default=256)
    figure3c_low_cutoffs.add_argument('--tile-overlap', type=int, default=64)
    figure3c_low_cutoffs.add_argument('--batch-size', type=int, default=4)
    figure3c_low_cutoffs.add_argument('--dog-small', type=float, default=1.0)
    figure3c_low_cutoffs.add_argument('--dog-large', type=float, default=20.0)
    figure3c_low_cutoffs.add_argument('--experimental-crop-size', type=int, default=512)
    figure3c_low_cutoffs.add_argument('--target-pixel-size-nm', type=float, default=0.013596564437455348)
    figure3c_low_cutoffs.add_argument('--cutoff-min', type=float, default=0.01)
    figure3c_low_cutoffs.add_argument('--cutoff-max', type=float, default=0.15)
    figure3c_low_cutoffs.add_argument('--steps', type=int, default=15)
    figure3c_low_cutoffs.add_argument('--peak-min-distance', type=int, default=3)
    figure3c_low_cutoffs.add_argument('--peak-window-size', type=int, default=5)
    figure3c_low_cutoffs.add_argument('--marker-size', type=float, default=40.0)
    figure3c_low_cutoffs.add_argument('--marker-linewidth', type=float, default=0.2)
    figure3c_low_cutoffs.add_argument('--marker-edge-color', default='white')
    figure3c_low_cutoffs.add_argument('--blobnet-marker-color', default='#D55E00')
    figure3c_low_cutoffs.add_argument('--scale-bar-length-nm', type=float, default=1.0)
    figure3c_low_cutoffs.add_argument('--scale-bar-linewidth', type=float, default=4.0)
    figure3c_low_cutoffs.set_defaults(func=make_figure_3c_blobnet_cutoff_gallery)

    quasicrystal_sweep = subparsers.add_parser(
        'quasicrystal-scale-sweep',
        help='Compare BlobNet predictions across interpolated quasicrystal pixel scales.',
    )
    _add_shared_arguments(quasicrystal_sweep)
    _add_model_arguments(quasicrystal_sweep)
    quasicrystal_sweep.add_argument(
        '--checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth',
    )
    quasicrystal_sweep.add_argument('--image', type=Path, default=repo_root / 'experimental_data/Al72Ni11Co17_quasicrystal.h5')
    quasicrystal_sweep.add_argument('--tile-size', type=int, default=256)
    quasicrystal_sweep.add_argument('--tile-overlap', type=int, default=64)
    quasicrystal_sweep.add_argument('--batch-size', type=int, default=4)
    quasicrystal_sweep.add_argument('--dog-small', type=float, default=1.0)
    quasicrystal_sweep.add_argument('--dog-large', type=float, default=20.0)
    quasicrystal_sweep.add_argument('--crop-size', type=int, default=512)
    quasicrystal_sweep.add_argument('--scale-min', type=float, default=0.5)
    quasicrystal_sweep.add_argument('--scale-max', type=float, default=1.5)
    quasicrystal_sweep.add_argument('--steps', type=int, default=20)
    quasicrystal_sweep.add_argument('--threshold-rel', type=float, default=0.35)
    quasicrystal_sweep.add_argument('--peak-min-distance', type=int, default=3)
    quasicrystal_sweep.add_argument('--peak-window-size', type=int, default=5)
    quasicrystal_sweep.set_defaults(func=make_quasicrystal_scale_sweep)

    common_pixel_size = subparsers.add_parser(
        'quasicrystal-common-pixel-size',
        help='Compare quasicrystal predictions at one common physical pixel size without changing FOV.',
    )
    _add_shared_arguments(common_pixel_size)
    _add_model_arguments(common_pixel_size)
    common_pixel_size.add_argument(
        '--checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth',
    )
    common_pixel_size.add_argument('--images', type=Path, nargs=3, required=True)
    common_pixel_size.add_argument('--target-pixel-size-nm', type=float)
    common_pixel_size.add_argument('--tile-size', type=int, default=256)
    common_pixel_size.add_argument('--tile-overlap', type=int, default=64)
    common_pixel_size.add_argument('--batch-size', type=int, default=4)
    common_pixel_size.add_argument('--dog-small', type=float, default=1.0)
    common_pixel_size.add_argument('--dog-large', type=float, default=20.0)
    common_pixel_size.add_argument('--crop-size', type=int, default=512)
    common_pixel_size.set_defaults(func=make_quasicrystal_common_pixel_size)

    figure3_pixel_sweep = subparsers.add_parser(
        'figure3-pixel-size-sweep',
        help='Sweep one common physical pixel size across all three Figure 3 images without changing FOV.',
    )
    _add_shared_arguments(figure3_pixel_sweep)
    _add_model_arguments(figure3_pixel_sweep)
    figure3_pixel_sweep.add_argument(
        '--checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth',
    )
    figure3_pixel_sweep.add_argument('--images', type=Path, nargs=3, required=True)
    figure3_pixel_sweep.add_argument('--base-pixel-size-nm', type=float, required=True)
    figure3_pixel_sweep.add_argument('--factor-min', type=float, default=0.7)
    figure3_pixel_sweep.add_argument('--factor-max', type=float, default=1.3)
    figure3_pixel_sweep.add_argument('--steps', type=int, default=20)
    figure3_pixel_sweep.add_argument('--tile-size', type=int, default=256)
    figure3_pixel_sweep.add_argument('--tile-overlap', type=int, default=64)
    figure3_pixel_sweep.add_argument('--batch-size', type=int, default=4)
    figure3_pixel_sweep.add_argument('--dog-small', type=float, default=1.0)
    figure3_pixel_sweep.add_argument('--dog-large', type=float, default=20.0)
    figure3_pixel_sweep.add_argument('--crop-size', type=int, default=512)
    figure3_pixel_sweep.set_defaults(func=make_figure3_pixel_size_sweep)

    figure3_localizations = subparsers.add_parser(
        'figure3-localizations',
        help='Experimental HAADF localization comparison with BlobNet and LoG under Poisson noise.',
    )
    _add_shared_arguments(figure3_localizations)
    _add_model_arguments(figure3_localizations)
    figure3_localizations.add_argument('--checkpoint', type=Path, default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth')
    figure3_localizations.add_argument('--data-dir', type=Path, default=repo_root / 'experimental_data')
    figure3_localizations.add_argument('--tile-size', type=int, default=256)
    figure3_localizations.add_argument('--tile-overlap', type=int, default=64)
    figure3_localizations.add_argument('--batch-size', type=int, default=4)
    figure3_localizations.add_argument('--dog-small', type=float, default=1.0)
    figure3_localizations.add_argument('--dog-large', type=float, default=20.0)
    figure3_localizations.add_argument('--experimental-measurements', type=Path, default=repo_root / 'outputs/experimental_feature_measurements_local/experimental_feature_measurements.json')
    figure3_localizations.add_argument('--feature-match-sigma-px', type=float, default=2.9)
    figure3_localizations.add_argument('--experimental-crop-size', type=int, default=512)
    figure3_localizations.add_argument('--seed', type=int, default=7)
    figure3_localizations.add_argument('--poisson-counts', type=float, default=80.0)
    figure3_localizations.add_argument('--heavy-poisson-counts', type=float, default=28.0)
    figure3_localizations.add_argument('--localization-threshold-rel', type=float, default=0.35)
    figure3_localizations.add_argument('--peak-min-distance', type=int, default=3)
    figure3_localizations.add_argument('--peak-window-size', type=int, default=5)
    figure3_localizations.add_argument('--max-peaks', type=int)
    figure3_localizations.add_argument('--log-sigma-px', type=float, default=2.9)
    figure3_localizations.add_argument('--log-threshold-rel', type=float, default=0.35)
    figure3_localizations.add_argument('--log-min-distance', type=int, default=5)
    figure3_localizations.set_defaults(func=make_figure_3_localizations)

    figure4 = subparsers.add_parser('figure4', help='Scale and spacing robustness figure.')
    _add_shared_arguments(figure4)
    _add_model_arguments(figure4)
    _add_feature_size_arguments(figure4)
    figure4.add_argument('--checkpoint', type=Path, default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth')
    figure4.add_argument('--sweep-csv', type=Path, default=repo_root / 'outputs/blobnet_pixel_size_sweep_random_4x/pixel_size_metrics.csv')
    figure4.add_argument('--sweep-dataset-config', type=Path)
    figure4.add_argument('--sweep-samples', type=int, default=64)
    figure4.add_argument('--sweep-pixel-size-factors', type=_parse_float_list)
    figure4.add_argument('--sweep-threshold-grid', type=_parse_float_list)
    figure4.add_argument('--regenerate-sweep', action='store_true')
    figure4.add_argument('--batch-size', type=int, default=8)
    figure4.set_defaults(func=make_figure_4)

    ws2_edge_comparison = subparsers.add_parser('ws2-edge-comparison', help='Simulated WS2 monolayer flake edge model comparison.')
    _add_shared_arguments(ws2_edge_comparison)
    _add_model_arguments(ws2_edge_comparison)
    _add_feature_size_arguments(ws2_edge_comparison)
    ws2_edge_comparison.add_argument('--square-checkpoint', type=Path, default=repo_root / 'artifacts/manuscript_models/square/unet_best.pth')
    ws2_edge_comparison.add_argument('--hexagonal-checkpoint', type=Path, default=repo_root / 'artifacts/manuscript_models/hexagonal/unet_best.pth')
    ws2_edge_comparison.add_argument('--random-checkpoint', type=Path, default=repo_root / 'artifacts/manuscript_models/random_dense/unet_best.pth')
    ws2_edge_comparison.add_argument('--seed', type=int, default=41)
    _add_edge_localization_arguments(ws2_edge_comparison)
    ws2_edge_comparison.set_defaults(func=make_ws2_edge_comparison)

    figure2 = subparsers.add_parser('figure2', help='Edge-structure TP/FP/FN diagnostics for WS2, SrTiO3, and graphene.')
    _add_shared_arguments(figure2)
    _add_model_arguments(figure2)
    _add_feature_size_arguments(figure2)
    figure2.add_argument('--square-checkpoint', type=Path, default=repo_root / 'artifacts/manuscript_models/square/unet_best.pth')
    figure2.add_argument('--hexagonal-checkpoint', type=Path, default=repo_root / 'artifacts/manuscript_models/hexagonal/unet_best.pth')
    figure2.add_argument('--random-checkpoint', type=Path, default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth')
    figure2.add_argument('--seed', type=int, default=41)
    figure2.add_argument('--with-offset-diagnostics', action='store_true')
    figure2.add_argument('--offset-range', type=float, default=2.0)
    figure2.add_argument('--offset-bins', type=int, default=48)
    _add_edge_localization_arguments(figure2)
    figure2.set_defaults(func=make_figure_2, feature_sigma_min=1.15, feature_sigma_max=2.65)

    all_parser = subparsers.add_parser('all', help='Build all manuscript figures.')
    _add_shared_arguments(all_parser)
    _add_model_arguments(all_parser)
    _add_dataset_config_arguments(all_parser)
    _add_feature_size_arguments(all_parser)
    all_parser.add_argument('--square-checkpoint', type=Path, default=repo_root / 'artifacts/manuscript_models/square/unet_best.pth')
    all_parser.add_argument('--hexagonal-checkpoint', type=Path, default=repo_root / 'artifacts/manuscript_models/hexagonal/unet_best.pth')
    all_parser.add_argument('--random-checkpoint', type=Path, default=repo_root / 'artifacts/manuscript_models/random/unet_best.pth')
    all_parser.add_argument('--ws2-random-checkpoint', type=Path, default=repo_root / 'artifacts/manuscript_models/random_dense/unet_best.pth')
    all_parser.add_argument(
        '--checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/figure3_random/unet_best.pth',
    )
    all_parser.add_argument(
        '--figure3-hexagonal-checkpoint',
        type=Path,
        default=repo_root / 'artifacts/manuscript_models/figure3_hexagonal/unet_best.pth',
    )
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
    all_parser.add_argument(
        '--quasicrystal-image',
        type=Path,
        default=repo_root / 'experimental_data/Al72Ni11Co17_quasicrystal.h5',
    )
    all_parser.add_argument('--target-pixel-size-nm', type=float, default=0.027193128874910695)
    all_parser.add_argument('--ws2-hexagonal-pixel-size-factor', type=float, default=0.70)
    all_parser.add_argument('--ws2-hexagonal-threshold-rel', type=float, default=0.30)
    all_parser.add_argument('--marker-size', type=float, default=40.0)
    all_parser.add_argument('--marker-linewidth', type=float, default=0.2)
    all_parser.add_argument('--marker-color', default='#D55E00')
    all_parser.add_argument('--marker-edge-color', default='white')
    all_parser.add_argument('--disagreement-marker', choices=['x', '+', 's'], default='+')
    all_parser.add_argument('--disagreement-marker-size-scale', type=float, default=1.2)
    all_parser.add_argument('--scale-bar-length-nm', type=float, default=1.0)
    all_parser.add_argument('--scale-bar-linewidth', type=float, default=4.0)
    _add_edge_localization_arguments(all_parser)
    all_parser.set_defaults(func=None, localization_threshold_rel=0.35)
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
            make_figure_4(args),
        ]
        args.random_checkpoint = args.ws2_random_checkpoint
        paths.append(make_ws2_edge_comparison(args))
        args.random_checkpoint = original_random_checkpoint
        args.random_checkpoint = original_random_checkpoint
    else:
        paths = [args.func(args)]
    for path in paths:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
