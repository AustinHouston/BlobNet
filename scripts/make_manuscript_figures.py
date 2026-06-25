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
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

_CACHE_DIR = Path(tempfile.gettempdir()) / 'blobnet-mpl-cache'
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_CACHE_DIR))
os.environ.setdefault('XDG_CACHE_HOME', str(_CACHE_DIR))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from blobnet.networks import build_unet
from blobnet.synthetic import (
    GeneratedAtomImageDataset,
    ImageFormationConfig,
    PeriodicLatticeConfig,
    RandomAtomImageConfig,
    generate_atom_image,
    generate_atoms_image,
    metadata_collate,
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


def _plot_clean_image(ax: plt.Axes, image: np.ndarray, title: str, cmap: str = 'gray') -> None:
    ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def _make_dataset_specs(repo_root: Path) -> list[DatasetSpec]:
    configs = {
        'square': repo_root / 'configs/dataset_configs/square.yaml',
        'hexagonal': repo_root / 'configs/dataset_configs/hexagonal.yaml',
        'random': repo_root / 'configs/dataset_configs/random.yaml',
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
        fontsize=7.5,
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.78, 'boxstyle': 'round,pad=0.18'},
    )


def make_figure_1(args: argparse.Namespace) -> Path:
    repo_root = _repo_root()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    datasets = _make_dataset_specs(repo_root)
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
        7,
        left=0.035,
        right=0.99,
        top=0.875,
        bottom=0.075,
        wspace=0.18,
        hspace=0.28,
    )
    fig.suptitle('Training Geometry Controls Synthetic Generalization', fontsize=16)

    for row, dataset in enumerate(datasets):
        ax = fig.add_subplot(grid[row, 0])
        _plot_clean_image(ax, examples[dataset.key]['image'], f'{dataset.label} training example')

    for row, dataset in enumerate(datasets):
        for col, model_spec in enumerate(models):
            ax = fig.add_subplot(grid[row, col + 1])
            if row == 0:
                ax.set_title(model_spec.label, fontsize=11)
            prediction = predictions.get((model_spec.key, dataset.key))
            if prediction is None:
                _axis_missing(ax, 'Checkpoint missing')
            else:
                cmap = MODEL_CMAPS.get(model_spec.key, 'viridis')
                ax.imshow(prediction, cmap=cmap, vmin=0.0, vmax=max(float(prediction.max()), 1e-6))
                if col == 0:
                    ax.set_ylabel(f'{dataset.label} test', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                _annotate_probability_metric(ax, offset_results.get((model_spec.key, dataset.key)))

        for col, model_spec in enumerate(models):
            ax = fig.add_subplot(grid[row, col + 4])
            if row == 0:
                ax.set_title(f'{model_spec.label} offsets', fontsize=11)
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
            ax.tick_params(labelsize=7)
            ax.text(
                0.04,
                0.96,
                f"F1={float(result['f1']):.3f}\nRMSE={float(result['rmse']):.2f}px",
                transform=ax.transAxes,
                ha='left',
                va='top',
                color='white',
                fontsize=7,
            )

    fig.text(0.105, 0.915, 'Representative synthetic training images', ha='center', fontsize=11)
    fig.text(0.455, 0.915, 'Raw model probability maps', ha='center', fontsize=11)
    fig.text(0.795, 0.915, 'Matched localization offset clouds', ha='center', fontsize=11)
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

    files = [
        ('WS2 grain boundary', args.data_dir / 'WS2.emd'),
        ('Quasicrystal', args.data_dir / 'QuasiCrystal.emd'),
        ('Twin boundary', args.data_dir / 'TwinBoundary.emd'),
        ('Twins overview', args.data_dir / 'TwinsOverview.emd'),
    ]
    images: list[tuple[str, np.ndarray]] = []
    outputs: list[np.ndarray] = []
    for label, path in files:
        image = _load_experimental_image(path)
        model_input = _normalize_image(gaussian_filter(image, args.dog_small) - gaussian_filter(image, args.dog_large))
        prediction = _predict_tiled(model, model_input, device, args.tile_size, args.tile_overlap, args.batch_size)
        images.append((label, image))
        outputs.append(prediction)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), constrained_layout=True)
    fig.suptitle('BlobNet Responses on Experimental HAADF-STEM Images', fontsize=16)
    for col, ((label, image), output) in enumerate(zip(images, outputs)):
        _plot_clean_image(axes[0, col], image, label, cmap='gray')
        axes[1, col].imshow(output, cmap=MODEL_CMAPS['random'], vmin=0.0, vmax=max(float(output.max()), 1e-6))
        axes[1, col].set_title('BlobNet output', fontsize=10)
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
            'output_mean': float(output.mean()),
            'output_max': float(output.max()),
        }
        for (label, image), output in zip(images, outputs)
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


def _spacing_variation_image() -> np.ndarray:
    from ase import Atoms

    shape = (360, 360)
    positions = []
    numbers = []
    for y, spacing in zip(np.linspace(32, 328, 8), [7.5, 10.5, 14.5, 20.5, 30.0, 45.0, 68.0, 104.0]):
        atom_count = max(3, int(round((shape[1] - 56) / spacing)) + 1)
        xs = np.linspace(28, shape[1] - 28, atom_count, dtype=np.float32)
        for x in xs:
            positions.append((float(x), float(y), 0.0))
            numbers.append(74)
    atoms = Atoms(numbers=numbers, positions=positions, cell=[shape[1], shape[0], 30.0], pbc=False)
    return _render_atoms_stem_panel(atoms, shape, seed=33, atom_sigma_range=(2.7, 2.7))


def _size_variation_image() -> np.ndarray:
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
    return _render_atoms_stem_panel(atoms, shape, seed=57, atom_sigma_range=(1.15, 6.15))


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
        pixel_size_factors=DEFAULT_PIXEL_SIZE_FACTORS,
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
        threshold_grid=DEFAULT_THRESHOLD_GRID,
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
    spacing_px = np.asarray([row['spacing_mean_px'] for row in rows], dtype=np.float32)
    train_pixel_size = _training_pixel_size(rows)

    f1_color = '#2f7f73'
    rmse_color = '#b9653e'
    feature_color = '#426aa8'
    spacing_color = '#8a58a2'

    f1_line = top_ax.plot(pixel_size, f1, color=f1_color, marker='o', linewidth=2.2, markersize=5.5, label='Localization F1')[0]
    for x, value in zip(pixel_size, f1):
        top_ax.text(float(x), float(value) + 0.018, f'{value:.3f}', ha='center', va='bottom', fontsize=8.5, color='#222222')
    train_line = top_ax.axvline(train_pixel_size, color='black', linestyle='--', linewidth=1.35, label='value used in training')
    top_ax.set_ylabel('Localization F1 (higher is better)', fontsize=11)
    top_ax.set_ylim(0.0, 1.02)
    top_ax.grid(alpha=0.25)

    rmse_ax = top_ax.twinx()
    rmse_line = rmse_ax.plot(pixel_size, rmse, color=rmse_color, marker='s', linewidth=2.0, markersize=5.0, label='Localization RMSE')[0]
    rmse_ax.set_ylabel('Localization RMSE (px, lower is better)', fontsize=11, color='#8f4d31')
    rmse_ax.tick_params(axis='y', colors='#8f4d31')
    rmse_ax.set_ylim(0.0, max(float(rmse.max()) * 1.05, 0.5))
    top_ax.legend(handles=[f1_line, rmse_line, train_line], loc='lower left', fontsize=8.5, frameon=True)

    bottom_ax.plot(pixel_size, feature_ratio, color=feature_color, marker='o', linewidth=2.2, markersize=5.5, label='mean blob width / receptive field')
    bottom_ax.plot(pixel_size, spacing_ratio, color=spacing_color, marker='s', linewidth=2.2, markersize=5.5, label='mean atom spacing / receptive field')
    bottom_ax.axvline(train_pixel_size, color='black', linestyle='--', linewidth=1.35, label='value used in training')
    for x, y, spacing in zip(pixel_size, spacing_ratio, spacing_px):
        bottom_ax.text(float(x), float(y) + 0.018, f'{spacing:.1f}px spacing', rotation=18, ha='center', va='bottom', fontsize=7.5, color='#222222')
    bottom_ax.set_xlabel('Assumed pixel size (angstrom / px)', fontsize=11)
    bottom_ax.set_ylabel('Ratio to U-Net bottleneck receptive field (68 px)', fontsize=11)
    bottom_ax.set_ylim(0.0, max(0.9, float(spacing_ratio.max()) * 1.05))
    bottom_ax.grid(alpha=0.25)
    bottom_ax.legend(loc='upper right', fontsize=8.5, frameon=True)


def make_figure_3(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    model = _load_blobnet_model(args.checkpoint, device, args.num_filters, args.dropout)
    _generate_missing_sweep_csv(args)
    rows = _read_sweep_rows(args.sweep_csv)
    spacing_image = _spacing_variation_image()
    size_image = _size_variation_image()
    spacing_prediction = _predict_array(model, spacing_image, device)
    size_prediction = _predict_array(model, size_image, device)

    fig = plt.figure(figsize=(17.5, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 0.08, 2.55], height_ratios=[1, 1])
    fig.suptitle('Random Blob Placement Supports Scale and Spacing Robustness', fontsize=16)

    ax_spacing = fig.add_subplot(grid[0, 0])
    _plot_clean_image(ax_spacing, spacing_image, 'Variable inter-feature spacing')
    ax_size = fig.add_subplot(grid[0, 1])
    _plot_clean_image(ax_size, size_image, 'Variable feature size')

    ax_spacing_prediction = fig.add_subplot(grid[1, 0])
    ax_spacing_prediction.imshow(spacing_prediction, cmap=MODEL_CMAPS['random'], vmin=0.0, vmax=max(float(spacing_prediction.max()), 1e-6))
    ax_spacing_prediction.set_title('Random model output', fontsize=10)
    ax_spacing_prediction.set_xticks([])
    ax_spacing_prediction.set_yticks([])

    ax_size_prediction = fig.add_subplot(grid[1, 1])
    ax_size_prediction.imshow(size_prediction, cmap=MODEL_CMAPS['random'], vmin=0.0, vmax=max(float(size_prediction.max()), 1e-6))
    ax_size_prediction.set_title('Random model output', fontsize=10)
    ax_size_prediction.set_xticks([])
    ax_size_prediction.set_yticks([])

    ax_top = fig.add_subplot(grid[0, 3])
    ax_bottom = fig.add_subplot(grid[1, 3], sharex=ax_top)
    ax_top.set_title('BlobNet Pixel-Size Sweep (random)', fontsize=15, fontweight='bold')
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


def build_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description='Build manuscript figures for BlobNet real-STEM utility.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    figure1 = subparsers.add_parser('figure1', help='Synthetic geometry comparison figure.')
    _add_shared_arguments(figure1)
    _add_model_arguments(figure1)
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
    figure2.set_defaults(func=make_figure_2)

    figure3 = subparsers.add_parser('figure3', help='Scale and spacing robustness figure.')
    _add_shared_arguments(figure3)
    _add_model_arguments(figure3)
    figure3.add_argument('--checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/random/unet_best.pth')
    figure3.add_argument('--sweep-csv', type=Path, default=repo_root / 'outputs/blobnet_pixel_size_sweep_random_4x/pixel_size_metrics.csv')
    figure3.add_argument('--sweep-dataset-config', type=Path)
    figure3.add_argument('--sweep-samples', type=int, default=64)
    figure3.add_argument('--regenerate-sweep', action='store_true')
    figure3.add_argument('--batch-size', type=int, default=8)
    figure3.set_defaults(func=make_figure_3)

    all_parser = subparsers.add_parser('all', help='Build all three figures.')
    _add_shared_arguments(all_parser)
    _add_model_arguments(all_parser)
    all_parser.add_argument('--square-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/square/unet_best.pth')
    all_parser.add_argument('--hexagonal-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/hexagonal/unet_best.pth')
    all_parser.add_argument('--random-checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/random/unet_best.pth')
    all_parser.add_argument('--checkpoint', type=Path, default=repo_root / 'outputs/manuscript_models/random/unet_best.pth')
    all_parser.add_argument('--data-dir', type=Path, default=repo_root / 'experimental_data')
    all_parser.add_argument('--sweep-csv', type=Path, default=repo_root / 'outputs/blobnet_pixel_size_sweep_random_4x/pixel_size_metrics.csv')
    all_parser.add_argument('--sweep-dataset-config', type=Path)
    all_parser.add_argument('--sweep-samples', type=int, default=64)
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
    all_parser.set_defaults(func=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == 'all':
        paths = [
            make_figure_1(args),
            make_figure_2(args),
            make_figure_3(args),
        ]
    else:
        paths = [args.func(args)]
    for path in paths:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
