from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_CACHE_DIR = Path(tempfile.gettempdir()) / 'blobnet-mpl-cache'
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_CACHE_DIR))
os.environ.setdefault('XDG_CACHE_HOME', str(_CACHE_DIR))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

from blobnet.metrics import aggregate_localization_metrics, evaluate_heatmap_localization
from blobnet.networks import build_unet
from blobnet.synthetic import GeneratedAtomImageDataset, RandomAtomImageConfig, metadata_collate


BOTTLENECK_RECEPTIVE_FIELD_PX = 68
DEFAULT_TRAIN_PIXEL_SIZE_ANGSTROM = 0.1062231596676199
DEFAULT_PIXEL_SIZE_FACTORS = [0.25, 0.33, 0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
DEFAULT_THRESHOLD_GRID = [0.1, 0.15, 0.25, 0.35, 0.5]
CSV_FIELDS = [
    'pixel_size_angstrom',
    'pixel_size_factor',
    'dataset',
    'threshold_rel',
    'precision',
    'recall',
    'f1',
    'mean_error_px',
    'median_error_px',
    'rmse_px',
    'tp',
    'fp',
    'fn',
    'samples',
    'sigma_min_px',
    'sigma_max_px',
    'sigma_mean_px',
    'feature_fwhm_mean_px',
    'spacing_min_px',
    'spacing_max_px',
    'spacing_mean_px',
    'target_sigma_px',
    'match_distance_px',
    'peak_min_distance_px',
    'peak_window_size_px',
    'bottleneck_receptive_field_px',
    'bottleneck_receptive_field_angstrom',
    'feature_fwhm_over_bottleneck_rf',
    'spacing_over_bottleneck_rf',
]


def _device_from_name(name: str) -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _load_model(checkpoint: Path, device: torch.device, num_filters: list[int], dropout: float) -> torch.nn.Module:
    if not checkpoint.is_file():
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


def _odd_pixel_count(value: float) -> int:
    count = max(1, int(round(float(value))))
    return count if count % 2 else count + 1


def _scaled_px(train_value_px: float, pixel_size_factor: float, fixed_evaluation_pixels: bool) -> float:
    if fixed_evaluation_pixels:
        return float(train_value_px)
    return float(train_value_px) / float(pixel_size_factor)


def _read_random_dataset_parameters(path: Path) -> tuple[dict[str, Any], str]:
    raw = yaml.safe_load(path.read_text())
    dataset_type = raw.get('type') or raw.get('dataset', {}).get('type')
    if dataset_type != 'random':
        raise ValueError(f'Pixel-size sweep expects a random dataset config, got {dataset_type!r} in {path}')
    parameters = raw.get('parameters')
    if not isinstance(parameters, dict):
        raise ValueError(f'Missing parameters in {path}')
    return dict(parameters), str(path)


def _resolve_training_parameters(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.dataset_config is not None:
        return _read_random_dataset_parameters(args.dataset_config)

    resolved_config = args.checkpoint.parent / 'resolved_config.yaml'
    if resolved_config.is_file():
        resolved = yaml.safe_load(resolved_config.read_text())
        dataset_path = Path(resolved['dataset']['path'])
        manifest_path = dataset_path / 'dataset_manifest.yaml'
        if manifest_path.is_file():
            return _read_random_dataset_parameters(manifest_path)

    return _read_random_dataset_parameters(Path('configs/dataset_configs/random.yaml'))


def _range_from_parameters(parameters: dict[str, Any], name: str) -> tuple[float, float]:
    value = parameters.get(name)
    if value is None:
        raise ValueError(f'Training dataset parameters must define {name}.')
    return float(value[0]), float(value[1])


def _optional_range_override(args: argparse.Namespace, low_name: str, high_name: str, fallback: tuple[float, float]) -> tuple[float, float]:
    low = getattr(args, low_name)
    high = getattr(args, high_name)
    return (
        float(fallback[0]) if low is None else float(low),
        float(fallback[1]) if high is None else float(high),
    )


def _apply_training_defaults(args: argparse.Namespace) -> None:
    parameters, source = _resolve_training_parameters(args)
    args.training_parameters = parameters
    args.training_parameter_source = source

    image_shape = parameters.get('image_shape', [256, 256])
    args.height = int(image_shape[0]) if args.height is None else int(args.height)
    args.width = int(image_shape[1]) if args.width is None else int(args.width)
    args.min_atoms = int(parameters.get('min_atoms', 80)) if args.min_atoms is None else int(args.min_atoms)
    args.max_atoms = int(parameters.get('max_atoms', args.min_atoms)) if args.max_atoms is None else int(args.max_atoms)

    sigma_min, sigma_max = _range_from_parameters(parameters, 'sigma_range')
    args.train_sigma_min = sigma_min if args.train_sigma_min is None else float(args.train_sigma_min)
    args.train_sigma_max = sigma_max if args.train_sigma_max is None else float(args.train_sigma_max)
    args.train_target_sigma = float(parameters.get('target_sigma', 0.9)) if args.train_target_sigma is None else float(args.train_target_sigma)

    spacing_range = parameters.get('min_separation_range')
    if spacing_range is None:
        spacing = float(parameters.get('min_separation', 12.0))
        spacing_range = (spacing, spacing)
    args.train_min_separation_range_min = (
        float(spacing_range[0])
        if args.train_min_separation_range_min is None
        else float(args.train_min_separation_range_min)
    )
    args.train_min_separation_range_max = (
        float(spacing_range[1])
        if args.train_min_separation_range_max is None
        else float(args.train_min_separation_range_max)
    )
    args.train_match_distance = 1.5 * args.train_target_sigma if args.train_match_distance is None else float(args.train_match_distance)
    args.train_peak_min_distance = max(1, int(round(args.train_target_sigma * 1.5))) if args.train_peak_min_distance is None else int(args.train_peak_min_distance)
    args.train_peak_window_size = _odd_pixel_count(args.train_target_sigma * 2.5) if args.train_peak_window_size is None else int(args.train_peak_window_size)


def _range_override(args: argparse.Namespace, parameters: dict[str, Any], field: str, low_name: str, high_name: str) -> tuple[float, float]:
    return _optional_range_override(args, low_name, high_name, _range_from_parameters(parameters, field))


def _build_random_config(args: argparse.Namespace, pixel_size_factor: float) -> tuple[RandomAtomImageConfig, dict[str, float]]:
    parameters = dict(args.training_parameters)
    sigma_min = _scaled_px(args.train_sigma_min, pixel_size_factor, args.fixed_evaluation_pixels)
    sigma_max = _scaled_px(args.train_sigma_max, pixel_size_factor, args.fixed_evaluation_pixels)
    target_sigma = _scaled_px(args.train_target_sigma, pixel_size_factor, args.fixed_evaluation_pixels)
    spacing_min = _scaled_px(args.train_min_separation_range_min, pixel_size_factor, args.fixed_evaluation_pixels)
    spacing_max = _scaled_px(args.train_min_separation_range_max, pixel_size_factor, args.fixed_evaluation_pixels)
    parameters.update(
        {
            'image_shape': (args.height, args.width),
            'min_atoms': args.min_atoms,
            'max_atoms': args.max_atoms,
            'min_separation': 0.5 * (spacing_min + spacing_max),
            'min_separation_range': (spacing_min, spacing_max),
            'sigma_range': (sigma_min, sigma_max),
            'target_sigma': target_sigma,
            'background_range': _range_override(args, parameters, 'background_range', 'background_min', 'background_max'),
            'inhomogeneous_background_range': _range_override(args, parameters, 'inhomogeneous_background_range', 'inhom_background_min', 'inhom_background_max'),
            'low_frequency_noise_range': _range_override(args, parameters, 'low_frequency_noise_range', 'low_freq_noise_min', 'low_freq_noise_max'),
            'read_noise_std_range': _range_override(args, parameters, 'read_noise_std_range', 'read_noise_min', 'read_noise_max'),
            'total_counts_range': _range_override(args, parameters, 'total_counts_range', 'total_counts_min', 'total_counts_max'),
            'blur_sigma_range': _range_override(args, parameters, 'blur_sigma_range', 'blur_sigma_min', 'blur_sigma_max'),
            'edge_padding': int(parameters.get('edge_padding', 0)) if args.edge_padding is None else int(args.edge_padding),
        }
    )
    config_fields = RandomAtomImageConfig.__dataclass_fields__
    config = RandomAtomImageConfig(**{name: parameters[name] for name in config_fields if name in parameters})
    return config, {
        'sigma_min_px': sigma_min,
        'sigma_max_px': sigma_max,
        'sigma_mean_px': 0.5 * (sigma_min + sigma_max),
        'feature_fwhm_mean_px': 2.355 * 0.5 * (sigma_min + sigma_max),
        'spacing_min_px': spacing_min,
        'spacing_max_px': spacing_max,
        'spacing_mean_px': 0.5 * (spacing_min + spacing_max),
        'target_sigma_px': target_sigma,
    }


def _predict_sweep_samples(
    model: torch.nn.Module,
    config: RandomAtomImageConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    dataset = GeneratedAtomImageDataset(args.samples_per_size, config, seed=args.seed, return_metadata=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=metadata_collate)
    samples = []
    example = {}
    with torch.inference_mode():
        for images, targets, metadata_list in loader:
            outputs = torch.sigmoid(model(images.to(device)))[:, 0].detach().cpu().numpy()
            for image, target, output, metadata in zip(images[:, 0].numpy(), targets[:, 0].numpy(), outputs, metadata_list):
                if not example:
                    example = {'image': image, 'target': target, 'output': output}
                samples.append({'prediction': output, 'coordinates': metadata['coordinates']})
    return samples, example


def _evaluate_thresholds(
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    pixel_size_factor: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    match_distance = _scaled_px(args.train_match_distance, pixel_size_factor, args.fixed_evaluation_pixels)
    peak_min_distance = max(1, int(round(_scaled_px(args.train_peak_min_distance, pixel_size_factor, args.fixed_evaluation_pixels))))
    peak_window_size = _odd_pixel_count(_scaled_px(args.train_peak_window_size, pixel_size_factor, args.fixed_evaluation_pixels))
    best_summary = None
    best_threshold = None
    for threshold in args.threshold_grid:
        results = [
            evaluate_heatmap_localization(
                sample['prediction'],
                sample['coordinates'],
                threshold_rel=threshold,
                min_distance=peak_min_distance,
                peak_window_size=peak_window_size,
                match_distance=match_distance,
            )
            for sample in samples
        ]
        summary = aggregate_localization_metrics(results)
        if best_summary is None or float(summary['f1']) > float(best_summary['f1']):
            best_summary = summary
            best_threshold = threshold
    return best_summary, {
        'threshold_rel': float(best_threshold),
        'match_distance_px': match_distance,
        'peak_min_distance_px': peak_min_distance,
        'peak_window_size_px': peak_window_size,
    }


def _make_row(
    summary: dict[str, Any],
    pixel_size_angstrom: float,
    pixel_size_factor: float,
    geometry: dict[str, float],
    evaluation: dict[str, float],
) -> dict[str, Any]:
    row = {
        'pixel_size_angstrom': pixel_size_angstrom,
        'pixel_size_factor': pixel_size_factor,
        'dataset': 'random',
        'threshold_rel': evaluation['threshold_rel'],
        'precision': float(summary['precision']),
        'recall': float(summary['recall']),
        'f1': float(summary['f1']),
        'mean_error_px': float(summary['mean_error']),
        'median_error_px': float(summary['median_error']),
        'rmse_px': float(summary['rmse']),
        'tp': int(summary['tp']),
        'fp': int(summary['fp']),
        'fn': int(summary['fn']),
        'samples': int(summary['samples']),
        'bottleneck_receptive_field_px': BOTTLENECK_RECEPTIVE_FIELD_PX,
        'bottleneck_receptive_field_angstrom': BOTTLENECK_RECEPTIVE_FIELD_PX * pixel_size_angstrom,
    }
    row.update(geometry)
    row.update(evaluation)
    row['feature_fwhm_over_bottleneck_rf'] = row['feature_fwhm_mean_px'] / BOTTLENECK_RECEPTIVE_FIELD_PX
    row['spacing_over_bottleneck_rf'] = row['spacing_mean_px'] / BOTTLENECK_RECEPTIVE_FIELD_PX
    return row


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in CSV_FIELDS} for row in rows)


def _save_accuracy_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    sorted_rows = sorted(rows, key=lambda row: row['pixel_size_angstrom'])
    pixel_sizes = [row['pixel_size_angstrom'] for row in sorted_rows]
    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    axis.plot(pixel_sizes, [row['f1'] for row in sorted_rows], marker='o', label='Localization F1')
    axis.set(xlabel='Assumed pixel size (angstrom / px)', ylabel='Localization F1 (higher is better)', ylim=(0.0, 1.02))
    axis.grid(alpha=0.25)
    ratio_axis = axis.twinx()
    ratio_axis.plot(pixel_sizes, [row['feature_fwhm_over_bottleneck_rf'] for row in sorted_rows], marker='s', color='#426aa8', label='blob width / receptive field')
    ratio_axis.plot(pixel_sizes, [row['spacing_over_bottleneck_rf'] for row in sorted_rows], marker='^', color='#8a58a2', label='atom spacing / receptive field')
    ratio_axis.set_ylabel('Ratio to 68 px U-Net bottleneck receptive field')
    lines, labels = axis.get_legend_handles_labels()
    ratio_lines, ratio_labels = ratio_axis.get_legend_handles_labels()
    axis.legend(lines + ratio_lines, labels + ratio_labels, loc='best')
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _save_example_gallery(path: Path, rows: list[dict[str, Any]], examples: list[dict[str, np.ndarray]], example_size_count: int | None) -> None:
    if not examples:
        return
    limit = len(examples) if example_size_count is None else min(len(examples), int(example_size_count))
    if limit < len(examples):
        indices = np.linspace(0, len(examples) - 1, limit, dtype=int).tolist()
    else:
        indices = list(range(len(examples)))
    fig, axes = plt.subplots(len(indices), 3, figsize=(7.8, 2.15 * len(indices)), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(indices), 3)
    for axis_row, index in zip(axes, indices):
        row = rows[index]
        example = examples[index]
        title = f"{row['pixel_size_angstrom']:.3f} A/px"
        for axis, key, label, cmap in zip(axis_row, ['image', 'target', 'output'], ['input', 'target', 'output'], ['gray', 'magma', 'viridis']):
            axis.imshow(example[key], cmap=cmap)
            axis.set_title(f'{title} {label}', fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(',') if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate synthetic pixel-size sweep samples, test BlobNet, and write sweep metrics.')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/blobnet_pixel_size_sweep_random_4x'))
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/inhom_background_unet_20epoch/unet/unet_best.pth'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--samples-per-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--dataset-config', type=Path, help='Random dataset config to use for factor-1 sweep geometry.')
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--train-pixel-size-angstrom', type=float, default=DEFAULT_TRAIN_PIXEL_SIZE_ANGSTROM)
    parser.add_argument('--pixel-size-factors', type=_parse_float_list, default=DEFAULT_PIXEL_SIZE_FACTORS)
    parser.add_argument('--pixel-sizes-angstrom', type=_parse_float_list)
    parser.add_argument('--num-filters', type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--train-sigma-min', type=float)
    parser.add_argument('--train-sigma-max', type=float)
    parser.add_argument('--train-target-sigma', type=float)
    parser.add_argument('--train-min-separation-range-min', type=float)
    parser.add_argument('--train-min-separation-range-max', type=float)
    parser.add_argument('--min-atoms', type=int)
    parser.add_argument('--max-atoms', type=int)
    parser.add_argument('--background-min', type=float)
    parser.add_argument('--background-max', type=float)
    parser.add_argument('--inhom-background-min', type=float)
    parser.add_argument('--inhom-background-max', type=float)
    parser.add_argument('--low-freq-noise-min', type=float)
    parser.add_argument('--low-freq-noise-max', type=float)
    parser.add_argument('--read-noise-min', type=float)
    parser.add_argument('--read-noise-max', type=float)
    parser.add_argument('--total-counts-min', type=float)
    parser.add_argument('--total-counts-max', type=float)
    parser.add_argument('--blur-sigma-min', type=float)
    parser.add_argument('--blur-sigma-max', type=float)
    parser.add_argument('--edge-padding', type=int)
    parser.add_argument('--threshold-grid', type=_parse_float_list, default=DEFAULT_THRESHOLD_GRID)
    parser.add_argument('--train-match-distance', type=float)
    parser.add_argument('--train-peak-min-distance', type=int)
    parser.add_argument('--train-peak-window-size', type=int)
    parser.add_argument('--fixed-evaluation-pixels', action='store_true')
    parser.add_argument('--example-size-count', type=int)
    return parser


def generate_pixel_size_sweep(args: argparse.Namespace) -> Path:
    if args.samples_per_size <= 0:
        raise ValueError('--samples-per-size must be greater than zero.')
    if not args.threshold_grid:
        raise ValueError('--threshold-grid must contain at least one value.')
    _apply_training_defaults(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_name(args.device)
    model = _load_model(args.checkpoint, device, args.num_filters, args.dropout)
    pixel_sizes = args.pixel_sizes_angstrom or [args.train_pixel_size_angstrom * factor for factor in args.pixel_size_factors]
    if not pixel_sizes:
        raise ValueError('Provide at least one pixel size or pixel-size factor.')
    rows = []
    examples = []
    for pixel_size in pixel_sizes:
        pixel_size_factor = float(pixel_size) / float(args.train_pixel_size_angstrom)
        config, geometry = _build_random_config(args, pixel_size_factor)
        samples, example = _predict_sweep_samples(model, config, args, device)
        summary, evaluation = _evaluate_thresholds(samples, args, pixel_size_factor)
        row = _make_row(summary, float(pixel_size), pixel_size_factor, geometry, evaluation)
        rows.append(row)
        examples.append(example)
        f1 = row['f1']
        threshold = row['threshold_rel']
        print(f'pixel_size={float(pixel_size):.5f} A/px factor={pixel_size_factor:.3f} f1={f1:.3f} threshold={threshold:.2f}', flush=True)

    ordered = sorted(zip(rows, examples), key=lambda item: item[0]['pixel_size_angstrom'])
    rows = [row for row, _example in ordered]
    examples = [example for _row, example in ordered]
    _write_metrics_csv(args.output_dir / 'pixel_size_metrics.csv', rows)
    (args.output_dir / 'pixel_size_metrics.json').write_text(json.dumps(rows, indent=2))
    config_json = {
        'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'bottleneck_receptive_field_px': BOTTLENECK_RECEPTIVE_FIELD_PX,
        'tested_pixel_sizes_angstrom': [row['pixel_size_angstrom'] for row in rows],
        'training_parameter_source': args.training_parameter_source,
        'training_physical_priors': {
            'sigma_range_angstrom': [args.train_sigma_min * args.train_pixel_size_angstrom, args.train_sigma_max * args.train_pixel_size_angstrom],
            'target_sigma_angstrom': args.train_target_sigma * args.train_pixel_size_angstrom,
            'min_separation_range_angstrom': [
                args.train_min_separation_range_min * args.train_pixel_size_angstrom,
                args.train_min_separation_range_max * args.train_pixel_size_angstrom,
            ],
        },
        'generation_config_template': asdict(_build_random_config(args, 1.0)[0]),
    }
    (args.output_dir / 'pixel_size_sweep_config.json').write_text(json.dumps(config_json, indent=2))
    _save_accuracy_plot(args.output_dir / 'pixel_size_accuracy_vs_feature_rf.png', rows)
    _save_example_gallery(args.output_dir / 'pixel_size_target_output_examples.png', rows, examples, args.example_size_count)
    return args.output_dir / 'pixel_size_metrics.csv'


def main() -> int:
    output_path = generate_pixel_size_sweep(build_parser().parse_args())
    print(output_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
