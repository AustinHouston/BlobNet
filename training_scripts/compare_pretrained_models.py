from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from GombNet.metrics import aggregate_localization_metrics, evaluate_heatmap_localization
from GombNet.synthetic import (
    AseStructureProjectionConfig,
    ImageFormationConfig,
    PeriodicLatticeConfig,
    RandomMicroscopeImageConfig,
    TightSpacingRandomMicroscopeImageConfig,
    generate_microscope_image,
)
from GombNet.utils import resolve_torch_device
from GombNet.visualization import build_prediction_gallery, plot_generalization_summary
from training_scripts.external_model_adapters import (
    ATOMAI_PRETRAINED_URLS,
    ATOMSEGNET_DEFAULT_LOCALIZERS,
    ATOMSEGNET_WEIGHT_FILES,
    TEM_IMAGENET_REPO,
    AtomAICandidate,
    AtomSegNetCandidate,
    BlobNetCandidate,
    ModelCandidate,
    load_tem_imagenet_arrays,
    maybe_download_tem_imagenet,
)
from training_scripts.io_utils import nan_last, nan_low, safe_nanmean, write_csv, write_json


@dataclass(frozen=True)
class BenchmarkSample:
    case_name: str
    sample_id: str
    image: np.ndarray
    target: np.ndarray
    coordinates_yx: np.ndarray
    metadata: dict[str, Any]


@dataclass
class CandidateStatus:
    model: str
    family: str
    status: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare BlobNet against pretrained AtomSegNet and AtomAI models on shared synthetic STEM-style images.'
    )
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-families', nargs='+', default=['blobnet', 'atomsegnet', 'atomai'], choices=['blobnet', 'atomsegnet', 'atomai'])
    parser.add_argument('--download-models', action='store_true')
    parser.add_argument('--model-cache-dir', type=Path, default=None)
    parser.add_argument('--blobnet-checkpoint', type=Path, default=Path('outputs/inhom_background_unet_20epoch/unet/unet_best.pth'))
    parser.add_argument('--blobnet-num-filters', type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--blobnet-dropout', type=float, default=0.2)
    parser.add_argument('--atomsegnet-models', nargs='+', default=ATOMSEGNET_DEFAULT_LOCALIZERS)
    parser.add_argument('--atomsegnet-iterations', type=int, default=1)
    parser.add_argument('--atomai-models', nargs='+', default=list(ATOMAI_PRETRAINED_URLS))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--samples-per-case', type=int, default=8)
    parser.add_argument('--cases', nargs='+', default=['random', 'tight_random', 'cubic', 'hexagonal', 'graphene', 'ws2', 'sto'], choices=['random', 'tight_random', 'cubic', 'hexagonal', 'graphene', 'ws2', 'sto'])
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--min-atoms', type=int, default=80)
    parser.add_argument('--max-atoms', type=int, default=180)
    parser.add_argument('--min-separation', type=float, default=11.5)
    parser.add_argument('--min-separation-range-min', type=float, default=10.0)
    parser.add_argument('--min-separation-range-max', type=float, default=14.5)
    parser.add_argument('--tight-spacing-min', type=float, default=10.5)
    parser.add_argument('--tight-spacing-max', type=float, default=13.5)
    parser.add_argument('--sigma-min', type=float, default=2.0)
    parser.add_argument('--sigma-max', type=float, default=3.8)
    parser.add_argument('--background-min', type=float, default=0.0)
    parser.add_argument('--background-max', type=float, default=0.30)
    parser.add_argument('--inhom-background-min', type=float, default=0.0)
    parser.add_argument('--inhom-background-max', type=float, default=0.16)
    parser.add_argument('--low-freq-noise-min', type=float, default=0.05)
    parser.add_argument('--low-freq-noise-max', type=float, default=0.28)
    parser.add_argument('--read-noise-min', type=float, default=0.02)
    parser.add_argument('--read-noise-max', type=float, default=0.12)
    parser.add_argument('--total-counts-min', type=float, default=50.0)
    parser.add_argument('--total-counts-max', type=float, default=25_000.0)
    parser.add_argument('--blur-sigma-min', type=float, default=0.2)
    parser.add_argument('--blur-sigma-max', type=float, default=1.1)
    parser.add_argument('--edge-padding', type=int, default=16)
    parser.add_argument('--lattice-spacing-min', type=float, default=10.5)
    parser.add_argument('--lattice-spacing-max', type=float, default=14.5)
    parser.add_argument('--lattice-min-atoms', type=int, default=120)
    parser.add_argument('--structure-pixel-size-angstrom', type=float, default=0.1062231596676199)
    parser.add_argument('--threshold-grid', type=float, nargs='+', default=[0.15, 0.25, 0.35, 0.50, 0.65])
    parser.add_argument('--match-distance', type=float, default=3.0)
    parser.add_argument('--min-distance', type=int, default=3)
    parser.add_argument('--peak-window-size', type=int, default=5)
    parser.add_argument('--gallery-samples', type=int, default=3)
    parser.add_argument('--gallery-models', type=int, default=4)
    parser.add_argument('--download-tem-imagenet', action='store_true')
    parser.add_argument('--tem-imagenet-dir', type=Path, default=None)
    parser.add_argument('--tem-imagenet-max-samples', type=int, default=64)
    parser.add_argument('--tem-imagenet-coordinate-order', choices=['xy', 'yx'], default='xy')
    parser.add_argument('--best-models-to-test', type=int, default=3)
    return parser.parse_args()


def common_image_settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        'image_shape': (args.height, args.width),
        'sigma_range': (args.sigma_min, args.sigma_max),
        'background_range': (args.background_min, args.background_max),
        'inhomogeneous_background_range': (args.inhom_background_min, args.inhom_background_max),
        'low_frequency_noise_range': (args.low_freq_noise_min, args.low_freq_noise_max),
        'read_noise_std_range': (args.read_noise_min, args.read_noise_max),
        'total_counts_range': (args.total_counts_min, args.total_counts_max),
        'counts_per_pixel_range': None,
        'blur_sigma_range': (args.blur_sigma_min, args.blur_sigma_max),
        'edge_padding': args.edge_padding,
    }


def build_case_configs(args: argparse.Namespace) -> dict[str, ImageFormationConfig]:
    common = common_image_settings(args)
    configs: dict[str, ImageFormationConfig] = {
        'random': RandomMicroscopeImageConfig(min_atoms=args.min_atoms, max_atoms=args.max_atoms, min_separation=args.min_separation, min_separation_range=(args.min_separation_range_min, args.min_separation_range_max), **common),
        'tight_random': TightSpacingRandomMicroscopeImageConfig(min_atoms=args.min_atoms, max_atoms=args.max_atoms, nearest_neighbor_spacing_range=(args.tight_spacing_min, args.tight_spacing_max), spacing_jitter_fraction_range=(0.03, 0.08), min_spacing_fraction=0.86, **common),
        'cubic': PeriodicLatticeConfig(lattice_type='cubic', lattice_spacing_range=(args.lattice_spacing_min, args.lattice_spacing_max), jitter_std_range=(0.0, 0.25), vacancy_fraction_range=(0.0, 0.03), min_atoms=args.lattice_min_atoms, **common),
        'hexagonal': PeriodicLatticeConfig(lattice_type='hexagonal', lattice_spacing_range=(args.lattice_spacing_min, args.lattice_spacing_max), jitter_std_range=(0.0, 0.25), vacancy_fraction_range=(0.0, 0.03), min_atoms=args.lattice_min_atoms, **common),
    }
    structure_common = {**common, 'image_shape': (args.height, args.width), 'pixel_size_angstrom': args.structure_pixel_size_angstrom, 'rotation_range': (0.0, 180.0), 'position_jitter_std_range': (0.0, 0.08)}
    configs.update({
        'graphene': AseStructureProjectionConfig(structure_name='graphene', species_intensity_power=1.2, **structure_common),
        'ws2': AseStructureProjectionConfig(structure_name='ws2', species_intensity_power=1.6, **structure_common),
        'sto': AseStructureProjectionConfig(structure_name='sto', species_intensity_power=1.5, **structure_common),
    })
    return {name: configs[name] for name in args.cases}


def generate_synthetic_samples(args: argparse.Namespace) -> list[BenchmarkSample]:
    samples = []
    for case_index, (case_name, config) in enumerate(build_case_configs(args).items()):
        for sample_index in range(int(args.samples_per_case)):
            record = generate_microscope_image(config, np.random.default_rng(args.seed + case_index * 100_000 + sample_index))
            metadata = {key: value for key, value in record.items() if key not in {'image', 'target', 'count_map', 'config'}}
            samples.append(BenchmarkSample(case_name, f'{case_name}_{sample_index:04d}', np.asarray(record['image'], dtype=np.float32), np.asarray(record['target'], dtype=np.float32), np.asarray(record['coordinates'], dtype=np.float32), metadata))
    return samples


def load_tem_imagenet_samples(dataset_dir: Path, max_samples: int, coordinate_order: str) -> list[BenchmarkSample]:
    return [
        BenchmarkSample('tem_imagenet_v1.3', item['sample_id'], item['image'], np.zeros_like(item['image'], dtype=np.float32), item['coordinates_yx'].astype(np.float32), item['metadata'])
        for item in load_tem_imagenet_arrays(dataset_dir, max_samples, coordinate_order)
    ]


def build_candidates(args: argparse.Namespace, device: torch.device) -> list[ModelCandidate]:
    cache_dir = args.model_cache_dir or args.output_dir / 'model_cache'
    candidates: list[ModelCandidate] = []
    if 'blobnet' in args.model_families:
        candidates.append(BlobNetCandidate(args.blobnet_checkpoint, device, args.blobnet_num_filters, args.blobnet_dropout))
    if 'atomsegnet' in args.model_families:
        names = sorted(ATOMSEGNET_WEIGHT_FILES) if args.atomsegnet_models == ['all'] else args.atomsegnet_models
        candidates.extend(AtomSegNetCandidate(name, cache_dir, args.download_models, device, args.atomsegnet_iterations) for name in names)
    if 'atomai' in args.model_families:
        candidates.extend(AtomAICandidate(name, cache_dir, args.download_models, device) for name in args.atomai_models)
    return candidates


def setup_available_candidates(candidates: Sequence[ModelCandidate]) -> tuple[list[ModelCandidate], list[CandidateStatus]]:
    available, statuses = [], []
    for candidate in candidates:
        try:
            print(f'Preparing {candidate.name}', flush=True)
            candidate.setup()
            available.append(candidate)
            statuses.append(CandidateStatus(candidate.name, candidate.family, 'available', 'ready'))
        except Exception as exc:  # noqa: BLE001 - keep the comparison running when optional models are missing.
            statuses.append(CandidateStatus(candidate.name, candidate.family, 'skipped', str(exc)))
            print(f'Skipping {candidate.name}: {exc}', flush=True)
    return available, statuses


def predict_all(models: Sequence[ModelCandidate], samples: Sequence[BenchmarkSample]) -> dict[tuple[str, str], np.ndarray]:
    predictions = {}
    for model in models:
        print(f'Predicting with {model.name} on {len(samples)} samples', flush=True)
        for sample in samples:
            predictions[(model.name, sample.sample_id)] = model.predict_heatmap(sample.image)
    return predictions


def metric_row(model: ModelCandidate, test_case: str, threshold: float, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        'model': model.name, 'family': model.family, 'test_case': test_case, 'threshold_rel': float(threshold),
        'precision': float(summary['precision']), 'recall': float(summary['recall']), 'f1': float(summary['f1']),
        'mean_error': float(summary['mean_error']), 'median_error': float(summary['median_error']), 'rmse': float(summary['rmse']),
        'tp': int(summary['tp']), 'fp': int(summary['fp']), 'fn': int(summary['fn']), 'samples': int(summary['samples']),
    }


def evaluate_prediction_grid(
    models: Sequence[ModelCandidate],
    samples: Sequence[BenchmarkSample],
    predictions: dict[tuple[str, str], np.ndarray],
    thresholds: Sequence[float],
    *,
    match_distance: float,
    min_distance: int,
    peak_window_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float], list[dict[str, Any]]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for model in models:
        for threshold in thresholds:
            for sample in samples:
                result = evaluate_heatmap_localization(predictions[(model.name, sample.sample_id)], sample.coordinates_yx, threshold_rel=float(threshold), min_distance=min_distance, peak_window_size=peak_window_size, match_distance=match_distance)
                grouped.setdefault((model.name, float(threshold), sample.case_name), []).append(result)

    model_by_name = {model.name: model for model in models}
    rows = [metric_row(model_by_name[model_name], case_name, threshold, aggregate_localization_metrics(results)) for (model_name, threshold, case_name), results in sorted(grouped.items())]
    best_thresholds, threshold_rows = {}, []
    for model in models:
        candidates = []
        for threshold in thresholds:
            case_rows = [row for row in rows if row['model'] == model.name and float(row['threshold_rel']) == float(threshold)]
            if case_rows:
                item = {'model': model.name, 'family': model.family, 'threshold_rel': float(threshold), 'mean_f1': safe_nanmean([float(row['f1']) for row in case_rows]), 'mean_rmse': safe_nanmean([float(row['rmse']) for row in case_rows]), 'cases': len(case_rows)}
                threshold_rows.append(item)
                candidates.append(item)
        if candidates:
            candidates.sort(key=lambda item: (-nan_low(float(item['mean_f1'])), nan_last(float(item['mean_rmse']))))
            best_thresholds[model.name] = float(candidates[0]['threshold_rel'])
    return rows, best_thresholds, threshold_rows


def select_best_rows(rows: Sequence[dict[str, Any]], best_thresholds: dict[str, float]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if best_thresholds.get(str(row['model'])) == float(row['threshold_rel'])]


def rank_models(best_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ranking = []
    for model_name in sorted({str(row['model']) for row in best_rows}):
        rows = [row for row in best_rows if row['model'] == model_name]
        ranking.append({
            'model': model_name,
            'family': str(rows[0]['family']),
            'best_threshold_rel': sorted({float(row['threshold_rel']) for row in rows})[0],
            'mean_f1': safe_nanmean([float(row['f1']) for row in rows]),
            'mean_rmse': safe_nanmean([float(row['rmse']) for row in rows]),
            'mean_precision': safe_nanmean([float(row['precision']) for row in rows]),
            'mean_recall': safe_nanmean([float(row['recall']) for row in rows]),
            'cases': len(rows),
        })
    ranking.sort(key=lambda item: (-nan_low(float(item['mean_f1'])), nan_last(float(item['mean_rmse']))))
    return ranking


def save_synthetic_gallery(
    output_path: Path,
    models: Sequence[ModelCandidate],
    samples: Sequence[BenchmarkSample],
    predictions: dict[tuple[str, str], np.ndarray],
    best_thresholds: dict[str, float],
    args: argparse.Namespace,
) -> Path | None:
    if not models or not samples or args.gallery_samples <= 0 or args.gallery_models <= 0:
        return None
    rows = []
    for sample in samples[: args.gallery_samples]:
        for model in models[: args.gallery_models]:
            heatmap = predictions[(model.name, sample.sample_id)]
            threshold = best_thresholds.get(model.name, 0.35)
            result = evaluate_heatmap_localization(heatmap, sample.coordinates_yx, threshold_rel=threshold, min_distance=args.min_distance, peak_window_size=args.peak_window_size, match_distance=args.match_distance)
            rmse_text = 'n/a' if np.isnan(result['rmse']) else f'{result["rmse"]:.3f} px'
            rows.append({
                'label': f'{model.name} | {sample.case_name}',
                'metrics_text': f'sample={sample.sample_id}  atoms={len(sample.coordinates_yx)}  pred={len(result["predicted_coordinates"])}  thr={threshold:.2f}  F1={result["f1"]:.3f}  RMSE={rmse_text}',
                'image': sample.image, 'target': sample.target, 'prediction': heatmap,
                'true_coords': sample.coordinates_yx, 'predicted_coords': result['predicted_coordinates'],
            })
    return build_prediction_gallery(rows, output_path)


def evaluate_tem_imagenet(
    models: Sequence[ModelCandidate],
    samples: Sequence[BenchmarkSample],
    best_thresholds: dict[str, float],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        print(f'Evaluating {model.name} on {len(samples)} TEM-ImageNet samples', flush=True)
        threshold = best_thresholds.get(model.name, 0.35)
        results = [
            evaluate_heatmap_localization(model.predict_heatmap(sample.image), sample.coordinates_yx, threshold_rel=threshold, min_distance=args.min_distance, peak_window_size=args.peak_window_size, match_distance=args.match_distance)
            for sample in samples
        ]
        if results:
            rows.append(metric_row(model, 'tem_imagenet_v1.3', threshold, aggregate_localization_metrics(results)))
    return rows


def save_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    write_json({
        'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'tem_imagenet_repo': TEM_IMAGENET_REPO,
        'atomai_pretrained': ATOMAI_PRETRAINED_URLS,
        'synthetic_cases': {name: {'class': type(config).__name__, 'config': asdict(config)} for name, config in build_case_configs(args).items()},
    }, output_dir / 'comparison_config.json')


def write_rows(base: Path, rows: Sequence[dict[str, Any]]) -> None:
    write_json(rows, base.with_suffix('.json'))
    write_csv(rows, base.with_suffix('.csv'))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(args, args.output_dir)
    device = resolve_torch_device(args.device, verbose=True)
    samples = generate_synthetic_samples(args)
    print(f'Generated {len(samples)} synthetic samples across {len(args.cases)} case(s)', flush=True)

    models, statuses = setup_available_candidates(build_candidates(args, device))
    write_rows(args.output_dir / 'model_status', [status.__dict__ for status in statuses])
    if not models:
        raise RuntimeError('No models were available. Check the BlobNet checkpoint, install atomai, or pass --download-models for external pretrained weights.')

    predictions = predict_all(models, samples)
    synthetic_rows, best_thresholds, threshold_rows = evaluate_prediction_grid(models, samples, predictions, args.threshold_grid, match_distance=args.match_distance, min_distance=args.min_distance, peak_window_size=args.peak_window_size)
    best_rows = select_best_rows(synthetic_rows, best_thresholds)
    ranking = rank_models(best_rows)
    write_rows(args.output_dir / 'synthetic_metrics_by_threshold', synthetic_rows)
    write_rows(args.output_dir / 'threshold_ranking', threshold_rows)
    write_rows(args.output_dir / 'synthetic_best_metrics', best_rows)
    write_rows(args.output_dir / 'model_ranking', ranking)

    model_order = [row['model'] for row in ranking]
    model_by_name = {model.name: model for model in models}
    ordered_models = [model_by_name[name] for name in model_order if name in model_by_name]
    plot_generalization_summary(best_rows, args.output_dir, case_order=args.cases, model_order=model_order, filename='synthetic_generalization_summary.png', title='BlobNet / AtomSegNet / AtomAI Synthetic Comparison')
    save_synthetic_gallery(args.output_dir / 'synthetic_prediction_gallery.png', ordered_models, samples, predictions, best_thresholds, args)

    tem_dir = args.tem_imagenet_dir or args.output_dir / 'TEM-ImageNet-v1.3'
    maybe_download_tem_imagenet(tem_dir, args.download_tem_imagenet)
    tem_rows: list[dict[str, Any]] = []
    if tem_dir.exists():
        tem_samples = load_tem_imagenet_samples(tem_dir, args.tem_imagenet_max_samples, args.tem_imagenet_coordinate_order)
        if tem_samples:
            tem_rows = evaluate_tem_imagenet(ordered_models[: max(1, int(args.best_models_to_test))], tem_samples, best_thresholds, args)
            write_rows(args.output_dir / 'tem_imagenet_metrics', tem_rows)
        else:
            write_json({'status': 'skipped', 'message': f'No TEM-ImageNet samples found in {tem_dir}'}, args.output_dir / 'tem_imagenet_metrics.json')
    else:
        write_json({'status': 'skipped', 'message': f'TEM-ImageNet directory {tem_dir} does not exist. Pass --download-tem-imagenet to clone it, or --tem-imagenet-dir to point at an existing checkout.'}, args.output_dir / 'tem_imagenet_metrics.json')

    best = ranking[0]
    print(f"Finished comparison. Best synthetic mean F1: {best['model']} (mean_f1={best['mean_f1']:.4f}, mean_rmse={best['mean_rmse']:.4f}, threshold={best['best_threshold_rel']:.2f}).", flush=True)
    if tem_rows:
        best_tem = sorted(tem_rows, key=lambda row: (-nan_low(float(row['f1'])), nan_last(float(row['rmse']))))[0]
        print(f"Best TEM-ImageNet F1 among tested models: {best_tem['model']} (f1={best_tem['f1']:.4f}, rmse={best_tem['rmse']:.4f}).", flush=True)
    print(f'Saved comparison outputs to {args.output_dir}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
