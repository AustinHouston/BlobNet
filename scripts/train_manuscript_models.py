from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


DATASET_CONFIGS = {
    'random': Path('configs/dataset_configs/random.yaml'),
    'square': Path('configs/dataset_configs/square.yaml'),
    'hexagonal': Path('configs/dataset_configs/hexagonal.yaml'),
}

MODEL_CONFIGS = {
    'random': Path('configs/model_configs/random_unet.yaml'),
    'square': Path('configs/model_configs/square_unet.yaml'),
    'hexagonal': Path('configs/model_configs/hexagonal_unet.yaml'),
}

CHECKPOINTS = {
    'random': Path('outputs/manuscript_models/random/unet_best.pth'),
    'square': Path('outputs/manuscript_models/square/unet_best.pth'),
    'hexagonal': Path('outputs/manuscript_models/hexagonal/unet_best.pth'),
}

PIXEL_SIZE_SWEEP_DIR = Path('outputs/blobnet_pixel_size_sweep_random_4x')


def run_command(command: list[str]) -> None:
    print('\n' + ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def read_dataset_plan(config_path: Path) -> tuple[Path, dict[str, int]]:
    config = yaml.safe_load(config_path.read_text())
    dataset_settings = config['dataset']
    split_settings = dataset_settings['splits']
    output_dir = Path(dataset_settings['output_dir'])
    expected_counts = {split: int(count) for split, count in split_settings.items()}
    return output_dir, expected_counts


def count_dataset_samples(output_dir: Path, expected_counts: dict[str, int]) -> dict[str, int]:
    return {split: len(list((output_dir / split).glob('*.npz'))) for split in expected_counts}


def format_counts(counts: dict[str, int]) -> str:
    return ', '.join(f'{split}={count}' for split, count in counts.items())


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate manuscript datasets, train U-Nets, and rebuild figures.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='cuda')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/manuscript_figures'))
    parser.add_argument('--epochs', type=int, help='Override epochs for every model config.')
    parser.add_argument('--batch-size', type=int, help='Override batch size for every model config.')
    parser.add_argument('--dataset-workers', type=int, default=0, help='Parallel workers for dataset generation. Use 0 for all available CPUs.')
    parser.add_argument('--skip-dataset-generation', action='store_true', help='Start at training and leave dataset directories untouched.')
    parser.add_argument('--regenerate-datasets', action='store_true', help='Overwrite and rebuild datasets before training.')
    parser.add_argument('--skip-pixel-size-sweep', action='store_true')
    parser.add_argument('--pixel-size-sweep-output-dir', type=Path, default=PIXEL_SIZE_SWEEP_DIR)
    parser.add_argument('--pixel-size-sweep-samples', type=int, default=64)
    parser.add_argument('--skip-figures', action='store_true')
    args = parser.parse_args()
    if args.skip_dataset_generation and args.regenerate_datasets:
        parser.error('--skip-dataset-generation and --regenerate-datasets cannot be used together.')

    if args.skip_dataset_generation:
        print('\nSkipping dataset generation; training will use dataset paths from the model configs.', flush=True)
    else:
        for name, config_path in DATASET_CONFIGS.items():
            dataset_dir, expected_counts = read_dataset_plan(config_path)
            actual_counts = count_dataset_samples(dataset_dir, expected_counts)
            if not args.regenerate_datasets and actual_counts == expected_counts:
                print(
                    f'\n{name}: using existing dataset at {dataset_dir} '
                    f'({format_counts(actual_counts)})',
                    flush=True,
                )
                continue
            if not args.regenerate_datasets and any(actual_counts.values()):
                raise FileExistsError(
                    f'{name}: found a partial or mismatched dataset at {dataset_dir}. '
                    f'Expected {format_counts(expected_counts)}, found {format_counts(actual_counts)}. '
                    'Pass --regenerate-datasets to overwrite and rebuild it, or pass '
                    '--skip-dataset-generation to start training from the model-config dataset paths.'
                )

            command = [
                sys.executable,
                'scripts/generate_training_dataset.py',
                '--config',
                str(config_path),
                '--num-workers',
                str(args.dataset_workers),
            ]
            if args.regenerate_datasets:
                command.append('--overwrite')
            run_command(command)

    for name, config_path in MODEL_CONFIGS.items():
        command = [
            sys.executable,
            'scripts/train_unet.py',
            '--config',
            str(config_path),
            '--device',
            args.device,
        ]
        if args.epochs is not None:
            command.extend(['--epochs', str(args.epochs)])
        if args.batch_size is not None:
            command.extend(['--batch-size', str(args.batch_size)])
        run_command(command)

        checkpoint = CHECKPOINTS[name]
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Expected checkpoint was not created: {checkpoint}')
        metrics_path = checkpoint.parent / 'training_metrics.json'
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            print(f"{name} test_loss={float(metrics['test_loss']):.6f}", flush=True)

    if not args.skip_pixel_size_sweep:
        run_command([
            sys.executable,
            'scripts/run_pixel_size_sweep.py',
            '--output-dir',
            str(args.pixel_size_sweep_output_dir),
            '--checkpoint',
            str(CHECKPOINTS['random']),
            '--device',
            args.device,
            '--samples-per-size',
            str(args.pixel_size_sweep_samples),
        ])

    if not args.skip_figures:
        run_command([
            sys.executable,
            'scripts/make_manuscript_figures.py',
            'all',
            '--output-dir',
            str(args.output_dir),
            '--device',
            args.device,
            '--square-checkpoint',
            str(CHECKPOINTS['square']),
            '--hexagonal-checkpoint',
            str(CHECKPOINTS['hexagonal']),
            '--random-checkpoint',
            str(CHECKPOINTS['random']),
            '--checkpoint',
            str(CHECKPOINTS['random']),
            '--sweep-csv',
            str(args.pixel_size_sweep_output_dir / 'pixel_size_metrics.csv'),
        ])

    print('\nManuscript training pipeline complete.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
