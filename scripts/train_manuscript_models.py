from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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


def run_command(command: list[str]) -> None:
    print('\n' + ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate manuscript datasets, train U-Nets, and rebuild figures.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/manuscript_figures'))
    parser.add_argument('--epochs', type=int, help='Override epochs for every model config.')
    parser.add_argument('--batch-size', type=int, help='Override batch size for every model config.')
    parser.add_argument('--skip-figures', action='store_true')
    args = parser.parse_args()

    for name, config_path in DATASET_CONFIGS.items():
        run_command([
            sys.executable,
            'scripts/generate_training_dataset.py',
            '--config',
            str(config_path),
            '--overwrite',
        ])

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
        ])

    print('\nManuscript training pipeline complete.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
