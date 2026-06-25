from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from blobnet.synthetic import (
    AseStructureProjectionConfig,
    PeriodicLatticeConfig,
    RandomAtomImageConfig,
    generate_and_save_dataset_splits,
)


CONFIG_TYPES = {
    'random': RandomAtomImageConfig,
    'periodic_lattice': PeriodicLatticeConfig,
    'ase_structure': AseStructureProjectionConfig,
}


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate and save a BlobNet training dataset from YAML.')
    parser.add_argument('--config', type=Path, default=Path('configs/dataset_configs/random.yaml'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--train-samples', type=int)
    parser.add_argument('--val-samples', type=int)
    parser.add_argument('--test-samples', type=int)
    parser.add_argument('--num-workers', type=int, default=0, help='Parallel generation workers. Use 0 for all available CPUs.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    raw_config = yaml.safe_load(args.config.read_text())
    dataset_settings = raw_config['dataset']
    split_settings = dataset_settings['splits']
    dataset_type = dataset_settings['type']
    if dataset_type not in CONFIG_TYPES:
        raise ValueError(f'Unsupported dataset type {dataset_type!r}. Choose from {sorted(CONFIG_TYPES)}.')

    output_dir = args.output_dir or Path(dataset_settings['output_dir'])
    split_counts = {
        'train': args.train_samples if args.train_samples is not None else int(split_settings['train']),
        'val': args.val_samples if args.val_samples is not None else int(split_settings['val']),
        'test': args.test_samples if args.test_samples is not None else int(split_settings['test']),
    }
    existing_samples = list(output_dir.glob('*/*.npz'))
    if existing_samples and not args.overwrite:
        raise FileExistsError(f'{output_dir} already contains NPZ samples. Pass --overwrite to replace them.')
    for path in existing_samples:
        path.unlink()

    image_config = CONFIG_TYPES[dataset_type](**raw_config['parameters'])
    saved = generate_and_save_dataset_splits(
        output_dir,
        train_samples=split_counts['train'],
        val_samples=split_counts['val'],
        test_samples=split_counts['test'],
        config=image_config,
        seed=int(dataset_settings.get('seed', 0)),
        prefix=dataset_settings.get('prefix', 'sample'),
        num_workers=args.num_workers,
    )

    manifest = {
        'name': dataset_settings['name'],
        'type': dataset_type,
        'seed': int(dataset_settings.get('seed', 0)),
        'splits': {name: len(paths) for name, paths in saved.items()},
        'num_workers': int(args.num_workers),
        'parameters': asdict(image_config),
        'source_config': str(args.config),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'dataset_manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f'Saved {sum(split_counts.values())} samples to {output_dir}')
    print(f'Splits: {split_counts}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
