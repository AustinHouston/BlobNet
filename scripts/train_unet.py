from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from blobnet.loss_func import CombinedGaussianLoss
from blobnet.networks import build_unet
from blobnet.synthetic import SavedMicroscopeDataset
from blobnet.utils import resolve_torch_device, train_model


def main() -> int:
    parser = argparse.ArgumentParser(description='Train BlobNet U-Net from a saved NPZ dataset.')
    parser.add_argument('--config', type=Path, default=Path('configs/model_configs/unet.yaml'))
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    model_settings = config['model']
    loss_settings = config['loss']
    training = config['training']
    dataset_dir = args.dataset_dir or Path(config['dataset']['path'])
    output_dir = args.output_dir or Path(training['output_dir'])
    epochs = args.epochs if args.epochs is not None else int(training['epochs'])
    batch_size = args.batch_size if args.batch_size is not None else int(training['batch_size'])
    learning_rate = args.learning_rate if args.learning_rate is not None else float(training['learning_rate'])
    device_name = args.device or training.get('device', 'auto')
    num_workers = args.num_workers if args.num_workers is not None else int(training.get('num_workers', 0))
    seed = args.seed if args.seed is not None else int(training.get('seed', 0))

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_torch_device(device_name, verbose=True)
    datasets = {split: SavedMicroscopeDataset(dataset_dir / split) for split in ('train', 'val', 'test')}
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == 'train',
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        for split, dataset in datasets.items()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        **config,
        'dataset': {'path': str(dataset_dir)},
        'training': {
            **training,
            'output_dir': str(output_dir),
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'device': device_name,
            'num_workers': num_workers,
            'seed': seed,
        },
    }
    (output_dir / 'resolved_config.yaml').write_text(yaml.safe_dump(resolved_config, sort_keys=False))

    model = build_unet(
        input_channels=int(model_settings.get('input_channels', 1)),
        num_classes=int(model_settings.get('output_channels', 1)),
        num_filters=model_settings['filters'],
        dropout=float(model_settings['dropout']),
    )
    criterion = CombinedGaussianLoss(**loss_settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    started = time.time()
    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        n_epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_name=str(output_dir / 'unet'),
        progress_interval=training.get('progress_interval'),
        early_stopping_patience=training.get('early_stopping_patience'),
        early_stopping_min_delta=float(training.get('early_stopping_min_delta', 0.0)),
    )

    model.eval()
    test_loss = 0.0
    with torch.inference_mode():
        for images, targets in loaders['test']:
            test_loss += float(criterion(model(images.to(device)), targets.to(device)).item())
    test_loss /= len(loaders['test'])

    loss_rows = [
        {'epoch': index + 1, 'train_loss': train_loss, 'val_loss': val_loss}
        for index, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses))
    ]
    with (output_dir / 'loss_history.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'train_loss', 'val_loss'])
        writer.writeheader()
        writer.writerows(loss_rows)

    best_epoch = int(np.argmin(val_losses)) + 1
    metrics = {
        'device': str(device),
        'epochs_completed': len(train_losses),
        'best_epoch': best_epoch,
        'best_validation_loss': float(val_losses[best_epoch - 1]),
        'final_training_loss': float(train_losses[-1]),
        'final_validation_loss': float(val_losses[-1]),
        'test_loss': test_loss,
        'training_seconds': time.time() - started,
        'train_samples': len(datasets['train']),
        'val_samples': len(datasets['val']),
        'test_samples': len(datasets['test']),
        'trainable_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    (output_dir / 'training_metrics.json').write_text(json.dumps(metrics, indent=2))

    epoch_numbers = np.arange(1, len(train_losses) + 1)
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(epoch_numbers, train_losses, marker='o', label='Training')
    axis.plot(epoch_numbers, val_losses, marker='o', label='Validation')
    axis.set(xlabel='Epoch', ylabel='Loss', title='BlobNet U-Net Training Loss')
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / 'loss_curves.png', dpi=200)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f'Saved model and metrics to {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
