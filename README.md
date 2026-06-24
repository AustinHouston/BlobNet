# BlobNet

BlobNet is a focused U-Net pipeline for atom localization in atomic-resolution STEM images.

The maintained workflow has two explicit steps:

1. Generate deterministic train, validation, and test datasets from YAML.
2. Train U-Net from the saved dataset and write checkpoints, losses, and metrics.

## Environment

```bash
uv sync
```

## Generate A Dataset

Dataset parameters live in `configs/dataset_configs/`. The default configuration generates random microscope images:

```bash
uv run blobnet-generate-dataset \
  --config configs/dataset_configs/random_microscope.yaml
```

Useful overrides:

```bash
uv run blobnet-generate-dataset \
  --config configs/dataset_configs/random_microscope.yaml \
  --output-dir /tmp/blobnet_dataset \
  --train-samples 128 \
  --val-samples 32 \
  --test-samples 32
```

The generator writes one compressed NPZ file per sample under `train/`, `val/`, and `test/`, plus `dataset_manifest.yaml` with the resolved generation parameters.

## Train U-Net

Model and training parameters live in `configs/model_configs/`:

```bash
uv run blobnet-train \
  --config configs/model_configs/unet.yaml
```

Common settings can be overridden without editing YAML:

```bash
uv run blobnet-train \
  --config configs/model_configs/unet.yaml \
  --dataset-dir /tmp/blobnet_dataset \
  --output-dir /tmp/blobnet_training \
  --epochs 10 \
  --batch-size 4 \
  --learning-rate 0.001 \
  --device auto
```

Training writes:

- `unet_best.pth`
- `unet_loss_history.npz`
- `loss_history.csv`
- `loss_curves.png`
- `training_metrics.json`
- `resolved_config.yaml`

## Experimental Images

`notebooks/emd_reader.ipynb` is a self-contained pyTEMlib notebook for loading and plotting every EMD file in `experimental_data/`.

## Python API

```python
from blobnet import RandomMicroscopeImageConfig, generate_microscope_image

config = RandomMicroscopeImageConfig(image_shape=(256, 256))
sample = generate_microscope_image(config)
image, target = sample['image'], sample['target']
```

## Layout

```text
blobnet/                     reusable models, data generation, metrics, and plotting
configs/
  dataset_configs/           saved-dataset generation parameters
  model_configs/             model, loss, and training parameters
scripts/
  generate_training_dataset.py
  train_unet.py
  check_mps.py
notebooks/                   dataset and experimental-image exploration
experimental_data/           tracked EMD examples
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions.
