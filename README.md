# BlobNet

BlobNet is a focused U-Net pipeline for atom localization in atomic-resolution STEM images.

The maintained workflow has two explicit steps:

1. Generate deterministic train, validation, and test datasets from YAML.
2. Train U-Net from the saved dataset and write checkpoints, losses, and metrics.

## Environment

```bash
uv sync
```

On Windows or Linux with an NVIDIA GPU, BlobNet uses the CUDA 12.8 PyTorch wheel index through `pyproject.toml`. Refresh the lockfile and environment after pulling changes:

```bash
uv lock
uv sync
```

Check that the `uv` environment can see CUDA before starting a long training run:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda device')"
```

If CUDA is still unavailable, confirm that the NVIDIA driver is visible outside Python:

```bash
nvidia-smi
```

## Generate A Dataset

Dataset parameters live in `configs/dataset_configs/`. The default configuration generates random atom images:

- `random.yaml`
- `square.yaml`
- `hexagonal.yaml`

```bash
uv run blobnet-generate-dataset \
  --config configs/dataset_configs/random.yaml
```

Useful overrides:

```bash
uv run blobnet-generate-dataset \
  --config configs/dataset_configs/random.yaml \
  --output-dir /tmp/blobnet_dataset \
  --train-samples 128 \
  --val-samples 32 \
  --test-samples 32 \
  --num-workers 0
```

The generator writes one compressed NPZ file per sample under `train/`, `val/`, and `test/`, plus `dataset_manifest.yaml` with the resolved generation parameters.
Use `--num-workers 0` to use all available CPU cores, or pass a positive worker count to cap CPU use.

## Train U-Net

Model and training parameters live in `configs/model_configs/`:

```bash
uv run blobnet-train \
  --config configs/model_configs/base_unet.yaml
```

Common settings can be overridden without editing YAML:

```bash
uv run blobnet-train \
  --config configs/model_configs/base_unet.yaml \
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

## Train Manuscript Models

The manuscript comparison uses three matched U-Net configs:

- `configs/model_configs/random_unet.yaml`
- `configs/model_configs/square_unet.yaml`
- `configs/model_configs/hexagonal_unet.yaml`

Run the full pipeline from existing datasets to trained models and figures:

```bash
uv run blobnet-train-manuscript --device cuda
```

This command reuses complete random, square, and hexagonal datasets when they already match the config split counts, trains all three U-Nets from scratch, checks that each checkpoint was written, and rebuilds the manuscript figures.
If a dataset is missing, the script generates it. If a dataset is partial or has mismatched split counts, the script stops instead of silently mixing old and new data.

Start from training when the datasets are already generated and should not be touched:

```bash
uv run blobnet-train-manuscript --device cuda --skip-dataset-generation
```

Regenerate every dataset before training when you need fresh data:

```bash
uv run blobnet-train-manuscript --device cuda --regenerate-datasets
```

Dataset generation runs in parallel by default when it is needed; use `--dataset-workers N` to cap it.

## Experimental Images

`notebooks/emd_reader.ipynb` is a self-contained pyTEMlib notebook for loading and plotting every EMD file in `experimental_data/`.

## Python API

```python
from blobnet import RandomAtomImageConfig, generate_atom_image

config = RandomAtomImageConfig(image_shape=(256, 256))
sample = generate_atom_image(config)
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
