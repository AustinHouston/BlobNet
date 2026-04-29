# Blob-Net

Blob-Net is a focused U-Net pipeline for atom localization in atomic-resolution STEM images.

The current repo is organized around one workflow:
- generate synthetic Gaussian training data, including edge-cutoff atoms
- train a U-Net to predict normalized Gaussian label maps
- evaluate on random, cubic, hexagonal, and ASE-projected structure datasets
- run inference on real EMD STEM images and visualize the results

## Environment

Create the repo environment with `uv`:

```bash
uv sync
```

When running on Apple Silicon, the most reliable path is:

```bash
source .venv/bin/activate
uv run python training_scripts/check_mps.py
```

## Main Scripts

- `training_scripts/generate_dataset.py`
  Creates `.npz` datasets or train/val/test splits.

- `training_scripts/export_examples.py`
  Exports a few PNG examples of the current synthetic training distribution.

- `training_scripts/train_unet.py`
  Trains the U-Net, evaluates on all synthetic test pipelines, and runs real-image inference.

- `training_scripts/analyze_real_image.py`
  Detects and fits blobs in a real EMD image to estimate realistic sigma and spacing priors.

- `training_scripts/check_mps.py`
  Verifies that PyTorch can actually allocate and compute on `mps`.

## Quick Start

Export a few edge-aware training examples:

```bash
source .venv/bin/activate
uv run python training_scripts/export_examples.py
```

Generate a split dataset on disk:

```bash
source .venv/bin/activate
uv run python training_scripts/generate_dataset.py \
  --output-dir /tmp/blobnet_dataset \
  --train-samples 64 \
  --val-samples 16 \
  --test-samples 16 \
  --preview-path /tmp/blobnet_dataset/preview.png
```

Train and evaluate the U-Net:

```bash
source .venv/bin/activate
uv run python training_scripts/train_unet.py \
  --output-dir /tmp/blobnet_run \
  --device auto
```

Analyze the real image to estimate sigma and spacing:

```bash
source .venv/bin/activate
uv run python training_scripts/analyze_real_image.py \
  --input-path real_data/WS2.emd \
  --output-dir /tmp/blobnet_real_image_analysis
```

## Outputs

A training run writes:
- `benchmark_metrics.json/csv`
- summary figures for periodic and structure-based tests
- prediction galleries
- offset clouds
- real-image overview and crop overlays
- the best checkpoint in `output_dir/unet/unet_best.pth`

## Project Layout

```text
GombNet/
  synthetic.py        synthetic data generation
  networks.py         U-Net model
  loss_func.py        heatmap regression loss
  metrics.py          localization metrics
  real_image.py       EMD loading and tiled inference helpers
  visualization.py    plotting and gallery helpers
  utils.py            device resolution and training loop

training_scripts/
  train_unet.py
  generate_dataset.py
  export_examples.py
  analyze_real_image.py
  check_mps.py
```
