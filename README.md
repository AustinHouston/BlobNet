# Blob-Net

Blob-Net is a focused U-Net pipeline for atom localization in atomic-resolution STEM images.

The current repo is organized around one workflow:
- generate synthetic microscope-image training data, including edge-cutoff atoms
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

- `training_scripts/export_low_count_demo.py`
  Exports a total-count or counts-per-pixel sweep for inspecting low-signal synthetic regimes.

- `training_scripts/train_unet.py`
  Trains the U-Net, evaluates on all synthetic test pipelines, and runs real-image inference.

- `training_scripts/run_dataset_study.py`
  Trains separate U-Nets on random, tight-spacing random, square-lattice, and hexagonal-lattice datasets, then compares every train/test pairing.

- `training_scripts/analyze_real_image.py`
  Detects and fits blobs in a real EMD image to estimate realistic sigma and spacing priors.

- `training_scripts/check_mps.py`
  Verifies that PyTorch can actually allocate and compute on `mps`.

## Notebooks

- `notebooks/dataset_generation_playground.ipynb`
  Interactive playground for inspecting random, periodic, ASE-projected, and low-count synthetic dataset generation.

- `notebooks/periodic_lattice_performance.ipynb`
  Post-training notebook for inspecting a training image, loss curves, cubic/hexagonal predictions, and offset clouds.

## Quick Start

Export a few edge-aware training examples:

```bash
source .venv/bin/activate
uv run python training_scripts/export_examples.py
```

Export a low-signal total-count sweep:

```bash
source .venv/bin/activate
uv run python training_scripts/export_low_count_demo.py \
  --output-dir /tmp/blobnet_low_count_demo \
  --total-counts 0 10 25 50 100 250 1000 5000
```

Or sweep counts per pixel:

```bash
source .venv/bin/activate
uv run python training_scripts/export_low_count_demo.py \
  --output-dir /tmp/blobnet_low_count_demo_cpp \
  --counts-per-pixel 0 0.0001 0.001 0.01
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

Run the four-dataset training study:

```bash
source .venv/bin/activate
uv run python training_scripts/run_dataset_study.py \
  --output-dir /tmp/blobnet_dataset_study \
  --epochs 10 \
  --output-sample-indices 0 1 2 \
  --progress-interval 25 \
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

A dataset study run writes:
- `dataset_study_metrics.json/csv` and `dataset_study_ranking.json/csv`
- grouped, mean-metric, and in-distribution bar charts
- train/test heatmaps, loss curves, prediction gallery, and offset clouds
- `model_output_series/*.png` comparisons showing the same held-out images through each trained model
- one best checkpoint per training dataset

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
  run_dataset_study.py
  generate_dataset.py
  export_examples.py
  export_low_count_demo.py
  analyze_real_image.py
  check_mps.py
```
