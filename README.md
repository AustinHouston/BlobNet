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

- `training_scripts/compare_pretrained_models.py`
  Compares Blob-Net with pretrained AtomSegNet and AtomAI models on the same synthetic images, sweeps detection thresholds, ranks the models, and can test the best models on TEM-ImageNet v1.3.

- `training_scripts/sweep_blobnet_pixel_size.py`
  Tests a trained Blob-Net checkpoint across generated datasets rendered at different pixel sizes, plotting accuracy against pixel size and feature size relative to the bottleneck receptive field.

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

Compare Blob-Net with external pretrained atom-localization models:

```bash
source .venv/bin/activate
uv run python training_scripts/compare_pretrained_models.py \
  --output-dir /tmp/blobnet_pretrained_compare \
  --download-models \
  --download-tem-imagenet \
  --tem-imagenet-max-samples 128 \
  --device auto
```

AtomAI is optional and must be installed separately for its pretrained models to run. Without `--download-models`, the comparison uses cached external weights if present and reports skipped models in `model_status.json`. The ImageNet option targets the AtomSegNet paper's TEM-ImageNet v1.3 dataset, available from `xinhuolin/TEM-ImageNet-v1.3`, rather than the natural-image ILSVRC ImageNet dataset.

Sweep Blob-Net across pixel sizes around the training pixel size:

```bash
source .venv/bin/activate
uv run python training_scripts/sweep_blobnet_pixel_size.py \
  --output-dir /tmp/blobnet_pixel_size_sweep \
  --samples-per-size 64 \
  --dataset random \
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

A pretrained-model comparison run writes:
- `synthetic_metrics_by_threshold.json/csv`
- `synthetic_best_metrics.json/csv` and `model_ranking.json/csv`
- `threshold_ranking.json/csv`
- `model_status.json/csv`
- `synthetic_generalization_summary.png` and `synthetic_prediction_gallery.png`
- `tem_imagenet_metrics.json/csv` when TEM-ImageNet is available or downloaded

A pixel-size sweep writes:
- `pixel_size_metrics.json/csv`
- `pixel_size_sweep_config.json`
- `pixel_size_accuracy_vs_feature_rf.png`
- `pixel_size_target_output_examples.png`

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
  compare_pretrained_models.py
  sweep_blobnet_pixel_size.py
  generate_dataset.py
  export_examples.py
  export_low_count_demo.py
  analyze_real_image.py
  check_mps.py
```
