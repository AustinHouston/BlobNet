# Blob-Net

Blob-Net trains one U-Net for atom localization in atomic-resolution STEM images.

The repo has one supported training workflow:
- generate synthetic microscope-image batches in memory
- train the U-Net on random atom-like peaks
- evaluate the same checkpoint on random, cubic, hexagonal, and ASE-projected structures
- run tiled inference on one real EMD image
- write metrics, plots, galleries, and the best checkpoint

## Environment

```bash
uv sync
```

On Apple Silicon, verify MPS before a long run:

```bash
uv run python training_scripts/check_mps.py
```

## Train

```bash
uv run python training_scripts/train_unet.py \
  --output-dir /tmp/blobnet_run \
  --device auto
```

The same command is also installed as:

```bash
uv run blobnet-train-unet --output-dir /tmp/blobnet_run --device auto
```

Useful smoke-test settings:

```bash
uv run python training_scripts/train_unet.py \
  --output-dir /tmp/blobnet_smoke \
  --epochs 1 \
  --batch-size 1 \
  --train-samples 4 \
  --val-samples 2 \
  --random-test-samples 1 \
  --periodic-test-samples 1 \
  --structure-test-samples 1 \
  --height 96 \
  --width 96 \
  --structure-height 96 \
  --structure-width 96 \
  --num-workers 0 \
  --device auto
```

## Outputs

A run writes:
- `benchmark_metrics.json` and `benchmark_metrics.csv`
- summary plots for random, periodic, and structure tests
- prediction galleries and offset clouds
- `real_image/` inference plots for the configured EMD file
- `unet/unet_best.pth`

## Layout

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
  train_unet.py       main workflow
  check_mps.py        Apple Silicon device check
```
