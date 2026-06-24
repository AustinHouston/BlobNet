# Contributing to BlobNet

BlobNet is intentionally small. Changes should strengthen the maintained training workflow or add reusable behavior to the `blobnet` package.

## Setup

```bash
uv sync
```

## Before Submitting Changes

Compile the package and confirm that the training CLI loads:

```bash
uv run python -m compileall -q blobnet scripts
uv run blobnet-train --help
```

For changes to training or data generation, run a tiny end-to-end pipeline:

```bash
uv run blobnet-generate-dataset --output-dir /tmp/blobnet_dataset --train-samples 4 --val-samples 2 --test-samples 2 --overwrite
uv run blobnet-train --dataset-dir /tmp/blobnet_dataset --output-dir /tmp/blobnet_training --epochs 1 --batch-size 1 --device cpu
```

## Repository Conventions

- Use `blobnet` for Python imports and `BlobNet` in prose.
- Keep dataset generation in `scripts/generate_training_dataset.py`.
- Keep training orchestration in `scripts/train_unet.py`.
- Put dataset and model parameters in the appropriate YAML file under `configs/`.
- Put reusable code in the `blobnet` package, not in one-off scripts.
- Prefer a configuration option over a nearly identical training script.
- Keep generated data, checkpoints, plots, and local environments out of version control.
- Preserve offset-cloud evaluation when changing localization metrics or visualization.
- Prefer single quotes in new Python code and keep functions only when they remove real duplication or clarify a meaningful operation.
