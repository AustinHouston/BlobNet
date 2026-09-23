# BlobNet Figure Editing And Generation Guide

This note is for an AI agent that has never seen this repository but needs to
edit figure/image-generation code and regenerate manuscript-style outputs. It is
not intended as a training guide.

## Repository Map

- `blobnet/` contains reusable package code: models, synthetic image generation,
  metrics, and visualization helpers.
- `scripts/` contains command-line workflows. Most figure work lives in
  `scripts/make_manuscript_figures.py`.
- `configs/` contains YAML configs for datasets and models.
- `experimental_data/` contains the tracked experimental `.hf5` NSID files used for
  experimental HAADF figures.
- `outputs/` contains generated datasets, checkpoints, measurements, sweeps, and
  figure outputs. Many subdirectories are ignored by git.
- `notebooks/` contains exploratory notebooks. Treat these as useful reference,
  but prefer scripts for reproducible figure generation.

## Setup

Use `uv` from the repository root:

```powershell
uv sync
```

For quick validation after edits:

```powershell
uv run python -m compileall -q blobnet scripts
uv run blobnet-manuscript-figures --help
```

On Windows or Linux with an NVIDIA GPU, the project is configured to use CUDA
PyTorch wheels through `pyproject.toml`. Figure commands accept
`--device auto`, `--device cuda`, and `--device cpu`.

## Main Figure Command

The figure entrypoint is:

```powershell
uv run blobnet-manuscript-figures <subcommand> [options]
```

The console script is declared in `pyproject.toml` and points to
`scripts.make_manuscript_figures:main`.

Useful subcommands:

- `figure1`: synthetic training geometry/generalization figure.
- `figure3`: experimental HAADF input/output figure.
- `figure4`: scale and spacing robustness figure.
- `figure2`: edge-structure TP/FP/FN diagnostics.
- `all`: regenerate the standard manuscript figure set.

Write new outputs to a fresh directory so older outputs are preserved:

```powershell
uv run blobnet-manuscript-figures figure3 --output-dir outputs/manuscript_figures_YYYYMMDD_HHMMSS
```

## Experimental Data Figures

Experimental `.hf5` NSID inputs live in:

```text
experimental_data/
```

Current experimental files:

- `pristine_monolayer_MoS2.hf5`
- `Sigma3_coherent_twin_grain_boundary_FCC_Al.hf5`
- `high_angle_grain_boundary_monolayer_WS2.hf5`
- `Al72Ni11Co17_quasicrystal.hf5`
- `gold_implanted_in_TiO2.hf5`

Feature measurement summaries used for feature-size matching are expected at:

```text
outputs/experimental_feature_measurements_local/experimental_feature_measurements.json
```

If the measurement file is missing or stale, regenerate it with:

```powershell
uv run python -m scripts.measure_experimental_features --data-dir experimental_data --output-dir outputs/experimental_feature_measurements_local
```

The experimental figure code loads the curated HAADF channel and its calibration
from the NSID files using `pyTEMlib.file_tools.open_file`.

## Regenerating The Current Experimental Outputs

Standard Figure 3:

```powershell
uv run blobnet-manuscript-figures figure3 --output-dir outputs/manuscript_figures_YYYYMMDD_HHMMSS
```

## Editing Figure Code

Prefer editing `scripts/make_manuscript_figures.py` for manuscript figure
layout, image preprocessing, plotting, and command-line options.

Helpful existing functions:

- `_normalize_image`: percentile normalization to `[0, 1]`.
- `_load_blobnet_model`: loads a U-Net checkpoint.
- `_predict_array`: runs model inference on one image.
- `_predict_tiled`: tiled inference for larger images.
- `_load_experimental_image`: reads and normalizes an experimental `.hf5` image through pyTEMlib.
- `_make_feature_matched_experimental_view`: DoG background subtraction,
  feature-size matching, and center crop/pad used by Figure 3.
- `_plot_clean_image`: simple image panel plotting.
- `extract_subpixel_peak_positions` from `blobnet.metrics`: local maximum peak
  extraction with subpixel refinement.

When adding a new figure:

1. Add helper functions near related figure code.
2. Add a `make_figure_*` function that accepts `argparse.Namespace`.
3. Add a subparser in `build_parser()`.
4. Save both a `.png` and a `.json` summary when useful.
5. Print the output path by returning it from the figure function.

Keep edits scoped. Do not move reusable package logic into notebooks.

## Verification Checklist

After changing figure or image-generation code:

```powershell
uv run python -m compileall -q blobnet scripts
uv run blobnet-manuscript-figures <changed-subcommand> --help
uv run blobnet-manuscript-figures <changed-subcommand> --output-dir outputs/<fresh-output-dir>
```

Then inspect the generated PNG visually. Check that:

- all expected panels rendered,
- axes/titles/legends do not overlap important image content,
- scatter markers align with image coordinates,
- noisy rows actually show the intended noise level,
- the JSON summary has plausible counts/statistics.

## What Not To Do For Figure-Only Work

- Do not retrain models unless explicitly asked.
- Do not overwrite old output directories unless explicitly asked.
- Do not edit notebooks as the primary source of truth for figures.
- Do not commit large generated outputs unless the user asks for them.
- Do not revert unrelated local changes.
