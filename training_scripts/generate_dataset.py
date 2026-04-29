from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GombNet.synthetic import (
    RandomGaussianConfig,
    generate_random_gaussian_sample,
    save_gaussian_dataset,
    save_gaussian_dataset_splits,
    save_gaussian_preview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an edge-aware synthetic Gaussian dataset for the Blob-Net U-Net pipeline."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--min-atoms", type=int, default=550)
    parser.add_argument("--max-atoms", type=int, default=800)
    parser.add_argument("--min-separation", type=float, default=14.5)
    parser.add_argument("--min-separation-range-min", type=float, default=12.5)
    parser.add_argument("--min-separation-range-max", type=float, default=16.7)
    parser.add_argument("--sigma-min", type=float, default=2.8)
    parser.add_argument("--sigma-max", type=float, default=4.3)
    parser.add_argument("--background-min", type=float, default=0.0)
    parser.add_argument("--background-max", type=float, default=0.30)
    parser.add_argument("--low-freq-noise-min", type=float, default=0.08)
    parser.add_argument("--low-freq-noise-max", type=float, default=0.30)
    parser.add_argument("--read-noise-min", type=float, default=0.03)
    parser.add_argument("--read-noise-max", type=float, default=0.14)
    parser.add_argument("--poisson-counts-min", type=float, default=50.0)
    parser.add_argument("--poisson-counts-max", type=float, default=25000.0)
    parser.add_argument("--blur-sigma-min", type=float, default=0.3)
    parser.add_argument("--blur-sigma-max", type=float, default=1.1)
    parser.add_argument("--edge-padding", type=int, default=24)
    parser.add_argument("--preview-path", type=Path, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RandomGaussianConfig:
    return RandomGaussianConfig(
        image_shape=(args.height, args.width),
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        min_separation=args.min_separation,
        min_separation_range=(args.min_separation_range_min, args.min_separation_range_max),
        sigma_range=(args.sigma_min, args.sigma_max),
        background_range=(args.background_min, args.background_max),
        low_frequency_noise_range=(args.low_freq_noise_min, args.low_freq_noise_max),
        read_noise_std_range=(args.read_noise_min, args.read_noise_max),
        poisson_counts_range=(args.poisson_counts_min, args.poisson_counts_max),
        blur_sigma_range=(args.blur_sigma_min, args.blur_sigma_max),
        edge_padding=args.edge_padding,
    )


def main() -> int:
    args = parse_args()
    config = build_config(args)

    split_args = [args.train_samples, args.val_samples, args.test_samples]
    use_explicit_splits = any(value is not None for value in split_args)
    if use_explicit_splits and not all(value is not None for value in split_args):
        raise ValueError("--train-samples, --val-samples, and --test-samples must all be set together.")

    if use_explicit_splits:
        saved = save_gaussian_dataset_splits(
            output_dir=args.output_dir,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            config=config,
            seed=args.seed,
        )
        print(
            f"Saved train={len(saved['train'])}, val={len(saved['val'])}, test={len(saved['test'])} "
            f"samples to {args.output_dir}",
            flush=True,
        )
    else:
        saved = save_gaussian_dataset(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            config=config,
            seed=args.seed,
        )
        print(f"Saved {len(saved)} samples to {args.output_dir}", flush=True)

    if args.preview_path is not None:
        preview = generate_random_gaussian_sample(config)
        saved_path = save_gaussian_preview(preview, args.preview_path)
        print(f"Saved preview image to {saved_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
