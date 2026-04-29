from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GombNet.synthetic import RandomGaussianConfig, generate_random_gaussian_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a few PNG examples of the current synthetic training distribution."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("examples/training_examples"))
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=90)
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
    return parser.parse_args()


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(image, dtype=np.float32) * 255.0), 0.0, 255.0).astype(np.uint8)


def save_overlay(image: np.ndarray, coordinates: np.ndarray, path: Path) -> None:
    rgb = np.stack([image, image, image], axis=-1)
    canvas = Image.fromarray(to_uint8(rgb), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    for y, x in np.asarray(coordinates, dtype=np.float32):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=(0, 255, 255), width=1)
    canvas.save(path)


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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(args.num_examples):
        sample = generate_random_gaussian_sample(config, rng=np.random.default_rng(args.seed + index))
        stem = f"training_example_{index:02d}"
        Image.fromarray(to_uint8(sample["image"]), mode="L").save(args.output_dir / f"{stem}_image.png")
        Image.fromarray(to_uint8(sample["target"]), mode="L").save(args.output_dir / f"{stem}_label.png")
        save_overlay(sample["image"], sample["coordinates"], args.output_dir / f"{stem}_overlay.png")
        metadata = {
            "visible_atom_count": int(sample.get("visible_atom_count", len(sample["coordinates"]))),
            "rendered_atom_count": int(sample.get("rendered_atom_count", len(sample["coordinates"]))),
            "sampled_min_separation": float(sample["sampled_min_separation"]),
            "sigma_range": list(config.sigma_range),
            "config": sample["config"],
        }
        (args.output_dir / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Saved examples to {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
