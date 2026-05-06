from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GombNet.synthetic import (  # noqa: E402
    RandomMicroscopeImageConfig,
    render_microscope_image,
    sample_atom_coordinates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a total-count sweep for visually testing low-signal regimes."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("examples/low_count_demo"))
    count_input = parser.add_mutually_exclusive_group()
    count_input.add_argument(
        "--total-counts",
        nargs="+",
        type=float,
        default=[0, 10, 25, 50, 100, 250, 1000, 5000],
    )
    count_input.add_argument("--counts-per-pixel", nargs="+", type=float, default=None)
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=130)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--min-atoms", type=int, default=120)
    parser.add_argument("--max-atoms", type=int, default=180)
    parser.add_argument("--min-separation", type=float, default=10.0)
    parser.add_argument("--min-separation-range-min", type=float, default=9.0)
    parser.add_argument("--min-separation-range-max", type=float, default=12.0)
    parser.add_argument("--sigma-min", type=float, default=1.8)
    parser.add_argument("--sigma-max", type=float, default=3.2)
    parser.add_argument("--background-min", type=float, default=0.0)
    parser.add_argument("--background-max", type=float, default=0.30)
    parser.add_argument("--inhom-background-min", type=float, default=0.0)
    parser.add_argument("--inhom-background-max", type=float, default=0.0)
    parser.add_argument("--low-freq-noise-min", type=float, default=0.08)
    parser.add_argument("--low-freq-noise-max", type=float, default=0.30)
    parser.add_argument("--blur-sigma-min", type=float, default=0.3)
    parser.add_argument("--blur-sigma-max", type=float, default=1.1)
    parser.add_argument("--edge-padding", type=int, default=0)
    parser.add_argument("--gallery-panel-size", type=int, default=256)
    return parser.parse_args()


def to_uint8(image: np.ndarray, display_scale: float = 1.0) -> np.ndarray:
    scaled = np.asarray(image, dtype=np.float32) / max(float(display_scale), 1e-6)
    return np.clip(np.round(scaled * 255.0), 0.0, 255.0).astype(np.uint8)


def save_gray(image: np.ndarray, path: Path, display_scale: float = 1.0) -> None:
    Image.fromarray(to_uint8(image, display_scale=display_scale), mode="L").save(path)


def overlay_points(
    image: np.ndarray,
    coordinates: np.ndarray,
    radius: float = 2.0,
    display_scale: float = 1.0,
) -> Image.Image:
    rgb = np.stack([image, image, image], axis=-1)
    canvas = Image.fromarray(to_uint8(rgb, display_scale=display_scale), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    for y, x in np.asarray(coordinates, dtype=np.float32):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(0, 255, 255), width=1)
    return canvas


def build_config(args: argparse.Namespace) -> RandomMicroscopeImageConfig:
    return RandomMicroscopeImageConfig(
        image_shape=(args.height, args.width),
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        min_separation=args.min_separation,
        min_separation_range=(args.min_separation_range_min, args.min_separation_range_max),
        sigma_range=(args.sigma_min, args.sigma_max),
        background_range=(args.background_min, args.background_max),
        inhomogeneous_background_range=(
            args.inhom_background_min,
            args.inhom_background_max,
        ),
        low_frequency_noise_range=(args.low_freq_noise_min, args.low_freq_noise_max),
        read_noise_std_range=(0.0, 0.0),
        total_counts_range=(
            (min(args.total_counts), max(args.total_counts))
            if args.counts_per_pixel is None
            else None
        ),
        counts_per_pixel_range=(
            (min(args.counts_per_pixel), max(args.counts_per_pixel))
            if args.counts_per_pixel is not None
            else None
        ),
        blur_sigma_range=(args.blur_sigma_min, args.blur_sigma_max),
        edge_padding=args.edge_padding,
    )


def make_scene(config: RandomMicroscopeImageConfig, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sampled_min_separation = (
        float(rng.uniform(*config.min_separation_range))
        if config.min_separation_range is not None
        else float(config.min_separation)
    )
    atom_count = int(rng.integers(config.min_atoms, config.max_atoms + 1))
    coordinates = sample_atom_coordinates(
        config,
        rng=rng,
        atom_count=atom_count,
        min_separation=sampled_min_separation,
    )
    intensities = rng.uniform(config.intensity_range[0], config.intensity_range[1], size=len(coordinates)).astype(np.float32)
    sigmas = rng.uniform(config.sigma_range[0], config.sigma_range[1], size=len(coordinates)).astype(np.float32)
    return {
        "coordinates": coordinates,
        "intensities": intensities,
        "sigmas": sigmas,
        "sampled_min_separation": sampled_min_separation,
    }


def render_count_sweep(
    config: RandomMicroscopeImageConfig,
    count_values: Sequence[float],
    counts_per_pixel: bool,
    scene: dict[str, Any],
    scene_index: int,
    seed: int,
) -> list[dict[str, Any]]:
    rendered_images = []
    height, width = config.image_shape
    for count_index, requested_value in enumerate(count_values):
        requested_total_counts = int(round(float(requested_value) * height * width)) if counts_per_pixel else int(round(float(requested_value)))
        count_config = (
            replace(
                config,
                total_counts_range=None,
                counts_per_pixel_range=(float(requested_value), float(requested_value)),
            )
            if counts_per_pixel
            else replace(
                config,
                total_counts_range=(float(requested_value), float(requested_value)),
                counts_per_pixel_range=None,
            )
        )
        image_record = render_microscope_image(
            coordinates=scene["coordinates"],
            config=count_config,
            rng=np.random.default_rng(seed + scene_index * 10_000 + count_index),
            intensities=scene["intensities"],
            sigmas=scene["sigmas"],
            target_coordinates=scene["coordinates"],
            metadata={
                "image_type": "low_count_demo",
                "scene_index": int(scene_index),
                "requested_total_counts": int(requested_total_counts),
                "requested_counts_per_pixel": (
                    float(requested_value) if counts_per_pixel else None
                ),
                "sampled_min_separation": float(scene["sampled_min_separation"]),
                "visible_atom_count": int(len(scene["coordinates"])),
                "rendered_atom_count": int(len(scene["coordinates"])),
            },
        )
        rendered_images.append(image_record)
    return rendered_images


def summarize_image_record(image_record: dict[str, Any]) -> dict[str, Any]:
    image = np.asarray(image_record["image"], dtype=np.float32)
    count_map = np.asarray(image_record.get("count_map", image), dtype=np.float32)
    return {
        "scene_index": int(image_record["scene_index"]),
        "requested_total_counts": int(image_record["requested_total_counts"]),
        "requested_counts_per_pixel": image_record["requested_counts_per_pixel"],
        "total_counts": int(image_record.get("total_counts", np.asarray(image_record["image"]).sum())),
        "count_map_sum": float(count_map.sum()),
        "count_map_nonzero": int((count_map > 0.0).sum()),
        "visible_atom_count": int(image_record["visible_atom_count"]),
        "sampled_min_separation": float(image_record["sampled_min_separation"]),
        "image_mean": float(image.mean()),
        "image_std": float(image.std()),
        "image_min": float(image.min()),
        "image_max": float(image.max()),
        "positive_intensity_sum": float(np.clip(image, 0.0, None).sum()),
        "nonzero_fraction": float((image > 0.0).mean()),
    }


def save_count_gallery(
    images_by_scene: list[list[dict[str, Any]]],
    count_values: Sequence[float],
    counts_per_pixel: bool,
    output_path: Path,
    panel_size: int,
    overlay: bool,
    display_scale: float,
) -> Path:
    font = ImageFont.load_default()
    label_width = 92
    header_height = 46
    footer_height = 18
    margin = 14
    gap = 10
    rows = len(count_values)
    columns = len(images_by_scene)
    width = label_width + margin * 2 + columns * panel_size + max(0, columns - 1) * gap
    height = header_height + margin * 2 + rows * (panel_size + footer_height) + max(0, rows - 1) * gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title = "Low-Count Overlay Sweep" if overlay else "Low-Count Input Sweep"
    draw.text((margin, 8), title, fill="black", font=font)
    for scene_index in range(columns):
        x0 = label_width + margin + scene_index * (panel_size + gap)
        draw.text((x0 + 4, 26), f"scene {scene_index:02d}", fill="black", font=font)

    for count_index, requested_value in enumerate(count_values):
        y0 = header_height + margin + count_index * (panel_size + footer_height + gap)
        label = (
            f"{requested_value:g} c/px"
            if counts_per_pixel
            else f"{requested_value:g} counts"
        )
        draw.text((margin, y0 + panel_size // 2 - 6), label, fill="black", font=font)
        for scene_index, rendered_images in enumerate(images_by_scene):
            x0 = label_width + margin + scene_index * (panel_size + gap)
            image_record = rendered_images[count_index]
            panel = (
                overlay_points(image_record["image"], image_record["coordinates"], display_scale=display_scale)
                if overlay
                else Image.fromarray(to_uint8(image_record["image"], display_scale=display_scale), mode="L").convert("RGB")
            )
            panel = panel.resize((panel_size, panel_size), resample=Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST)
            canvas.paste(panel, (x0, y0))
            draw.rectangle((x0, y0, x0 + panel_size - 1, y0 + panel_size - 1), outline="black", width=1)
            draw.text((x0 + 4, y0 + panel_size + 3), f"atoms={len(image_record['coordinates'])}", fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def write_manifest(rows: list[dict[str, Any]], output_dir: Path) -> None:
    csv_path = output_dir / "manifest.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "manifest.json").write_text(json.dumps(rows, indent=2))


def main() -> int:
    args = parse_args()
    if args.num_scenes < 1:
        raise ValueError("--num-scenes must be at least 1.")
    count_values = (
        [float(counts) for counts in args.counts_per_pixel]
        if args.counts_per_pixel is not None
        else [float(counts) for counts in args.total_counts]
    )
    if any(counts < 0 for counts in count_values):
        raise ValueError("Count values must be non-negative.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args)
    using_counts_per_pixel = args.counts_per_pixel is not None
    images_by_scene: list[list[dict[str, Any]]] = []
    manifest_rows: list[dict[str, Any]] = []

    for scene_index in range(args.num_scenes):
        scene = make_scene(config, args.seed + scene_index)
        scene_dir = args.output_dir / f"scene_{scene_index:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        rendered_images = render_count_sweep(
            config,
            count_values,
            using_counts_per_pixel,
            scene,
            scene_index,
            args.seed + 1_000,
        )
        images_by_scene.append(rendered_images)

    display_scale = 1.0

    for scene_index, rendered_images in enumerate(images_by_scene):
        scene_dir = args.output_dir / f"scene_{scene_index:02d}"
        for image_record in rendered_images:
            count_label = f"{image_record['requested_total_counts']:g}".replace(".", "p")
            stem = f"scene_{scene_index:02d}_counts_{count_label}"
            save_gray(image_record["image"], scene_dir / f"{stem}_image.png", display_scale=display_scale)
            save_gray(image_record["target"], scene_dir / f"{stem}_target.png")
            save_gray(image_record["count_map"], scene_dir / f"{stem}_count_map.png")
            overlay_points(image_record["image"], image_record["coordinates"], display_scale=display_scale).save(scene_dir / f"{stem}_overlay.png")

            row = summarize_image_record(image_record)
            row.update(
                {
                    "image_path": str((scene_dir / f"{stem}_image.png").relative_to(args.output_dir)),
                    "target_path": str((scene_dir / f"{stem}_target.png").relative_to(args.output_dir)),
                    "count_map_path": str((scene_dir / f"{stem}_count_map.png").relative_to(args.output_dir)),
                    "overlay_path": str((scene_dir / f"{stem}_overlay.png").relative_to(args.output_dir)),
                }
            )
            manifest_rows.append(row)

            metadata = {
                key: row[key]
                for key in [
                    "scene_index",
                    "requested_total_counts",
                    "requested_counts_per_pixel",
                    "total_counts",
                    "count_map_sum",
                    "count_map_nonzero",
                    "visible_atom_count",
                    "sampled_min_separation",
                    "image_mean",
                    "image_std",
                    "image_min",
                    "image_max",
                    "positive_intensity_sum",
                    "nonzero_fraction",
                ]
            }
            metadata["display_scale"] = display_scale
            metadata["config"] = image_record["config"]
            (scene_dir / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2))

    write_manifest(manifest_rows, args.output_dir)
    input_gallery = save_count_gallery(
        images_by_scene,
        count_values,
        using_counts_per_pixel,
        args.output_dir / "low_count_input_sweep.png",
        args.gallery_panel_size,
        overlay=False,
        display_scale=display_scale,
    )
    overlay_gallery = save_count_gallery(
        images_by_scene,
        count_values,
        using_counts_per_pixel,
        args.output_dir / "low_count_overlay_sweep.png",
        args.gallery_panel_size,
        overlay=True,
        display_scale=display_scale,
    )
    print(f"Saved low-count demo to {args.output_dir.resolve()}", flush=True)
    print(f"Display scale: {display_scale:g} raw intensity units -> white", flush=True)
    print(f"Input sweep: {input_gallery.resolve()}", flush=True)
    print(f"Overlay sweep: {overlay_gallery.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
