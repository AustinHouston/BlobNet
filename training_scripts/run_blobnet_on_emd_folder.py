from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any, Dict, List

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GombNet.metrics import extract_subpixel_peak_positions  # noqa: E402
from GombNet.networks import build_unet  # noqa: E402
from GombNet.real_image import (  # noqa: E402
    get_real_image_pixel_size_angstrom,
    load_velox_emd_image,
    predict_heatmap_tiled,
    preprocess_real_image_variants,
    select_informative_crops,
)
from GombNet.utils import resolve_torch_device  # noqa: E402
from GombNet.visualization import (  # noqa: E402
    save_real_crop_gallery,
    save_real_image_overview,
)
from training_scripts.io_utils import write_csv, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained BlobNet checkpoint over every EMD file in a folder and "
            "save per-image visualizations plus a combined contact sheet."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("real_data/ims_from_elizabeth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/elizabeth_emd_blobnet"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/inhom_background_unet_20epoch/unet/unet_best.pth"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--variant", default="flatfield_normalized")
    parser.add_argument("--threshold-rel", type=float, default=0.65)
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--peak-window-size", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-crops", type=int, default=3)
    parser.add_argument("--num-filters", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def slugify(path: Path) -> str:
    stem = path.stem.strip()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "image"


def load_blobnet(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = build_unet(
        input_channels=1,
        num_classes=1,
        num_filters=args.num_filters,
        dropout=args.dropout,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def save_contact_sheet(
    output_path: Path,
    rows: List[Dict[str, Any]],
    processed_images: Dict[str, np.ndarray],
    heatmaps: Dict[str, np.ndarray],
    coordinates: Dict[str, np.ndarray],
) -> Path:
    if not rows:
        raise ValueError("No rows supplied for contact sheet.")

    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(13.5, 4.0 * len(rows)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row_index, row in enumerate(rows):
        slug = str(row["slug"])
        image = processed_images[slug]
        heatmap = heatmaps[slug]
        coords = coordinates[slug]
        title = (
            f"{row['file_name']}\n"
            f"{row['shape_y']}x{row['shape_x']}, "
            f"{row['pixel_size_angstrom']:.4f} A/px, "
            f"detections={row['detections']}"
            if row.get("pixel_size_angstrom") is not None
            else f"{row['file_name']}\n{row['shape_y']}x{row['shape_x']}, detections={row['detections']}"
        )

        axes[row_index, 0].imshow(image, cmap="gray")
        axes[row_index, 0].set_title(title, fontsize=10)
        axes[row_index, 1].imshow(heatmap, cmap="magma")
        axes[row_index, 1].set_title("BlobNet heatmap", fontsize=10)
        axes[row_index, 2].imshow(image, cmap="gray")
        if len(coords):
            axes[row_index, 2].scatter(
                coords[:, 1],
                coords[:, 0],
                s=5,
                c="cyan",
                alpha=0.55,
                linewidths=0,
            )
        axes[row_index, 2].set_title("Overlay", fontsize=10)
        for axis in axes[row_index]:
            axis.axis("off")

    fig.suptitle("BlobNet on Elizabeth EMD Folder", fontsize=18, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_one_file(
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    slug = slugify(path)
    image_output_dir = args.output_dir / slug
    image_output_dir.mkdir(parents=True, exist_ok=True)

    raw_image, metadata = load_velox_emd_image(path)
    pixel_size_angstrom = get_real_image_pixel_size_angstrom(metadata)
    variants = preprocess_real_image_variants(raw_image)
    if args.variant not in variants:
        raise ValueError(
            f"Variant '{args.variant}' is not available. Choices: {sorted(variants)}"
        )
    processed = variants[args.variant]
    heatmap = predict_heatmap_tiled(
        model=model,
        image=processed,
        device=device,
        tile_size=args.tile_size,
        overlap=args.tile_overlap,
        batch_size=args.batch_size,
    )
    coords = extract_subpixel_peak_positions(
        heatmap,
        threshold_rel=args.threshold_rel,
        min_distance=args.min_distance,
        window_size=args.peak_window_size,
    )
    crop_boxes = select_informative_crops(
        processed,
        crop_size=args.crop_size,
        num_crops=args.num_crops,
    )

    overview_path = save_real_image_overview(
        output_path=image_output_dir / "blobnet_overview.png",
        image_variants={args.variant: processed},
        heatmaps={args.variant: heatmap},
        coordinates={args.variant: coords},
        pixel_size_angstrom=pixel_size_angstrom,
        threshold_rel=args.threshold_rel,
    )
    crops_path = save_real_crop_gallery(
        output_path=image_output_dir / "blobnet_crops.png",
        variant_name=args.variant,
        image=processed,
        heatmap=heatmap,
        coordinates=coords,
        crop_boxes=crop_boxes,
    )
    np.savez_compressed(
        image_output_dir / "blobnet_outputs.npz",
        processed_image=processed.astype(np.float32),
        heatmap=heatmap.astype(np.float32),
        coordinates=coords.astype(np.float32),
        crop_boxes=np.asarray(crop_boxes, dtype=np.int32),
    )

    summary = {
        "file_name": path.name,
        "path": path,
        "slug": slug,
        "shape_y": int(raw_image.shape[0]),
        "shape_x": int(raw_image.shape[1]),
        "pixel_size_angstrom": pixel_size_angstrom,
        "variant": args.variant,
        "threshold_rel": float(args.threshold_rel),
        "detections": int(len(coords)),
        "heatmap_mean": float(np.mean(heatmap)),
        "heatmap_max": float(np.max(heatmap)),
        "processed_mean": float(np.mean(processed)),
        "processed_std": float(np.std(processed)),
        "overview_path": overview_path,
        "crops_path": crops_path,
        "npz_path": image_output_dir / "blobnet_outputs.npz",
        "crop_boxes_yxyx": crop_boxes,
    }
    write_json(summary, image_output_dir / "summary.json")
    return summary, processed, heatmap, coords


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    emd_paths = sorted(args.input_dir.glob("*.emd"))
    if args.max_files is not None:
        emd_paths = emd_paths[: int(args.max_files)]
    if not emd_paths:
        raise FileNotFoundError(f"No .emd files found in {args.input_dir}")

    device = resolve_torch_device(args.device, verbose=True)
    model = load_blobnet(args, device)
    print(
        f"Running BlobNet on {len(emd_paths)} EMD file(s) from {args.input_dir}",
        flush=True,
    )

    summaries: List[Dict[str, Any]] = []
    processed_images: Dict[str, np.ndarray] = {}
    heatmaps: Dict[str, np.ndarray] = {}
    coordinates: Dict[str, np.ndarray] = {}
    failures: List[Dict[str, str]] = []
    for index, path in enumerate(emd_paths, start=1):
        print(f"[{index}/{len(emd_paths)}] {path.name}", flush=True)
        try:
            summary, processed, heatmap, coords = run_one_file(path, model, device, args)
        except Exception as exc:  # noqa: BLE001 - keep batch processing useful.
            failures.append({"file_name": path.name, "path": str(path), "error": str(exc)})
            print(f"  failed: {exc}", flush=True)
            continue
        summaries.append(summary)
        processed_images[str(summary["slug"])] = processed
        heatmaps[str(summary["slug"])] = heatmap
        coordinates[str(summary["slug"])] = coords
        print(
            f"  detections={summary['detections']}, "
            f"pixel_size={summary['pixel_size_angstrom']}",
            flush=True,
        )

    write_csv(summaries, args.output_dir / "blobnet_emd_summary.csv")
    write_json(
        {
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "summaries": summaries,
            "failures": failures,
        },
        args.output_dir / "blobnet_emd_summary.json",
    )
    if summaries:
        contact_path = save_contact_sheet(
            args.output_dir / "blobnet_emd_contact_sheet.png",
            summaries,
            processed_images,
            heatmaps,
            coordinates,
        )
        print(f"Saved contact sheet to {contact_path}", flush=True)
    if failures:
        print(f"Finished with {len(failures)} failure(s). See blobnet_emd_summary.json.", flush=True)
    else:
        print(f"Finished all files. Outputs saved to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
