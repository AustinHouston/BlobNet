from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GombNet.real_image import (
    get_real_image_pixel_size_angstrom,
    load_velox_emd_image,
    preprocess_real_image_variants,
    select_informative_crops,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect atom-like blobs in a real EMD image with a standard LoG blob finder, "
            "fit local Gaussians, and summarize sigma/spacing distributions."
        )
    )
    parser.add_argument("--input-path", type=Path, default=Path("real_data/WS2.emd"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/blobnet_real_image_analysis"))
    parser.add_argument(
        "--variant",
        type=str,
        default="flatfield_normalized",
        choices=["raw_percentile", "flatfield_normalized", "background_subtracted"],
    )
    parser.add_argument("--min-sigma", type=float, default=1.0)
    parser.add_argument("--max-sigma", type=float, default=5.0)
    parser.add_argument("--num-sigma", type=int, default=17)
    parser.add_argument("--threshold-rel", type=float, default=0.18)
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--exclude-border", type=int, default=8)
    parser.add_argument("--max-fit-blobs", type=int, default=2500)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-crops", type=int, default=3)
    return parser.parse_args()


def detect_log_blobs(
    image: np.ndarray,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    threshold_rel: float,
    min_distance: int,
    exclude_border: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sigmas = np.linspace(min_sigma, max_sigma, int(num_sigma), dtype=np.float32)
    responses = []
    for sigma in sigmas:
        response = -gaussian_laplace(image, sigma=float(sigma), mode="reflect") * (float(sigma) ** 2)
        responses.append(response.astype(np.float32))
    response_stack = np.stack(responses, axis=0)

    threshold_abs = float(response_stack.max()) * float(threshold_rel)
    local_max = response_stack == maximum_filter(
        response_stack,
        size=(3, 2 * int(min_distance) + 1, 2 * int(min_distance) + 1),
        mode="nearest",
    )
    candidate_mask = local_max & (response_stack >= threshold_abs)

    if exclude_border > 0:
        candidate_mask[:, :exclude_border, :] = False
        candidate_mask[:, -exclude_border:, :] = False
        candidate_mask[:, :, :exclude_border] = False
        candidate_mask[:, :, -exclude_border:] = False

    scale_idx, ys, xs = np.nonzero(candidate_mask)
    if len(ys) == 0:
        return np.zeros((0, 4), dtype=np.float32), sigmas

    scores = response_stack[scale_idx, ys, xs]
    order = np.argsort(scores)[::-1]

    blobs: List[np.ndarray] = []
    for index in order.tolist():
        y = float(ys[index])
        x = float(xs[index])
        sigma = float(sigmas[scale_idx[index]])
        score = float(scores[index])
        keep = True
        for blob in blobs:
            if math.hypot(y - float(blob[0]), x - float(blob[1])) < max(float(min_distance), 0.6 * (sigma + float(blob[2]))):
                keep = False
                break
        if keep:
            blobs.append(np.array([y, x, sigma, score], dtype=np.float32))

    return (np.stack(blobs, axis=0) if blobs else np.zeros((0, 4), dtype=np.float32)), sigmas


def gaussian_2d(
    xy: Tuple[np.ndarray, np.ndarray],
    amplitude: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    offset: float,
) -> np.ndarray:
    x, y = xy
    return (
        amplitude
        * np.exp(-(((x - x0) ** 2) / (2.0 * sigma_x**2) + ((y - y0) ** 2) / (2.0 * sigma_y**2)))
        + offset
    ).ravel()


def fit_blob_gaussian(image: np.ndarray, y: float, x: float, sigma_init: float) -> Dict[str, float] | None:
    radius = max(6, int(math.ceil(4.0 * float(sigma_init))))
    y_center = int(round(y))
    x_center = int(round(x))
    y0 = max(0, y_center - radius)
    y1 = min(image.shape[0], y_center + radius + 1)
    x0 = max(0, x_center - radius)
    x1 = min(image.shape[1], x_center + radius + 1)
    if (y1 - y0) < 7 or (x1 - x0) < 7:
        return None

    patch = image[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.meshgrid(
        np.arange(y0, y1, dtype=np.float32),
        np.arange(x0, x1, dtype=np.float32),
        indexing="ij",
    )
    amplitude0 = float(max(patch.max() - patch.min(), 1e-3))
    offset0 = float(patch.min())
    p0 = [amplitude0, float(x), float(y), float(sigma_init), float(sigma_init), offset0]
    lower = [0.0, float(x0), float(y0), 0.5, 0.5, -0.5]
    upper = [2.0, float(x1), float(y1), 8.0, 8.0, 1.5]

    try:
        popt, _ = curve_fit(gaussian_2d, (xx, yy), patch.ravel(), p0=p0, bounds=(lower, upper), maxfev=5000)
    except Exception:
        return None

    amplitude, x_fit, y_fit, sigma_x, sigma_y, offset = [float(value) for value in popt]
    model = gaussian_2d((xx, yy), *popt).reshape(patch.shape)
    sigma_eq = float(np.sqrt(max(sigma_x, 1e-6) * max(sigma_y, 1e-6)))
    return {
        "y": y_fit,
        "x": x_fit,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "sigma_eq": sigma_eq,
        "amplitude": amplitude,
        "offset": offset,
        "rmse": float(np.sqrt(np.mean((model - patch) ** 2))),
    }


def summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else float("nan"),
        "median": float(np.median(values)) if len(values) else float("nan"),
        "std": float(np.std(values)) if len(values) else float("nan"),
        "p05": float(np.percentile(values, 5)) if len(values) else float("nan"),
        "p25": float(np.percentile(values, 25)) if len(values) else float("nan"),
        "p75": float(np.percentile(values, 75)) if len(values) else float("nan"),
        "p95": float(np.percentile(values, 95)) if len(values) else float("nan"),
    }


def save_overview(
    output_path: Path,
    raw_image: np.ndarray,
    processed_image: np.ndarray,
    fitted_coords: np.ndarray,
    crop_boxes: List[Tuple[int, int, int, int]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    axes[0, 0].imshow(raw_image, cmap="gray")
    axes[0, 0].set_title("Raw image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(processed_image, cmap="gray")
    if len(fitted_coords):
        axes[0, 1].scatter(fitted_coords[:, 1], fitted_coords[:, 0], s=6, c="cyan", linewidths=0)
    axes[0, 1].set_title("Processed image + fitted centers")
    axes[0, 1].axis("off")

    for ax_index, box in enumerate(crop_boxes[:2], start=2):
        y0, y1, x0, x1 = box
        ax = axes.flat[ax_index]
        ax.imshow(processed_image[y0:y1, x0:x1], cmap="gray")
        if len(fitted_coords):
            mask = (
                (fitted_coords[:, 0] >= y0)
                & (fitted_coords[:, 0] < y1)
                & (fitted_coords[:, 1] >= x0)
                & (fitted_coords[:, 1] < x1)
            )
            coords = fitted_coords[mask]
            if len(coords):
                ax.scatter(coords[:, 1] - x0, coords[:, 0] - y0, s=12, c="lime", linewidths=0)
        ax.set_title(f"Crop {ax_index - 1}")
        ax.axis("off")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_histograms(
    output_path: Path,
    sigma_eq: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    nn_spacing_px: np.ndarray,
    pixel_size_angstrom: float | None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes[0, 0].hist(sigma_eq, bins=40, color="tab:blue", alpha=0.85)
    axes[0, 0].set_title("Equivalent fitted sigma")
    axes[0, 0].set_xlabel("pixels")

    axes[0, 1].hist(sigma_x, bins=40, alpha=0.65, label="sigma_x")
    axes[0, 1].hist(sigma_y, bins=40, alpha=0.65, label="sigma_y")
    axes[0, 1].set_title("Axis-wise fitted sigma")
    axes[0, 1].set_xlabel("pixels")
    axes[0, 1].legend()

    axes[1, 0].hist(nn_spacing_px, bins=50, color="tab:green", alpha=0.85)
    axes[1, 0].set_title("Nearest-neighbor spacing")
    axes[1, 0].set_xlabel("pixels")

    if pixel_size_angstrom is not None:
        axes[1, 1].hist(nn_spacing_px * float(pixel_size_angstrom), bins=50, color="tab:orange", alpha=0.85)
        axes[1, 1].set_title("Nearest-neighbor spacing")
        axes[1, 1].set_xlabel("angstrom")
    else:
        axes[1, 1].axis("off")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_image, metadata = load_velox_emd_image(args.input_path)
    pixel_size_angstrom = get_real_image_pixel_size_angstrom(metadata)
    image = preprocess_real_image_variants(raw_image)[args.variant]

    blobs, sigma_grid = detect_log_blobs(
        image=image,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        num_sigma=args.num_sigma,
        threshold_rel=args.threshold_rel,
        min_distance=args.min_distance,
        exclude_border=args.exclude_border,
    )
    print(
        f"Detected {len(blobs)} LoG blobs in variant={args.variant} "
        f"with sigma grid {sigma_grid[0]:.2f}-{sigma_grid[-1]:.2f}",
        flush=True,
    )

    fits: List[Dict[str, float]] = []
    for index, blob in enumerate(blobs[: args.max_fit_blobs], start=1):
        fit = fit_blob_gaussian(image, float(blob[0]), float(blob[1]), float(blob[2]))
        if fit is not None:
            fits.append(fit)
        if index % 500 == 0:
            print(f"Fitted {index} / {min(len(blobs), args.max_fit_blobs)} blobs", flush=True)

    if not fits:
        raise RuntimeError("No Gaussian fits succeeded.")

    fitted_coords = np.array([[fit["y"], fit["x"]] for fit in fits], dtype=np.float32)
    sigma_x = np.array([fit["sigma_x"] for fit in fits], dtype=np.float32)
    sigma_y = np.array([fit["sigma_y"] for fit in fits], dtype=np.float32)
    sigma_eq = np.array([fit["sigma_eq"] for fit in fits], dtype=np.float32)
    rmse = np.array([fit["rmse"] for fit in fits], dtype=np.float32)
    nn_spacing_px = cKDTree(fitted_coords[:, ::-1]).query(fitted_coords[:, ::-1], k=2)[0][:, 1].astype(np.float32)
    crop_boxes = select_informative_crops(image, crop_size=args.crop_size, num_crops=args.num_crops)

    save_overview(args.output_dir / "real_image_overview.png", raw_image, image, fitted_coords, crop_boxes)
    save_histograms(args.output_dir / "real_image_histograms.png", sigma_eq, sigma_x, sigma_y, nn_spacing_px, pixel_size_angstrom)

    summary = {
        "input_path": str(args.input_path),
        "variant": args.variant,
        "pixel_size_angstrom": pixel_size_angstrom,
        "blob_detector": {
            "method": "scale_normalized_log",
            "min_sigma": args.min_sigma,
            "max_sigma": args.max_sigma,
            "num_sigma": args.num_sigma,
            "threshold_rel": args.threshold_rel,
            "min_distance": args.min_distance,
            "exclude_border": args.exclude_border,
        },
        "blob_count": int(len(blobs)),
        "fit_count": int(len(fits)),
        "fit_rmse": summarize(rmse),
        "sigma_x_pixels": summarize(sigma_x),
        "sigma_y_pixels": summarize(sigma_y),
        "sigma_equivalent_pixels": summarize(sigma_eq),
        "nearest_neighbor_spacing_pixels": summarize(nn_spacing_px),
        "nearest_neighbor_spacing_angstrom": summarize(nn_spacing_px * float(pixel_size_angstrom)) if pixel_size_angstrom is not None else None,
        "recommended_training_ranges": {
            "sigma_pixels": [float(np.percentile(sigma_eq, 5)), float(np.percentile(sigma_eq, 95))],
            "spacing_pixels": [float(np.percentile(nn_spacing_px, 5)), float(np.percentile(nn_spacing_px, 95))],
        },
        "crop_boxes_yxyx": crop_boxes,
    }
    (args.output_dir / "real_image_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.output_dir / "real_image_fits.npz",
        blobs=blobs,
        fitted_coords=fitted_coords,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        sigma_eq=sigma_eq,
        rmse=rmse,
        nn_spacing_px=nn_spacing_px,
    )

    print(f"Saved analysis to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
