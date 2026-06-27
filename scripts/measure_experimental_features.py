from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

_CACHE_DIR = Path(tempfile.gettempdir()) / 'blobnet-mpl-cache'
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_CACHE_DIR))
os.environ.setdefault('XDG_CACHE_HOME', str(_CACHE_DIR))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _normalize_image(image: np.ndarray, low: float = 1.0, high: float = 99.8) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [low, high])
    return np.clip((image - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0).astype(np.float32)


def _find_haadf_with_pytemlib(path: Path) -> np.ndarray | None:
    try:
        import pyTEMlib.file_tools as ft
    except ImportError:
        return None

    dataset = ft.open_file(str(path))
    for key in dataset.keys():
        candidate = dataset[key]
        if getattr(candidate, 'title', '') == 'HAADF':
            return np.asarray(candidate, dtype=np.float32)
    return None


def _find_haadf_with_h5py(path: Path) -> np.ndarray:
    arrays: list[tuple[int, str, np.ndarray]] = []
    with h5py.File(path, 'r') as handle:
        def visit(name: str, obj: Any) -> None:
            if not hasattr(obj, 'shape') or name.endswith('/Metadata'):
                return
            if len(obj.shape) < 2 or not np.issubdtype(obj.dtype, np.number):
                return
            data = np.squeeze(np.asarray(obj))
            if data.ndim == 2:
                arrays.append((data.size, name, data.astype(np.float32)))

        handle.visititems(visit)

    if not arrays:
        raise ValueError(f'No 2D numeric image dataset found in {path}')
    return max(arrays, key=lambda item: item[0])[2]


def load_experimental_image(path: Path) -> np.ndarray:
    image = _find_haadf_with_pytemlib(path)
    if image is None:
        image = _find_haadf_with_h5py(path)
    return _normalize_image(np.squeeze(image))


def detect_blobs(
    image: np.ndarray,
    *,
    dog_small: float,
    dog_large: float,
    min_distance: int,
    threshold_percentile: float,
    max_peaks: int,
) -> tuple[np.ndarray, np.ndarray]:
    smoothed = gaussian_filter(image, dog_small)
    background = gaussian_filter(image, dog_large)
    highpass = smoothed - background
    local_variance = gaussian_filter(highpass**2, max(dog_large, 1.0))
    enhanced_z = highpass / np.sqrt(np.maximum(local_variance, 1e-8))
    enhanced = _normalize_image(enhanced_z, low=0.5, high=99.7)
    local_max = enhanced == maximum_filter(enhanced, size=2 * int(min_distance) + 1, mode='nearest')
    threshold = float(np.percentile(enhanced, threshold_percentile))
    border = max(int(min_distance) * 2, 8)
    mask = np.zeros_like(local_max, dtype=bool)
    mask[border : image.shape[0] - border, border : image.shape[1] - border] = True
    candidates = np.argwhere(local_max & mask & (enhanced >= threshold))
    if len(candidates) == 0:
        return np.zeros((0, 2), dtype=np.float32), enhanced
    values = enhanced[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(values)[::-1][: int(max_peaks)]
    return candidates[order].astype(np.float32), enhanced


def estimate_blob_sigma(
    image: np.ndarray,
    center_yx: np.ndarray,
    *,
    profile_radius: int,
    annulus_inner: int,
    annulus_outer: int,
) -> float | None:
    center_y, center_x = float(center_yx[0]), float(center_yx[1])
    y0 = max(0, int(round(center_y)) - annulus_outer)
    y1 = min(image.shape[0], int(round(center_y)) + annulus_outer + 1)
    x0 = max(0, int(round(center_x)) - annulus_outer)
    x1 = min(image.shape[1], int(round(center_x)) + annulus_outer + 1)
    if (y1 - y0) < 2 * annulus_outer or (x1 - x0) < 2 * annulus_outer:
        return None

    yy, xx = np.meshgrid(
        np.arange(y0, y1, dtype=np.float32),
        np.arange(x0, x1, dtype=np.float32),
        indexing='ij',
    )
    rr = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    patch = image[y0:y1, x0:x1].astype(np.float32)
    annulus = (rr >= float(annulus_inner)) & (rr <= float(annulus_outer))
    if not np.any(annulus):
        return None
    background = float(np.median(patch[annulus]))
    signal = np.clip(patch - background, 0.0, None)
    core = rr <= float(profile_radius)
    total = float(signal[core].sum())
    if total <= 1e-6:
        return None
    centroid_y = float((yy[core] * signal[core]).sum() / total)
    centroid_x = float((xx[core] * signal[core]).sum() / total)
    rr2 = (yy - centroid_y) ** 2 + (xx - centroid_x) ** 2
    sigma = float(np.sqrt((signal[core] * rr2[core]).sum() / max(2.0 * total, 1e-6)))
    if not np.isfinite(sigma):
        return None
    return sigma


def robust_range(values: np.ndarray, low: float, high: float) -> list[float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [float('nan'), float('nan')]
    return [float(np.percentile(values, low)), float(np.percentile(values, high))]


def summarize_image(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    image = load_experimental_image(path)
    peaks, enhanced = detect_blobs(
        image,
        dog_small=args.dog_small,
        dog_large=args.dog_large,
        min_distance=args.min_distance,
        threshold_percentile=args.threshold_percentile,
        max_peaks=args.max_peaks,
    )
    sigmas = []
    for peak in peaks:
        sigma = estimate_blob_sigma(
            image,
            peak,
            profile_radius=args.profile_radius,
            annulus_inner=args.annulus_inner,
            annulus_outer=args.annulus_outer,
        )
        if sigma is not None and args.min_sigma <= sigma <= args.max_sigma:
            sigmas.append(sigma)
    sigmas_array = np.asarray(sigmas, dtype=np.float32)

    if len(peaks) >= 2:
        nearest = cKDTree(peaks).query(peaks, k=2)[0][:, 1].astype(np.float32)
        nearest = nearest[(nearest >= args.min_spacing) & (nearest <= args.max_spacing)]
    else:
        nearest = np.zeros((0,), dtype=np.float32)

    summary = {
        'image': path.stem,
        'path': str(path),
        'shape_y': int(image.shape[0]),
        'shape_x': int(image.shape[1]),
        'detected_peaks': int(len(peaks)),
        'profiled_peaks': int(len(sigmas_array)),
        'sigma_px_median': float(np.median(sigmas_array)) if len(sigmas_array) else float('nan'),
        'sigma_px_mean': float(np.mean(sigmas_array)) if len(sigmas_array) else float('nan'),
        'sigma_px_p10': robust_range(sigmas_array, 10, 90)[0],
        'sigma_px_p90': robust_range(sigmas_array, 10, 90)[1],
        'sigma_px_p20': robust_range(sigmas_array, 20, 80)[0],
        'sigma_px_p80': robust_range(sigmas_array, 20, 80)[1],
        'fwhm_px_median': float(2.355 * np.median(sigmas_array)) if len(sigmas_array) else float('nan'),
        'spacing_px_median': float(np.median(nearest)) if len(nearest) else float('nan'),
        'spacing_px_mean': float(np.mean(nearest)) if len(nearest) else float('nan'),
        'spacing_px_p10': robust_range(nearest, 10, 90)[0],
        'spacing_px_p90': robust_range(nearest, 10, 90)[1],
        'spacing_px_p20': robust_range(nearest, 20, 80)[0],
        'spacing_px_p80': robust_range(nearest, 20, 80)[1],
    }
    return summary | {'_image_array': image, '_enhanced_array': enhanced, '_peaks': peaks, '_sigmas': sigmas_array, '_spacing': nearest}


def write_diagnostic(summary: dict[str, Any], output_dir: Path) -> None:
    image = summary['_image_array']
    enhanced = summary['_enhanced_array']
    peaks = summary['_peaks']
    sigmas = summary['_sigmas']
    spacing = summary['_spacing']
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), constrained_layout=True)
    axes[0].imshow(image, cmap='gray', vmin=0.0, vmax=1.0)
    axes[0].scatter(peaks[:, 1], peaks[:, 0], s=4, c='cyan', alpha=0.55, linewidths=0)
    axes[0].set_title(f"{summary['image']} detections")
    axes[1].imshow(enhanced, cmap='magma', vmin=0.0, vmax=1.0)
    axes[1].set_title('DoG enhanced')
    axes[2].hist(sigmas, bins=48, color='#2f8f4e')
    axes[2].axvline(summary['sigma_px_median'], color='black', linestyle='--')
    axes[2].set_title('Profile sigma (px)')
    axes[3].hist(spacing, bins=48, color='#426aa8')
    axes[3].axvline(summary['spacing_px_median'], color='black', linestyle='--')
    axes[3].set_title('Nearest spacing (px)')
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(output_dir / f"{summary['image']}_feature_measurements.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description='Measure blob feature sizes and spacing in experimental HAADF EMD images.')
    parser.add_argument('--data-dir', type=Path, default=Path('experimental_data'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/experimental_feature_measurements'))
    parser.add_argument('--dog-small', type=float, default=0.8)
    parser.add_argument('--dog-large', type=float, default=7.0)
    parser.add_argument('--min-distance', type=int, default=4)
    parser.add_argument('--threshold-percentile', type=float, default=98.7)
    parser.add_argument('--max-peaks', type=int, default=5000)
    parser.add_argument('--profile-radius', type=int, default=7)
    parser.add_argument('--annulus-inner', type=int, default=8)
    parser.add_argument('--annulus-outer', type=int, default=12)
    parser.add_argument('--min-sigma', type=float, default=0.45)
    parser.add_argument('--max-sigma', type=float, default=4.5)
    parser.add_argument('--min-spacing', type=float, default=3.0)
    parser.add_argument('--max-spacing', type=float, default=40.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in sorted(args.data_dir.glob('*.emd')):
        summary = summarize_image(path, args)
        write_diagnostic(summary, args.output_dir)
        clean = {key: value for key, value in summary.items() if not key.startswith('_')}
        summaries.append(clean)

    fieldnames = list(summaries[0]) if summaries else []
    with (args.output_dir / 'experimental_feature_measurements.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    finite_sigmas = []
    finite_spacings = []
    for row in summaries:
        for key in ('sigma_px_p20', 'sigma_px_p80'):
            if np.isfinite(row[key]):
                finite_sigmas.append(row[key])
        for key in ('spacing_px_p20', 'spacing_px_p80'):
            if np.isfinite(row[key]):
                finite_spacings.append(row[key])
    aggregate = {
        'images': summaries,
        'recommended_sigma_range_px': robust_range(np.asarray(finite_sigmas, dtype=np.float32), 10, 90),
        'recommended_spacing_range_px': robust_range(np.asarray(finite_spacings, dtype=np.float32), 10, 90),
        'method': {
            'dog_small': args.dog_small,
            'dog_large': args.dog_large,
            'min_distance': args.min_distance,
            'threshold_percentile': args.threshold_percentile,
            'profile_radius': args.profile_radius,
            'annulus_inner': args.annulus_inner,
            'annulus_outer': args.annulus_outer,
        },
    }
    (args.output_dir / 'experimental_feature_measurements.json').write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
