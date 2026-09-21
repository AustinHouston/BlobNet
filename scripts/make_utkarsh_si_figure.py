"""Regenerate SI Figure S17 directly from the tracked Au-in-TiO2 HDF5 image."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patheffects
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from blobnet.metrics import extract_subpixel_peak_positions
from scripts.make_manuscript_figures import (
    _add_physical_scale_bar, _interpolate_image, _load_blobnet_model,
    _normalize_image, _predict_tiled,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "experimental_data/gold_implanted_in_TiO2.h5"
MODEL_ROOT = ROOT / "artifacts/manuscript_models"
OUT = ROOT / "outputs/utkarsh_si_figure"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_image() -> tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(IMAGE_PATH, "r") as handle:
        full_image = np.asarray(handle["image"], dtype=np.float32)
        pixel_size_nm = float(handle["image"].attrs["pixel_size_nm"])
    crop = full_image[256:768, 256:768]
    field = gaussian_filter(crop, 5, mode="reflect")
    field_median = float(np.median(field))
    corrected = crop / np.maximum(field, 0.05 * field_median) * field_median
    normalized = _normalize_image(corrected, low=1.0, high=99.8)
    display = _normalize_image(
        gaussian_filter(normalized, 1, mode="reflect")
        - gaussian_filter(normalized, 10, mode="reflect"),
        low=1.0, high=99.8,
    )
    inference = _normalize_image(_interpolate_image(display, (301, 301)), low=1.0, high=99.8)
    return display, inference, pixel_size_nm


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    display, inference, pixel_size_nm = prepare_image()
    device = torch.device("cpu")
    models = [
        ("Blob-Net", MODEL_ROOT / "random/unet_best.pth", "#E69F00", 36),
        ("Hex-Net", MODEL_ROOT / "hexagonal/unet_best.pth", "#00BFFF", 18),
        ("Square-Net", MODEL_ROOT / "square/unet_best.pth", "#CC33CC", 6),
    ]
    positions = {}
    for name, checkpoint, _color, _size in models:
        model = _load_blobnet_model(checkpoint, device, [32, 64, 128, 256], 0.2)
        prediction = _predict_tiled(model, inference, device, 256, 64, 4)
        coordinates = extract_subpixel_peak_positions(
            prediction, threshold_rel=0.10, min_distance=3, window_size=5
        )
        positions[name] = np.asarray(coordinates, dtype=np.float32) * (511.0 / 300.0)

    windows = [(0, 512, 0, 512), (180, 340, 100, 260), (352, 512, 220, 380)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), constrained_layout=True)
    for panel, (axis, (y0, y1, x0, x1)) in enumerate(zip(axes, windows)):
        axis.imshow(display[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1)
        for name, _checkpoint, color, size in models:
            coordinates = positions[name]
            inside = ((coordinates[:, 0] >= y0) & (coordinates[:, 0] < y1)
                      & (coordinates[:, 1] >= x0) & (coordinates[:, 1] < x1))
            plotted = coordinates[inside] - np.array([y0, x0])
            scatter = axis.scatter(
                plotted[:, 1], plotted[:, 0], s=size * (5 if panel else 1),
                facecolors="none", edgecolors=color, linewidths=1 if panel else 0.7,
            )
            scatter.set_path_effects([
                patheffects.Stroke(linewidth=1.7 if panel else 1.15, foreground="black", alpha=0.65),
                patheffects.Normal(),
            ])
        axis.set(xticks=[], yticks=[], xlim=(-0.5, x1-x0-0.5), ylim=(y1-y0-0.5, -0.5))
        axis.set_title(chr(ord("a") + panel), loc="left", fontsize=20, fontweight="bold")
        _add_physical_scale_bar(axis, (y1-y0, x1-x0), pixel_size_nm,
                                length_nm=1 if panel == 0 else 0.5, linewidth=3)
    for label, window, color in [("b", windows[1], "white"), ("c", windows[2], "#FF66CC")]:
        y0, y1, x0, x1 = window
        axes[0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color, linewidth=1.3)
        axes[0].text(x0+4, y0+17, label, color=color, fontsize=13, fontweight="bold")
    fig.legend(handles=[Line2D([], [], linestyle="none", marker="o", markerfacecolor="none",
                              markeredgecolor=color, markersize=np.sqrt(size)+2, label=name)
                        for name, _checkpoint, color, size in models],
               loc="outside upper center", ncol=3, frameon=False, fontsize=14)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"fig-S17.{extension}", dpi=300)
    plt.close(fig)
    provenance = {
        "figure": "S17", "image": str(IMAGE_PATH.relative_to(ROOT)),
        "image_sha256": sha256(IMAGE_PATH), "image_crop_yx": [256, 256, 512, 512],
        "flat_field_sigma_source_px": 5, "dog_sigmas_source_px": [1, 10],
        "normalization_percentiles": [1, 99.8], "inference_shape": [301, 301],
        "threshold_rel": 0.10,
        "model_sha256": {name: sha256(checkpoint) for name, checkpoint, _color, _size in models},
        "prediction_counts": {name: len(value) for name, value in positions.items()},
    }
    (OUT / "fig-S17.json").write_text(json.dumps(provenance, indent=2))
    np.savez_compressed(OUT / "fig-S17.npz", background=display,
                        **{name.replace("-", "_")+"_coordinates_yx": value
                           for name, value in positions.items()})
    (OUT / "utkarsh_gold_tio2.tex").write_text(r'''\clearpage
\setcounter{section}{16}
\section{Gold implanted in titanium dioxide}
\begin{figure}[!htbp]\centering
\includegraphics[width=\textwidth]{fig-S17.pdf}
\caption{Three-model localization comparison for experimental gold implanted in TiO$_2$. \textbf{a,} Centered 512 by 512 source-pixel HAADF-STEM view. \textbf{b,c,} Boxed 160 by 160 pixel regions. Orange, cyan, and magenta rings show Blob-Net, Hex-Net, and Square-Net predictions, respectively. The tracked DCFI(HAADF) image is locally normalized with a Gaussian field ($\sigma=5$ source pixels), processed using a difference of Gaussians ($\sigma=1$ and 10 source pixels), and resampled to 301 by 301 pixels for inference. Scale bars: a, 1 nm; b,c, 0.5 nm.}
\label{fig:au-tio2-three-models}
\end{figure}
''')
    print(OUT / "fig-S17.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
