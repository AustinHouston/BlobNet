from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_CACHE_DIR = Path(tempfile.gettempdir()) / "blobnet-mpl-cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from blobnet.metrics import evaluate_heatmap_localization


def save_metrics_table(rows: List[Dict[str, float]], output_dir: Path) -> None:
    fieldnames = [
        "model",
        "test_case",
        "precision",
        "recall",
        "f1",
        "mean_error",
        "median_error",
        "rmse",
        "tp",
        "fp",
        "fn",
        "samples",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "benchmark_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    json_path = output_dir / "benchmark_metrics.json"
    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2)


def _metric_lookup(rows: Sequence[Dict[str, float]]) -> Dict[tuple[str, str], Dict[str, float]]:
    return {(row["model"], row["test_case"]): row for row in rows}


def plot_generalization_summary(
    rows: List[Dict[str, float]],
    output_dir: Path,
    case_order: Sequence[str],
    model_order: Sequence[str],
    filename: str,
    title: str,
) -> Path:
    if not rows:
        raise ValueError("No rows supplied for summary plotting.")

    lookup = _metric_lookup(rows)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    palette = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948"]
    x = np.arange(len(model_order))
    width = 0.8 / max(len(case_order), 1)

    for case_index, case_name in enumerate(case_order):
        color = palette[case_index % len(palette)]
        f1_values = []
        rmse_values = []
        for model_name in model_order:
            row = lookup.get((model_name, case_name))
            f1_values.append(np.nan if row is None else float(row["f1"]))
            rmse_values.append(np.nan if row is None else float(row["rmse"]))
        offset = (case_index - (len(case_order) - 1) / 2.0) * width
        axes[0, 0].bar(x + offset, np.nan_to_num(f1_values, nan=0.0), width=width, label=case_name, color=color)
        axes[0, 1].bar(x + offset, np.nan_to_num(rmse_values, nan=0.0), width=width, label=case_name, color=color)

    axes[0, 0].set_title("F1 by Test Case")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_xticks(x, model_order)
    axes[0, 0].legend()

    rmse_max = max(
        [float(row["rmse"]) for row in rows if not math.isnan(float(row["rmse"]))] or [1.0]
    )
    axes[0, 1].set_title("RMSE by Test Case (pixels)")
    axes[0, 1].set_ylim(0.0, rmse_max * 1.15)
    axes[0, 1].set_xticks(x, model_order)
    axes[0, 1].legend()

    f1_table = np.full((len(model_order), len(case_order)), np.nan, dtype=float)
    rmse_table = np.full((len(model_order), len(case_order)), np.nan, dtype=float)
    for model_index, model_name in enumerate(model_order):
        for case_index, case_name in enumerate(case_order):
            row = lookup.get((model_name, case_name))
            if row is not None:
                f1_table[model_index, case_index] = float(row["f1"])
                rmse_table[model_index, case_index] = float(row["rmse"])

    for ax, table, panel_title, vmin, vmax in [
        (axes[1, 0], f1_table, "F1 Table", 0.0, 1.0),
        (axes[1, 1], rmse_table, "RMSE Table", 0.0, max(rmse_max, 1e-6)),
    ]:
        im = ax.imshow(np.nan_to_num(table, nan=0.0), cmap="YlGnBu", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(panel_title)
        ax.set_xticks(np.arange(len(case_order)), case_order)
        ax.set_yticks(np.arange(len(model_order)), model_order)
        for row_idx in range(table.shape[0]):
            for col_idx in range(table.shape[1]):
                value = table[row_idx, col_idx]
                ax.text(
                    col_idx,
                    row_idx,
                    "NA" if math.isnan(float(value)) else f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_path = output_dir / filename
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_case_summaries(
    rows: List[Dict[str, float]],
    output_dir: Path,
    case_order: Sequence[str],
) -> None:
    model_order = sorted({row["model"] for row in rows})
    for case_name in case_order:
        case_rows = [row for row in rows if row["test_case"] == case_name]
        if not case_rows:
            continue
        plot_generalization_summary(
            rows=case_rows,
            output_dir=output_dir,
            case_order=[case_name],
            model_order=model_order,
            filename=f"benchmark_summary_{case_name}.png",
            title=f"BlobNet Summary: {case_name.title()}",
        )


def plot_loss_curves(model_names: Sequence[str], output_dir: Path) -> Path | None:
    histories = []
    for model_name in model_names:
        history_path = output_dir / model_name / f"{model_name}_loss_history.npz"
        if not history_path.exists():
            continue
        history = np.load(history_path)
        histories.append(
            (
                model_name,
                history["train_loss_history"].astype(float),
                history["val_loss_history"].astype(float),
            )
        )

    if not histories:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    fig.suptitle("BlobNet Training Curves", fontsize=14)

    for model_name, train_values, val_values in histories:
        epochs = np.arange(1, len(train_values) + 1)
        axes[0].plot(epochs, train_values, marker="o", linewidth=1.5, markersize=2.5, label=model_name)
        axes[1].plot(epochs, val_values, marker="o", linewidth=1.5, markersize=2.5, label=model_name)

    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.25)

    output_path = output_dir / "loss_curves.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _array_to_rgb(array: np.ndarray, mode: str = "gray") -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = array - array.min()
    peak = float(array.max())
    if peak > 0:
        array = array / peak
    array = np.clip(array, 0.0, 1.0)

    if mode == "gray":
        return (np.stack([array, array, array], axis=-1) * 255).astype(np.uint8)
    if mode == "heat":
        rgb = np.stack([array, np.sqrt(array), 1.0 - array * 0.6], axis=-1)
        return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    raise ValueError(f"Unsupported mode '{mode}'.")


def _draw_points(
    image: Image.Image,
    true_coords: np.ndarray,
    predicted_coords: np.ndarray,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for y, x in np.asarray(true_coords, dtype=np.float32):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=(0, 255, 0), width=1)
    for y, x in np.asarray(predicted_coords, dtype=np.float32):
        draw.line((x - 3, y - 3, x + 3, y + 3), fill=(0, 255, 255), width=1)
        draw.line((x - 3, y + 3, x + 3, y - 3), fill=(0, 255, 255), width=1)
    return output


def build_prediction_gallery(
    rows: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    panel_size = 220
    header_height = 44
    label_height = 20
    margin = 18
    columns = ["Input", "Target", "Prediction", "Overlay"]
    width = margin * 2 + len(columns) * panel_size
    height = margin * 2 + len(rows) * (panel_size + header_height + label_height)

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 8), "Prediction Gallery", fill="black", font=title_font)

    for row_index, row in enumerate(rows):
        y0 = margin + row_index * (panel_size + header_height + label_height)
        draw.text((margin, y0 + 4), row["label"], fill="black", font=title_font)
        draw.text((margin, y0 + 20), row["metrics_text"], fill="black", font=font)

        panels = [
            Image.fromarray(_array_to_rgb(row["image"], mode="gray")),
            Image.fromarray(_array_to_rgb(row["target"], mode="heat")),
            Image.fromarray(_array_to_rgb(row["prediction"], mode="heat")),
            _draw_points(
                Image.fromarray(_array_to_rgb(row["image"], mode="gray")),
                row["true_coords"],
                row["predicted_coords"],
            ),
        ]

        for column_index, (column_name, panel) in enumerate(zip(columns, panels)):
            x0 = margin + column_index * panel_size
            panel = panel.resize((panel_size - 10, panel_size - 10), resample=Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
            canvas.paste(panel, (x0 + 5, y0 + header_height))
            draw.text((x0 + 5, y0 + header_height + panel_size - 6), column_name, fill="black", font=font)
            draw.rectangle(
                (x0 + 4, y0 + header_height - 1, x0 + panel_size - 6, y0 + header_height + panel_size - 11),
                outline="black",
                width=1,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def collect_matched_offsets(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    threshold_rel: float,
    match_distance: float,
) -> Dict[str, np.ndarray | float | int]:
    model.eval()
    offsets_xy: List[np.ndarray] = []
    tp = fp = fn = samples = 0

    with torch.no_grad():
        for images, _targets, metadata_list in dataloader:
            predictions = torch.sigmoid(model(images.to(device)))[:, 0].detach().cpu().numpy()
            for heatmap, metadata in zip(predictions, metadata_list):
                result = evaluate_heatmap_localization(
                    heatmap,
                    metadata["coordinates"],
                    threshold_rel=threshold_rel,
                    match_distance=match_distance,
                )
                matched_truth = result["matched_truth"]
                matched_predicted = result["matched_predicted"]
                if len(matched_truth):
                    dx = matched_predicted[:, 1] - matched_truth[:, 1]
                    dy = matched_predicted[:, 0] - matched_truth[:, 0]
                    offsets_xy.append(np.stack([dx, dy], axis=1).astype(np.float32))
                tp += int(result["tp"])
                fp += int(result["fp"])
                fn += int(result["fn"])
                samples += 1

    offsets = np.concatenate(offsets_xy, axis=0) if offsets_xy else np.zeros((0, 2), dtype=np.float32)
    errors = np.linalg.norm(offsets, axis=1) if len(offsets) else np.zeros((0,), dtype=np.float32)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "offsets_xy": offsets,
        "errors": errors,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "samples": samples,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_dx": float(offsets[:, 0].mean()) if len(offsets) else float("nan"),
        "mean_dy": float(offsets[:, 1].mean()) if len(offsets) else float("nan"),
        "std_dx": float(offsets[:, 0].std()) if len(offsets) else float("nan"),
        "std_dy": float(offsets[:, 1].std()) if len(offsets) else float("nan"),
        "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else float("nan"),
    }


def save_offsets(case_results: Dict[str, Dict[str, np.ndarray | float | int]], output_path: Path) -> Path:
    np.savez_compressed(
        output_path,
        **{
            case_name: np.asarray(case_data["offsets_xy"], dtype=np.float32)
            for case_name, case_data in case_results.items()
        },
    )
    return output_path


def plot_offset_cloud(
    case_results: Dict[str, Dict[str, np.ndarray | float | int]],
    output_path: Path,
    model_name: str,
    plot_range: float,
    bins: int,
    point_alpha: float,
    point_size: float,
    dpi: int,
) -> Path:
    case_labels = {
        "random": "Random",
        "cubic": "Cubic",
        "hexagonal": "Hexagonal",
        "graphene": "Graphene",
        "ws2": "WS2",
        "sto": "STO",
    }
    case_names = list(case_results.keys())
    fig, axes = plt.subplots(1, len(case_names), figsize=(5.2 * max(len(case_names), 1), 5.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    image_handle = None

    for axis_index, (ax, case_name) in enumerate(zip(axes, case_names)):
        case_data = case_results[case_name]
        offsets = np.asarray(case_data["offsets_xy"], dtype=np.float32)
        in_view = (
            float(np.mean(np.linalg.norm(offsets, axis=1) <= plot_range))
            if len(offsets)
            else 0.0
        )

        if len(offsets):
            hist, x_edges, y_edges = np.histogram2d(
                offsets[:, 0],
                offsets[:, 1],
                bins=bins,
                range=[[-plot_range, plot_range], [-plot_range, plot_range]],
            )
            hist = hist.T
            vmax = max(float(hist.max()), 1.0)
            image_handle = ax.imshow(
                hist,
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                origin="lower",
                cmap="magma",
                norm=LogNorm(vmin=1, vmax=vmax),
                aspect="auto",
            )
            ax.scatter(offsets[:, 0], offsets[:, 1], s=point_size, c="white", alpha=point_alpha, linewidths=0)

        for radius in [0.1, 0.25, 0.5, 1.0]:
            if radius >= plot_range:
                continue
            ax.add_patch(plt.Circle((0.0, 0.0), radius, fill=False, linestyle=":", linewidth=0.8, color="white", alpha=0.35))

        ax.axhline(0.0, color="white", alpha=0.5, linewidth=1.0)
        ax.axvline(0.0, color="white", alpha=0.5, linewidth=1.0)
        ax.scatter([0.0], [0.0], s=70, c="white", marker="+", linewidths=1.8)
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)
        ax.set_xlabel("dx = pred_x - true_x [px]")
        if axis_index == 0:
            ax.set_ylabel("dy = pred_y - true_y [px]")
        ax.set_title(
            f"{case_labels.get(case_name, case_name.title())} test\n"
            f"matched={len(offsets)}  F1={float(case_data['f1']):.4f}  "
            f"RMSE={float(case_data['rmse']):.4f} px\n"
            f"mean(dx,dy)=({float(case_data['mean_dx']):.4f}, {float(case_data['mean_dy']):.4f})  "
            f"in-view={in_view * 100:.1f}%"
        )
        ax.set_facecolor("#120f1f")

    fig.suptitle(
        f"{model_name} Localization Error Cloud\n"
        "Each point is a matched atom offset relative to ground truth at (0, 0).",
        fontsize=16,
    )
    if image_handle is not None:
        fig.colorbar(image_handle, ax=axes.tolist(), fraction=0.025, pad=0.02, label="Matched atoms per bin")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

