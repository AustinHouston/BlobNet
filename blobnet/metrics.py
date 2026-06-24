from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment


def _ensure_2d_heatmap(heatmap: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim == 3 and heatmap.shape[0] == 1:
        heatmap = heatmap[0]
    if heatmap.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got shape {heatmap.shape}")
    return heatmap


def extract_subpixel_peak_positions(
    heatmap: np.ndarray | torch.Tensor,
    threshold_rel: float = 0.35,
    threshold_abs: Optional[float] = None,
    min_distance: int = 3,
    window_size: int = 5,
    max_peaks: Optional[int] = None,
) -> np.ndarray:
    """
    Find local maxima and refine them to sub-pixel coordinates with a weighted centroid.

    Coordinates are returned in ``(y, x)`` pixel units.
    """

    heatmap = _ensure_2d_heatmap(heatmap)

    if heatmap.size == 0 or float(heatmap.max()) <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    peak_threshold = (
        float(threshold_abs)
        if threshold_abs is not None
        else float(heatmap.max()) * float(threshold_rel)
    )
    footprint = 2 * int(min_distance) + 1
    local_max = heatmap == maximum_filter(heatmap, size=footprint, mode="nearest")
    candidate_mask = local_max & (heatmap >= peak_threshold)
    candidates = np.argwhere(candidate_mask)

    if len(candidates) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    candidate_values = heatmap[candidate_mask]
    order = np.argsort(candidate_values)[::-1]
    candidates = candidates[order]

    refined_positions: List[np.ndarray] = []
    half_window = max(1, window_size // 2)

    for y, x in candidates:
        if max_peaks is not None and len(refined_positions) >= max_peaks:
            break

        if refined_positions:
            distances = np.linalg.norm(np.stack(refined_positions) - np.array([y, x]), axis=1)
            if np.any(distances < min_distance):
                continue

        y0 = max(0, y - half_window)
        y1 = min(heatmap.shape[0], y + half_window + 1)
        x0 = max(0, x - half_window)
        x1 = min(heatmap.shape[1], x + half_window + 1)

        patch = heatmap[y0:y1, x0:x1].copy()
        patch = patch - patch.min()
        weight_sum = float(patch.sum())

        if weight_sum <= 0:
            refined_positions.append(np.array([y, x], dtype=np.float32))
            continue

        yy, xx = np.meshgrid(
            np.arange(y0, y1, dtype=np.float32),
            np.arange(x0, x1, dtype=np.float32),
            indexing="ij",
        )
        refined_y = float((yy * patch).sum() / weight_sum)
        refined_x = float((xx * patch).sum() / weight_sum)
        refined_positions.append(np.array([refined_y, refined_x], dtype=np.float32))

    if not refined_positions:
        return np.zeros((0, 2), dtype=np.float32)

    return np.stack(refined_positions, axis=0)


def match_coordinate_sets(
    predicted: Sequence[Sequence[float]],
    truth: Sequence[Sequence[float]],
    max_distance: float = 3.0,
) -> Dict[str, Any]:
    """
    Match predicted and true coordinates with Hungarian assignment.

    Coordinates use ``(y, x)`` ordering in pixels.
    """

    predicted = np.asarray(predicted, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)

    if predicted.size == 0:
        predicted = predicted.reshape(0, 2)
    if truth.size == 0:
        truth = truth.reshape(0, 2)

    if len(predicted) == 0 and len(truth) == 0:
        return {
            "matched_predicted": np.zeros((0, 2), dtype=np.float32),
            "matched_truth": np.zeros((0, 2), dtype=np.float32),
            "errors": np.zeros((0,), dtype=np.float32),
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

    if len(predicted) == 0:
        return {
            "matched_predicted": np.zeros((0, 2), dtype=np.float32),
            "matched_truth": np.zeros((0, 2), dtype=np.float32),
            "errors": np.zeros((0,), dtype=np.float32),
            "tp": 0,
            "fp": 0,
            "fn": int(len(truth)),
        }

    if len(truth) == 0:
        return {
            "matched_predicted": np.zeros((0, 2), dtype=np.float32),
            "matched_truth": np.zeros((0, 2), dtype=np.float32),
            "errors": np.zeros((0,), dtype=np.float32),
            "tp": 0,
            "fp": int(len(predicted)),
            "fn": 0,
        }

    distance_matrix = np.linalg.norm(
        predicted[:, None, :] - truth[None, :, :], axis=2
    )
    large_cost = max_distance * 1000.0
    cost_matrix = np.where(distance_matrix <= max_distance, distance_matrix, large_cost)
    pred_indices, truth_indices = linear_sum_assignment(cost_matrix)

    matched_predicted: List[np.ndarray] = []
    matched_truth: List[np.ndarray] = []
    errors: List[float] = []

    used_pred = set()
    used_truth = set()

    for pred_idx, truth_idx in zip(pred_indices.tolist(), truth_indices.tolist()):
        distance = float(distance_matrix[pred_idx, truth_idx])
        if distance > max_distance:
            continue
        used_pred.add(pred_idx)
        used_truth.add(truth_idx)
        matched_predicted.append(predicted[pred_idx])
        matched_truth.append(truth[truth_idx])
        errors.append(distance)

    tp = len(errors)
    fp = int(len(predicted) - len(used_pred))
    fn = int(len(truth) - len(used_truth))

    return {
        "matched_predicted": np.stack(matched_predicted, axis=0)
        if matched_predicted
        else np.zeros((0, 2), dtype=np.float32),
        "matched_truth": np.stack(matched_truth, axis=0)
        if matched_truth
        else np.zeros((0, 2), dtype=np.float32),
        "errors": np.asarray(errors, dtype=np.float32),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_heatmap_localization(
    predicted_heatmap: np.ndarray | torch.Tensor,
    true_coordinates: Sequence[Sequence[float]],
    threshold_rel: float = 0.35,
    threshold_abs: Optional[float] = None,
    min_distance: int = 3,
    peak_window_size: int = 5,
    match_distance: float = 3.0,
    max_peaks: Optional[int] = None,
) -> Dict[str, Any]:
    predicted_coordinates = extract_subpixel_peak_positions(
        predicted_heatmap,
        threshold_rel=threshold_rel,
        threshold_abs=threshold_abs,
        min_distance=min_distance,
        window_size=peak_window_size,
        max_peaks=max_peaks,
    )

    matches = match_coordinate_sets(
        predicted_coordinates, true_coordinates, max_distance=match_distance
    )
    errors = matches["errors"]
    tp = matches["tp"]
    fp = matches["fp"]
    fn = matches["fn"]

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "predicted_coordinates": predicted_coordinates,
        "true_coordinates": np.asarray(true_coordinates, dtype=np.float32),
        "matched_predicted": matches["matched_predicted"],
        "matched_truth": matches["matched_truth"],
        "errors": errors,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_error": float(errors.mean()) if len(errors) else np.nan,
        "median_error": float(np.median(errors)) if len(errors) else np.nan,
        "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan,
    }


def aggregate_localization_metrics(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    results = list(results)
    if not results:
        raise ValueError("Expected at least one sample result to aggregate.")

    all_errors = [
        np.asarray(result["errors"], dtype=np.float32)
        for result in results
        if len(result["errors"]) > 0
    ]
    concatenated_errors = (
        np.concatenate(all_errors, axis=0)
        if all_errors
        else np.zeros((0,), dtype=np.float32)
    )

    tp = int(sum(result["tp"] for result in results))
    fp = int(sum(result["fp"] for result in results))
    fn = int(sum(result["fn"] for result in results))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "samples": len(results),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_error": float(concatenated_errors.mean())
        if len(concatenated_errors)
        else np.nan,
        "median_error": float(np.median(concatenated_errors))
        if len(concatenated_errors)
        else np.nan,
        "rmse": float(np.sqrt(np.mean(concatenated_errors**2)))
        if len(concatenated_errors)
        else np.nan,
        "all_errors": concatenated_errors,
    }


def evaluate_model_localization(
    model: torch.nn.Module,
    dataloader,
    device: torch.device | str = "cpu",
    channel: int = 0,
    apply_sigmoid: bool = False,
    threshold_rel: float = 0.35,
    threshold_abs: Optional[float] = None,
    min_distance: int = 3,
    peak_window_size: int = 5,
    match_distance: float = 3.0,
    max_peaks: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    model.to(device)
    sample_results: List[Dict[str, Any]] = []

    with torch.no_grad():
        for images, _targets, metadata_list in dataloader:
            predictions = model(images.to(device))
            if apply_sigmoid:
                predictions = torch.sigmoid(predictions)
            predictions = predictions[:, channel].detach().cpu().numpy()

            for heatmap, metadata in zip(predictions, metadata_list):
                sample_results.append(
                    evaluate_heatmap_localization(
                        heatmap,
                        metadata["coordinates"],
                        threshold_rel=threshold_rel,
                        threshold_abs=threshold_abs,
                        min_distance=min_distance,
                        peak_window_size=peak_window_size,
                        match_distance=match_distance,
                        max_peaks=max_peaks,
                    )
                )

    summary = aggregate_localization_metrics(sample_results)
    summary["sample_results"] = sample_results
    return summary


def plot_localization_result(
    image: np.ndarray | torch.Tensor,
    predicted_heatmap: np.ndarray | torch.Tensor,
    true_coordinates: Sequence[Sequence[float]],
    predicted_coordinates: Optional[Sequence[Sequence[float]]] = None,
    figsize: Tuple[float, float] = (12.0, 4.0),
):
    import matplotlib.pyplot as plt

    image = _ensure_2d_heatmap(image)
    predicted_heatmap = _ensure_2d_heatmap(predicted_heatmap)
    true_coordinates = np.asarray(true_coordinates, dtype=np.float32)
    predicted_coordinates = (
        np.asarray(predicted_coordinates, dtype=np.float32)
        if predicted_coordinates is not None
        else np.zeros((0, 2), dtype=np.float32)
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(predicted_heatmap, cmap="magma")
    axes[1].set_title("Predicted Heatmap")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    if len(true_coordinates):
        axes[2].scatter(
            true_coordinates[:, 1],
            true_coordinates[:, 0],
            s=30,
            c="lime",
            label="True",
            edgecolors="black",
            linewidths=0.4,
        )
    if len(predicted_coordinates):
        axes[2].scatter(
            predicted_coordinates[:, 1],
            predicted_coordinates[:, 0],
            s=25,
            c="cyan",
            marker="x",
            label="Pred",
            linewidths=1.2,
        )
    axes[2].set_title("Localization")
    axes[2].axis("off")
    if len(true_coordinates) or len(predicted_coordinates):
        axes[2].legend(loc="upper right")

    return fig, axes
