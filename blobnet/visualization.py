from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from blobnet.metrics import evaluate_heatmap_localization


def collect_matched_offsets(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    threshold_rel: float,
    match_distance: float,
) -> Dict[str, np.ndarray | float | int]:
    """Collect matched localization offsets and aggregate detection metrics."""
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
