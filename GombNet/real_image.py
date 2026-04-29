from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, uniform_filter


def load_velox_emd_image(path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    with h5py.File(path, "r") as handle:
        image_group = next(iter(handle["Data/Image"].keys()))
        data = np.array(handle[f"Data/Image/{image_group}/Data"])
        metadata_raw = np.array(handle[f"Data/Image/{image_group}/Metadata"]).astype(np.uint8).reshape(-1)

    image = np.squeeze(data).astype(np.float32)
    metadata_text = bytes(metadata_raw.tolist()).decode("utf-8", errors="ignore")
    metadata_text = metadata_text.lstrip("\x00\r\n\t ")
    metadata, _ = json.JSONDecoder().raw_decode(metadata_text)
    return image, metadata


def get_real_image_pixel_size_angstrom(metadata: Dict[str, object]) -> float | None:
    try:
        pixel_size_m = float(metadata["BinaryResult"]["PixelSize"]["width"])
        return pixel_size_m * 1e10
    except Exception:
        return None


def percentile_normalize(image: np.ndarray, pmin: float = 1.0, pmax: float = 99.7) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo = float(np.percentile(image, pmin))
    hi = float(np.percentile(image, pmax))
    if hi <= lo:
        lo = float(image.min())
        hi = float(image.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (image - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def preprocess_real_image_variants(image: np.ndarray) -> Dict[str, np.ndarray]:
    raw = percentile_normalize(image, pmin=0.5, pmax=99.8)
    background = gaussian_filter(image.astype(np.float32), sigma=18.0, mode="reflect")
    flatfield = image.astype(np.float32) / np.maximum(background, 1e-6)
    flatfield = percentile_normalize(flatfield, pmin=0.5, pmax=99.5)
    highpass = image.astype(np.float32) - background
    highpass = percentile_normalize(highpass, pmin=1.0, pmax=99.5)
    return {
        "raw_percentile": raw,
        "flatfield_normalized": flatfield,
        "background_subtracted": highpass,
    }


def generate_tile_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    if length <= tile_size:
        return [0]
    step = max(1, tile_size - overlap)
    starts = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_blend_window(height: int, width: int) -> np.ndarray:
    wy = np.hanning(height) if height > 1 else np.ones((1,), dtype=np.float32)
    wx = np.hanning(width) if width > 1 else np.ones((1,), dtype=np.float32)
    window = np.outer(wy, wx).astype(np.float32)
    return 0.1 + 0.9 * window


def predict_heatmap_tiled(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: int,
    overlap: int,
    batch_size: int = 4,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    height, width = image.shape

    if height <= tile_size and width <= tile_size:
        with torch.inference_mode():
            tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
            return torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy().astype(np.float32)

    y_starts = generate_tile_starts(height, tile_size, overlap)
    x_starts = generate_tile_starts(width, tile_size, overlap)
    accum = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    tiles: List[np.ndarray] = []
    coords: List[Tuple[int, int, int, int]] = []
    with torch.inference_mode():
        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(height, y0 + tile_size)
                x1 = min(width, x0 + tile_size)
                tiles.append(image[y0:y1, x0:x1])
                coords.append((y0, y1, x0, x1))

                if len(tiles) < batch_size:
                    continue

                batch = torch.from_numpy(np.stack(tiles, axis=0)).unsqueeze(1).to(device)
                predictions = torch.sigmoid(model(batch))[:, 0].detach().cpu().numpy()
                for prediction, (ty0, ty1, tx0, tx1) in zip(predictions, coords):
                    window = make_blend_window(ty1 - ty0, tx1 - tx0)
                    accum[ty0:ty1, tx0:tx1] += prediction[: ty1 - ty0, : tx1 - tx0] * window
                    weight[ty0:ty1, tx0:tx1] += window
                tiles.clear()
                coords.clear()

        if tiles:
            batch = torch.from_numpy(np.stack(tiles, axis=0)).unsqueeze(1).to(device)
            predictions = torch.sigmoid(model(batch))[:, 0].detach().cpu().numpy()
            for prediction, (ty0, ty1, tx0, tx1) in zip(predictions, coords):
                window = make_blend_window(ty1 - ty0, tx1 - tx0)
                accum[ty0:ty1, tx0:tx1] += prediction[: ty1 - ty0, : tx1 - tx0] * window
                weight[ty0:ty1, tx0:tx1] += window

    return (accum / np.maximum(weight, 1e-6)).astype(np.float32)


def select_informative_crops(
    image: np.ndarray,
    crop_size: int,
    num_crops: int,
) -> List[Tuple[int, int, int, int]]:
    height, width = image.shape
    crop_size = min(crop_size, height, width)
    half = crop_size // 2

    mean = uniform_filter(image.astype(np.float32), size=max(8, crop_size // 6), mode="reflect")
    sq_mean = uniform_filter((image.astype(np.float32) ** 2), size=max(8, crop_size // 6), mode="reflect")
    local_variance = np.maximum(sq_mean - mean**2, 0.0)
    score = gaussian_filter(local_variance, sigma=max(1.0, crop_size / 10.0), mode="reflect")

    flat_indices = np.argsort(score.ravel())[::-1]
    boxes: List[Tuple[int, int, int, int]] = []

    for flat_index in flat_indices:
        y, x = np.unravel_index(int(flat_index), score.shape)
        y0 = int(np.clip(y - half, 0, max(0, height - crop_size)))
        x0 = int(np.clip(x - half, 0, max(0, width - crop_size)))
        box = (y0, y0 + crop_size, x0, x0 + crop_size)

        if not boxes:
            boxes.append(box)
        else:
            cy = y0 + crop_size / 2.0
            cx = x0 + crop_size / 2.0
            too_close = False
            for by0, by1, bx0, bx1 in boxes:
                if np.hypot(cy - (by0 + by1) / 2.0, cx - (bx0 + bx1) / 2.0) < crop_size * 0.6:
                    too_close = True
                    break
            if not too_close:
                boxes.append(box)

        if len(boxes) >= num_crops:
            break

    if len(boxes) < num_crops:
        center_y0 = max(0, (height - crop_size) // 2)
        center_x0 = max(0, (width - crop_size) // 2)
        center_box = (center_y0, center_y0 + crop_size, center_x0, center_x0 + crop_size)
        if center_box not in boxes:
            boxes.append(center_box)

    return boxes[:num_crops]
