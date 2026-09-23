"""Load the curated experimental microscopy images through pyTEMlib."""

from pathlib import Path
from typing import Any

import numpy as np
import pyTEMlib.file_tools


def open_experimental_image(path: str | Path, channel: str = "Channel_000") -> tuple[np.ndarray, dict[str, Any]]:
    """Return one 2D image and its stored metadata from an NSID file."""
    path = Path(path)
    datasets = pyTEMlib.file_tools.open_file(str(path))
    if channel not in datasets:
        raise KeyError(f"{channel!r} is not present in {path}; available channels: {list(datasets)}")

    dataset = datasets[channel]
    image = np.asarray(dataset, dtype=np.float32).squeeze()
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image in {path}:{channel}, found shape {image.shape}")

    metadata = dict(getattr(dataset, "metadata", {}))
    if "pixel_size_nm" not in metadata:
        raise ValueError(f"Missing pixel_size_nm metadata in {path}:{channel}")
    return image, metadata

