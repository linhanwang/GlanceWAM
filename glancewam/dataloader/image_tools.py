"""Image conversion helpers shared by the frameworks and the policy server."""

from typing import Any

import numpy as np
from PIL import Image


def to_pil_preserve(images: Any, scale_float: bool = True):
    """Convert (possibly nested) numpy image arrays to ``PIL.Image`` **without**
    changing spatial shape or nesting structure.

    Accepts ``np.ndarray`` of shape ``(H, W, C)`` with ``C in {1, 3, 4}`` (uint8 or
    float), ``PIL.Image`` (returned as-is), and nested lists/tuples of either.

    No resize / pad / crop is performed — only dtype (float -> uint8) and channel-mode
    adaptation. Float arrays are assumed to be in ``[0, 1]`` when ``scale_float``.
    """

    def _convert(obj):
        if isinstance(obj, list):
            return [_convert(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_convert(x) for x in obj)
        if isinstance(obj, Image.Image):
            return obj

        if isinstance(obj, np.ndarray):
            arr = obj
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D array (H,W,C), got shape={arr.shape}")
            if arr.shape[2] not in (1, 3, 4):
                raise ValueError(f"Channel count must be 1/3/4, got {arr.shape[2]}")
            if np.issubdtype(arr.dtype, np.floating):
                if not scale_float:
                    raise TypeError("Float array provided but scale_float=False")
                arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)

            if arr.shape[2] == 1:
                return Image.fromarray(arr[:, :, 0], mode="L")
            return Image.fromarray(arr, mode="RGB" if arr.shape[2] == 3 else "RGBA")

        raise TypeError(f"Unsupported element type: {type(obj)}")

    return _convert(images)
