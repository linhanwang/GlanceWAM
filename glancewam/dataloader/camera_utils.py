"""Camera-view packing helpers shared across the dataloader, the VAE-cache
precompute tool, and the world-model frameworks.

The only entry point today is :func:`stitch_views_side_by_side`, which packs
multiple camera views into a single wide frame so a *single-stream* world model
(e.g. the Cosmos-Predict2 VAE/DiT) can condition on more than one camera. This
mirrors the LIBERO packing used by the sibling DiT4DiT codebase: each view is
square-resized and concatenated left-to-right (primary | wrist | ...).
"""

from typing import Sequence, Union

import numpy as np
from PIL import Image

ImageLike = Union[Image.Image, np.ndarray]


def stitch_views_side_by_side(
    views: Sequence[ImageLike],
    height: int = 224,
    per_view_width: int = 224,
) -> Image.Image:
    """Concatenate camera views side-by-side (along width) into one image.

    Each view is resized to ``(height, per_view_width)`` and pasted left-to-right
    in the order given (``[primary, wrist, ...]``), yielding a single
    ``(per_view_width * n, height)`` RGB image. A downstream single-stream world
    model then sees both cameras in one coherent frame.

    Args:
        views: one PIL.Image or ``(H, W, C)`` numpy array per camera.
        height: output height (per view and overall).
        per_view_width: output width allotted to each view; total width is
            ``per_view_width * len(views)``.

    Returns:
        A ``PIL.Image`` of size ``(per_view_width * len(views), height)``.
    """
    if not views:
        raise ValueError("stitch_views_side_by_side got an empty view list")
    n = len(views)
    canvas = Image.new("RGB", (per_view_width * n, height))
    for i, v in enumerate(views):
        if not isinstance(v, Image.Image):
            v = Image.fromarray(np.asarray(v))
        canvas.paste(v.convert("RGB").resize((per_view_width, height)), (i * per_view_width, 0))
    return canvas


def _as_pil(v: ImageLike) -> Image.Image:
    return v.convert("RGB") if isinstance(v, Image.Image) else Image.fromarray(np.asarray(v)).convert("RGB")


def stitch_primary_with_insets(
    primary: ImageLike,
    insets: Sequence[ImageLike],
    main_size: int = 224,
    inset_size: int = 112,
) -> Image.Image:
    """Pack a full-res primary view plus smaller inset views into one frame.

    The primary is placed at ``main_size x main_size`` on the left; each inset is
    resized to ``inset_size x inset_size`` and stacked top-to-bottom in a right-hand
    column of width ``inset_size``. Used by the RoboCasa-kitchen 3-camera layout
    (``camera_concat: primary_inset``): primary 224 + [secondary, wrist] 112 insets
    -> one ``336 x 224`` (WxH) frame for the single-stream world model.

    ``len(insets) * inset_size`` should equal ``main_size`` so the inset column
    exactly spans the primary's height (2 x 112 == 224); a shorter column is
    top-anchored on a black canvas, a taller one is clipped by the paste.

    Returns:
        A ``PIL.Image`` of size ``(main_size + inset_size, main_size)``.
    """
    if not insets:
        raise ValueError("stitch_primary_with_insets got no inset views")
    canvas = Image.new("RGB", (main_size + inset_size, main_size))
    canvas.paste(_as_pil(primary).resize((main_size, main_size)), (0, 0))
    for i, v in enumerate(insets):
        canvas.paste(_as_pil(v).resize((inset_size, inset_size)), (main_size, i * inset_size))
    return canvas
