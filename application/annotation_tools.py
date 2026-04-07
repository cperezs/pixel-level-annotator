"""Pure-function annotation tool computations.

All functions take and return NumPy arrays.  There are no side effects,
no Qt dependencies, and no domain state accessed here.  This makes every
function trivially unit-testable.
"""
from __future__ import annotations

import cv2
import numpy as np


# ------------------------------------------------------------------
# Pen tool
# ------------------------------------------------------------------

def compute_pen_mask(
    image_height: int,
    image_width: int,
    pos_x: int,
    pos_y: int,
    size: int,
) -> np.ndarray:
    """Return a circular uint8 mask of *size* centred on *(pos_x, pos_y)*."""
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    y, x = np.ogrid[:image_height, :image_width]
    radius = size // 2
    if size % 2 == 0:
        hit = (x - pos_x + 0.5) ** 2 + (y - pos_y + 0.5) ** 2 <= radius ** 2
    else:
        hit = (x - pos_x) ** 2 + (y - pos_y) ** 2 <= radius ** 2
    mask[hit] = 255
    return mask


# ------------------------------------------------------------------
# Mask manipulation
# ------------------------------------------------------------------

def apply_overwrite_guard(mask: np.ndarray, annotated: np.ndarray) -> np.ndarray:
    """Remove pixels already annotated by other layers from *mask*."""
    return cv2.bitwise_and(mask, cv2.bitwise_not(annotated))


def smooth_mask(mask: np.ndarray) -> np.ndarray:
    """Dilate then erode to smooth jagged edges in a binary mask."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    expanded = cv2.dilate(mask, kernel, iterations=1)
    return cv2.erode(expanded, kernel, iterations=1)


def expand_mask(
    mask: np.ndarray,
    annotated: np.ndarray | None = None,
) -> np.ndarray:
    """Grow the selection mask by one pixel in all eight directions.

    If *annotated* is given, already-annotated pixels are excluded from
    the grown region (overwrite guard).
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    grown = cv2.dilate(mask, kernel, iterations=1)
    if annotated is not None:
        grown = apply_overwrite_guard(grown, annotated)
    return grown


def shrink_mask(mask: np.ndarray) -> np.ndarray:
    """Erode the selection mask by one pixel in all eight directions."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.erode(mask, kernel, iterations=1)


# ------------------------------------------------------------------
# Compositing (for the viewer)
# ------------------------------------------------------------------

def build_annotation_rgba(
    annotations: np.ndarray,
    layer_colors: list[tuple[int, int, int]],
    active_layer: int,
    show_other_layers: bool,
    opacity: int = 128,
) -> np.ndarray:
    """Composite all visible annotation layers into a single RGBA image.

    Parameters
    ----------
    annotations:
        Shape (N, H, W), values 0 or 255.
    layer_colors:
        One (R, G, B) tuple per layer, same order as *annotations*.
    active_layer:
        Index of the currently selected layer.
    show_other_layers:
        When False, only the active layer is rendered.
    opacity:
        Alpha value used for annotated pixels (0–255).
    """
    n, h, w = annotations.shape
    composite = np.zeros((h, w, 4), dtype=np.uint8)

    for i in range(n):
        if i != active_layer and not show_other_layers:
            continue
        mask = annotations[i] > 0
        r, g, b = layer_colors[i]
        composite[mask] = (r, g, b, opacity)

    # Boost opacity where the primary channel is saturated (visual feedback).
    composite[:, :, 3] = np.where(composite[:, :, 0] == 255, 128, composite[:, :, 3])
    composite[:, :, 3] = np.where(composite[:, :, 2] == 255, 128, composite[:, :, 3])

    return composite


def build_mask_rgba(
    mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    opacity: int = 64,
) -> np.ndarray:
    """Convert a binary mask to a coloured RGBA overlay."""
    h, w = mask.shape
    result = np.zeros((h, w, 4), dtype=np.uint8)
    result[mask > 0] = color_rgb + (opacity,)
    return result
