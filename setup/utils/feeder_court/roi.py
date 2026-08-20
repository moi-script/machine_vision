# ============================================================
# roi.py — Court-band crop geometry.
#
# The OV9281 frames are 1280x720 but the upper third is unlit ceiling where no
# shuttle ever appears. Cropping that away is 2.7x cheaper than letterboxing the
# full frame to 1280x1280 and costs zero shuttle pixels.
#
# The same band MUST be applied at training and at inference. A mismatch silently
# rescales every object, which is how the imgsz-640 half-scaling bug happened.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CropBand:
    """A horizontal slice of the frame, in native pixels."""

    top: int
    height: int

    @property
    def bottom(self) -> int:
        return self.top + self.height


def band_from_extent(
    y_min: float,
    y_max: float,
    frame_h: int,
    margin: int = 24,
    multiple_of: int = 32,
) -> CropBand:
    """Smallest band covering [y_min, y_max] plus margin, snapped for the network.

    Height is rounded UP to a multiple of `multiple_of` (YOLO strides), then the
    band is clamped inside the frame. Clamping can only shrink height, never push
    the band outside the image.
    """
    lo = int(np.floor(y_min - margin))
    hi = int(np.ceil(y_max + margin))
    lo = max(lo, 0)
    hi = min(hi, frame_h)

    height = hi - lo
    remainder = height % multiple_of
    if remainder:
        height += multiple_of - remainder

    # Grow downward first, then upward, then clamp.
    if lo + height > frame_h:
        lo = max(frame_h - height, 0)
    height = min(height, frame_h - lo)

    return CropBand(top=lo, height=height)


def apply_crop(frame: np.ndarray, band: CropBand) -> np.ndarray:
    """Slice the band out of a frame. Works for 2D grayscale and 3D colour."""
    return frame[band.top:band.bottom]


def shift_boxes_into_band(boxes_xyxy: np.ndarray, band: CropBand) -> np.ndarray:
    """Rebase absolute xyxy boxes into the cropped frame's coordinates."""
    out = np.asarray(boxes_xyxy, dtype=float).copy()
    out[:, 1] -= band.top
    out[:, 3] -= band.top
    return out


def boxes_fully_inside(boxes_xyxy: np.ndarray, band: CropBand) -> np.ndarray:
    """Mask of boxes wholly within the band.

    Boxes crossing an edge are rejected rather than clipped: a clipped shuttle box
    is a wrong label, and wrong labels are what poisoned scene_v2.
    """
    b = np.asarray(boxes_xyxy, dtype=float)
    return (b[:, 1] >= band.top) & (b[:, 3] <= band.bottom)
