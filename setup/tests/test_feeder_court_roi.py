import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feeder_court import roi  # noqa: E402


def test_band_snaps_up_to_multiple_and_clamps_to_frame():
    band = roi.band_from_extent(y_min=300.0, y_max=520.0, frame_h=720, margin=24, multiple_of=32)
    assert band.height % 32 == 0
    assert band.top >= 0
    assert band.bottom <= 720
    # The requested extent plus margin must still fit inside the band.
    assert band.top <= 300 - 24 + 1
    assert band.bottom >= 520 + 24 - 1


def test_band_never_exceeds_frame_when_extent_touches_edges():
    band = roi.band_from_extent(y_min=0.0, y_max=719.0, frame_h=720, margin=48, multiple_of=32)
    assert band.top == 0
    assert band.bottom <= 720


def test_apply_crop_returns_band_height():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    band = roi.CropBand(top=192, height=448)
    out = roi.apply_crop(frame, band)
    assert out.shape == (448, 1280, 3)


def test_apply_crop_preserves_pixels():
    frame = np.arange(720 * 1280, dtype=np.uint8).reshape(720, 1280)
    band = roi.CropBand(top=100, height=64)
    out = roi.apply_crop(frame, band)
    assert np.array_equal(out, frame[100:164])


def test_shift_boxes_subtracts_top_from_y_only():
    band = roi.CropBand(top=192, height=448)
    boxes = np.array([[10.0, 200.0, 30.0, 240.0]])
    out = roi.shift_boxes_into_band(boxes, band)
    assert out.tolist() == [[10.0, 8.0, 30.0, 48.0]]


def test_boxes_fully_inside_rejects_boxes_crossing_the_band_edge():
    band = roi.CropBand(top=192, height=448)
    boxes = np.array([
        [10.0, 200.0, 30.0, 240.0],   # inside
        [10.0, 180.0, 30.0, 210.0],   # crosses top edge
        [10.0, 620.0, 30.0, 700.0],   # crosses bottom edge
    ])
    mask = roi.boxes_fully_inside(boxes, band)
    assert mask.tolist() == [True, False, False]


def test_crop_then_shift_round_trips_to_the_same_pixel():
    frame = np.zeros((720, 1280), dtype=np.uint8)
    frame[300, 640] = 255
    band = roi.CropBand(top=192, height=448)
    cropped = roi.apply_crop(frame, band)
    boxes = np.array([[639.0, 299.0, 641.0, 301.0]])
    shifted = roi.shift_boxes_into_band(boxes, band)
    y = int(shifted[0][1]) + 1
    x = int(shifted[0][0]) + 1
    assert cropped[y, x] == 255


def test_frame_containment_wins_over_stride_alignment_at_full_frame():
    # frame_h=720 is not a multiple of 32, so these two constraints cannot both
    # hold. Containment must win: a band outside the frame would crash the crop.
    band = roi.band_from_extent(y_min=0.0, y_max=719.0, frame_h=720, margin=48, multiple_of=32)
    assert band.bottom <= 720
    assert band.height % 32 != 0  # documents the deliberate trade-off


def test_apply_crop_preserves_pixels_for_colour_frames():
    frame = np.arange(720 * 1280 * 3, dtype=np.uint8).reshape(720, 1280, 3)
    band = roi.CropBand(top=100, height=64)
    out = roi.apply_crop(frame, band)
    assert np.array_equal(out, frame[100:164])
