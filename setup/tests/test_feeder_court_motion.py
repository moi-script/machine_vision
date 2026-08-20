import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feeder_court import motion  # noqa: E402


def _scene(shift=0):
    """A textured static background with a bright vertical line, optionally shifted.

    The WHOLE scene is rolled, texture included. Shifting only the line would
    leave the texture unmoved, and the true translation between two such frames
    is zero - phase correlation reports that correctly, at 0.94 confidence.
    """
    rng = np.random.default_rng(0)
    img = (rng.random((180, 320)) * 40).astype(np.uint8)
    img[:, 100:104] = 220                          # a "court line"
    return np.roll(img, shift, axis=1)


def test_alignment_recovers_a_known_translation():
    ref = _scene(0)
    src = _scene(6)
    aligned = motion.translation_align(src, ref)
    # After alignment the bright line should sit back near column 100.
    col = int(np.argmax(aligned.astype(float).mean(axis=0)))
    assert abs(col - 101) <= 2


def test_three_frame_diff_cancels_a_drifting_static_scene():
    prev, cur, nxt = _scene(-4), _scene(0), _scene(4)
    d = motion.three_frame_diff(prev, cur, nxt)
    # The static line must not survive: nothing should be strongly lit.
    assert d.max() < 60


def test_three_frame_diff_keeps_a_fast_mover():
    prev, cur, nxt = _scene(0), _scene(0), _scene(0)
    cur = cur.copy()
    cur[80:88, 200:208] = 255          # a shuttle present only in the middle frame
    d = motion.three_frame_diff(prev, cur, nxt)
    assert d[80:88, 200:208].max() > 100


def test_candidate_boxes_finds_the_mover():
    d = np.zeros((180, 320), dtype=np.uint8)
    d[80:88, 200:208] = 255
    cands = motion.candidate_boxes(d, frame_index=7)
    assert len(cands) == 1
    c = cands[0]
    assert c.frame == 7
    assert 195 <= c.x <= 205
    assert 4 <= c.w <= 12


def test_candidate_boxes_rejects_objects_that_are_too_large():
    d = np.zeros((180, 320), dtype=np.uint8)
    d[20:120, 20:120] = 255            # a person, not a shuttle
    assert motion.candidate_boxes(d, frame_index=0) == []


def test_candidate_boxes_rejects_extreme_aspect_ratios():
    d = np.zeros((180, 320), dtype=np.uint8)
    d[100:104, 20:140] = 255           # a line fragment: 120x4
    assert motion.candidate_boxes(d, frame_index=0) == []


def test_candidate_boxes_rejects_sparse_blobs():
    d = np.zeros((180, 320), dtype=np.uint8)
    # A diagonal streak: square bounding box, very low fill (0.098). A hollow
    # ring does not work here - a 30x30 ring with a 5 px wall fills 0.556 of its
    # box, well above min_fill. The streak is also the case min_fill uniquely
    # catches: its bbox is square, so the aspect-ratio filter lets it through.
    for i in range(30):
        d[60 + i, 60 + i:63 + i] = 255
    assert motion.candidate_boxes(d, frame_index=0) == []


def test_candidate_boxes_records_peak_intensity():
    d = np.zeros((180, 320), dtype=np.uint8)
    d[80:86, 200:206] = 140
    cands = motion.candidate_boxes(d, frame_index=0)
    assert len(cands) == 1
    assert cands[0].peak == 140.0
