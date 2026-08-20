# ============================================================
# tracks.py — Turn noisy per-frame candidates into shuttle flights.
#
# A shuttlecock flight is 15-40 frames, moves fast, and goes roughly where its
# velocity says it will. Court-line artifacts produced tracks 100-738 frames
# long; that length cap alone removes most of them.
#
# bucket_frames implements the three-way positive/negative/discard split. The
# discard bucket is the important one: a frame that produced a candidate we
# REJECTED is not evidence of an empty frame, and writing it as an empty label
# teaches the model that a real shuttle is background.
# ============================================================

from __future__ import annotations

import collections
import math

from .motion import Candidate


def link_tracks(
    cands: list[Candidate],
    max_gap: int = 2,
    max_dist: float = 38.0,
) -> list[list[Candidate]]:
    """Greedy nearest-neighbour linking with constant-velocity prediction."""
    by_frame: dict[int, list[Candidate]] = collections.defaultdict(list)
    for c in cands:
        by_frame[c.frame].append(c)

    finished: list[list[Candidate]] = []
    active: list[list[Candidate]] = []

    for frame in sorted(by_frame):
        dets = by_frame[frame]
        used: set[int] = set()
        still_active: list[list[Candidate]] = []

        for track in active:
            last = track[-1]
            if frame - last.frame > max_gap:
                finished.append(track)
                continue

            if len(track) >= 2:
                vx = last.x - track[-2].x
                vy = last.y - track[-2].y
            else:
                vx = vy = 0
            px, py = last.x + vx, last.y + vy

            best, best_dist = None, float("inf")
            for j, d in enumerate(dets):
                if j in used:
                    continue
                dist = math.hypot(d.x - px, d.y - py)
                if dist < best_dist:
                    best_dist, best = dist, j

            if best is not None and best_dist < max_dist:
                used.add(best)
                track.append(dets[best])
            still_active.append(track)

        for j, d in enumerate(dets):
            if j not in used:
                still_active.append([d])
        active = still_active

    finished.extend(active)
    return finished


def is_flight(
    track: list[Candidate],
    min_len: int = 5,
    max_len: int = 60,
    min_step: float = 6.0,
    min_span: float = 60.0,
    min_directness: float = 0.45,
) -> bool:
    """Whether a track looks like a shuttlecock in flight."""
    if not (min_len <= len(track) <= max_len):
        return False

    steps = [
        math.hypot(b.x - a.x, b.y - a.y)
        for a, b in zip(track, track[1:])
    ]
    if not steps or sum(steps) / len(steps) < min_step:
        return False

    span = math.hypot(track[-1].x - track[0].x, track[-1].y - track[0].y)
    if span < min_span:
        return False

    # Directness: straight-line distance over path length. Jitter scores near 0.
    if span / (sum(steps) + 1e-6) < min_directness:
        return False

    return True


def bucket_frames(
    all_frames: list[int],
    flight_frames: set[int],
    reject_frames: set[int],
) -> dict[int, str]:
    """Assign every frame to positive / negative / discard.

    A frame is only NEGATIVE if nothing at all was proposed in it. Anything that
    produced a candidate we could not confirm goes to DISCARD and is never used
    as a training label in either direction.
    """
    out = {}
    for frame in all_frames:
        if frame in flight_frames:
            out[frame] = "positive"
        elif frame in reject_frames:
            out[frame] = "discard"
        else:
            out[frame] = "negative"
    return out
