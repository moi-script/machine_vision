"""Per-player skill accumulation + rule-based rubric.

RunningStat accumulates count/sum/sum-of-squares so per-session aggregates can
be merged into a cumulative career profile without keeping raw samples."""
from __future__ import annotations
import math


class RunningStat:
    def __init__(self, n: int = 0, sum: float = 0.0, sumsq: float = 0.0):
        self.n = n
        self.sum = sum
        self.sumsq = sumsq

    def add(self, x: float) -> None:
        x = float(x)
        self.n += 1
        self.sum += x
        self.sumsq += x * x

    def merge(self, other: "RunningStat") -> None:
        self.n += other.n
        self.sum += other.sum
        self.sumsq += other.sumsq

    def mean(self) -> float:
        return self.sum / self.n if self.n else 0.0

    def var(self) -> float:
        if self.n == 0:
            return 0.0
        m = self.mean()
        return max(0.0, self.sumsq / self.n - m * m)

    def std(self) -> float:
        return math.sqrt(self.var())

    def to_dict(self) -> dict:
        return {"n": self.n, "sum": self.sum, "sumsq": self.sumsq}

    @classmethod
    def from_dict(cls, d: dict) -> "RunningStat":
        d = d or {}
        return cls(int(d.get("n", 0)), float(d.get("sum", 0.0)),
                   float(d.get("sumsq", 0.0)))


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class SkillAccumulator:
    """Accumulates one drill session's skill metrics for one athlete."""

    def __init__(self, kp_conf, bbox_min_h, react_dist, swing_speed):
        self.kp_conf = kp_conf
        self.bbox_min_h = bbox_min_h
        self.react_dist = react_dist
        self.swing_speed = swing_speed

        self.stats = {k: RunningStat() for k in
                      ("move_speed", "reaction", "knee", "stance", "torso",
                       "swing_peak")}
        self.coverage = set()
        self.shots = 0
        self.scores = 0
        self.swings = 0
        self.counts = {"move": 0, "posture": 0, "stroke": 0, "accuracy": 0}

        # temporal state
        self._prev_ankle = None
        self._prev_now = None
        self._prev_kps = None
        self._stroke_prev_now = None
        # pending reaction: (feeder_ts, baseline_ankle) or None
        self._pending = None
        # swing edge state
        self._in_swing = False
        self._cur_peak = 0.0

    # ---- event hooks ----
    def on_feeder_fired(self, now, zone):
        self._pending = (now, self._prev_ankle)

    def on_shot(self):
        self.shots += 1
        self.counts["accuracy"] += 1

    def on_score(self):
        self.scores += 1

    # ---- per-frame ----
    def add_frame(self, keypoints, box, court_ankle, now):
        from utils import pose_features as pf
        from utils.zones import get_zone_from_position
        bbox_h = pf.bbox_height(box)

        # ---- Tier A: movement (always, needs court position) ----
        if court_ankle is not None:
            zone = get_zone_from_position(court_ankle[0], court_ankle[1])
            if zone is not None:
                self.coverage.add(zone)
            if self._prev_ankle is not None and self._prev_now is not None:
                dt = now - self._prev_now
                if dt > 0:
                    speed = _dist(court_ankle, self._prev_ankle) / dt
                    self.stats["move_speed"].add(speed)
                    self.counts["move"] += 1
            # reaction: displacement from feeder-time baseline
            if self._pending is not None:
                _ts, base = self._pending
                if base is not None and _dist(court_ankle, base) >= self.react_dist:
                    self.stats["reaction"].add(now - _ts)
                    self._pending = None
            self._prev_ankle = court_ankle
            self._prev_now = now

        near = bbox_h >= self.bbox_min_h

        # ---- Tier B: posture (gated by distance) ----
        if near:
            sampled = False
            knee = pf.knee_angle(keypoints, self.kp_conf)
            if knee is not None:
                self.stats["knee"].add(knee); sampled = True
            stance = pf.stance_width(keypoints, self.kp_conf)
            if stance is not None:
                self.stats["stance"].add(stance); sampled = True
            torso = pf.torso_lean(keypoints, self.kp_conf)
            if torso is not None:
                self.stats["torso"].add(torso); sampled = True
            if sampled:
                self.counts["posture"] += 1

        # ---- Tier C: stroke (gated by distance) ----
        if near:
            self._update_swing(keypoints, now)

        self._prev_kps = keypoints

    def _update_swing(self, keypoints, now):
        from utils import pose_features as pf
        if self._prev_kps is None or self._stroke_prev_now is None:
            self._stroke_prev_now = now
            return
        dt = now - self._stroke_prev_now
        self._stroke_prev_now = now
        if dt <= 0:
            return
        ws = pf.max_wrist_speed(keypoints, self._prev_kps, dt, self.kp_conf)
        if ws is None:
            return
        self.counts["stroke"] += 1
        if not self._in_swing and ws >= self.swing_speed:
            self._in_swing = True
            self._cur_peak = ws
        elif self._in_swing:
            self._cur_peak = max(self._cur_peak, ws)
            if ws < self.swing_speed:
                self.stats["swing_peak"].add(self._cur_peak)
                self.swings += 1
                self._in_swing = False

    # ---- output ----
    def snapshot(self) -> dict:
        return {
            "sampleCounts": dict(self.counts),
            "coverage": sorted(self.coverage),
            "shots": self.shots,
            "scores": self.scores,
            "swings": self.swings,
            "stats": {k: v.to_dict() for k, v in self.stats.items()},
        }

    @staticmethod
    def merge_snapshot(cumulative: dict, snap: dict) -> dict:
        cumulative = cumulative or {}
        out = {
            "sampleCounts": dict(cumulative.get("sampleCounts", {})),
            "coverage": set(cumulative.get("coverage", [])),
            "shots": cumulative.get("shots", 0),
            "scores": cumulative.get("scores", 0),
            "swings": cumulative.get("swings", 0),
            "stats": {},
        }
        for fam, c in snap["sampleCounts"].items():
            out["sampleCounts"][fam] = out["sampleCounts"].get(fam, 0) + c
        out["coverage"].update(snap.get("coverage", []))
        out["shots"] += snap.get("shots", 0)
        out["scores"] += snap.get("scores", 0)
        out["swings"] += snap.get("swings", 0)
        keys = set(cumulative.get("stats", {})) | set(snap.get("stats", {}))
        for k in keys:
            a = RunningStat.from_dict(cumulative.get("stats", {}).get(k))
            a.merge(RunningStat.from_dict(snap.get("stats", {}).get(k)))
            out["stats"][k] = a.to_dict()
        out["coverage"] = sorted(out["coverage"])
        return out
