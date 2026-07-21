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


from config import settings as _cfg


def default_config() -> dict:
    return {
        "refs": _cfg.SKILL_REFS,
        "weights": _cfg.SKILL_FAMILY_WEIGHTS,
        "family_min": _cfg.SKILL_FAMILY_MIN_SAMPLES,
        "tiers": _cfg.SKILL_TIERS,
        "min_shots": _cfg.SKILL_MIN_SHOTS,
    }


def _norm(value, ref):
    t = ref["type"]
    if t == "monotonic":
        lo, hi = ref["lo"], ref["hi"]
        frac = 0.0 if hi == lo else (value - lo) / (hi - lo)
        frac = max(0.0, min(1.0, frac))
        if ref.get("invert"):
            frac = 1.0 - frac
        return 100.0 * frac
    if t == "target":
        d = abs(value - ref["target"])
        return 100.0 * max(0.0, 1.0 - d / ref["tol"])
    if t == "consistency":
        hi = ref["hi"]
        frac = 0.0 if hi == 0 else 1.0 - (value / hi)
        return 100.0 * max(0.0, min(1.0, frac))
    return 0.0


def _mean(stats, key):
    return RunningStat.from_dict(stats.get(key)).mean()


def _std(stats, key):
    return RunningStat.from_dict(stats.get(key)).std()


def evaluate_rubric(profile: dict, config: dict | None = None) -> dict:
    cfg = config or default_config()
    refs = cfg["refs"]
    stats = profile.get("stats", {})
    counts = profile.get("sampleCounts", {})
    shots = profile.get("shots", 0)
    scores = profile.get("scores", 0)

    breakdown = {}

    def score(metric, value):
        s = _norm(value, refs[metric])
        breakdown[metric] = round(s, 1)
        return s

    # movement
    move_parts = []
    if counts.get("move", 0) > 0:
        move_parts.append(score("move_speed", _mean(stats, "move_speed")))
        move_parts.append(score("coverage", len(profile.get("coverage", []))))
        if RunningStat.from_dict(stats.get("reaction")).n > 0:
            move_parts.append(score("reaction", _mean(stats, "reaction")))
    # accuracy
    acc_parts = []
    if shots > 0:
        acc_parts.append(score("accuracy", 100.0 * scores / shots))
    # stroke
    stroke_parts = []
    if RunningStat.from_dict(stats.get("swing_peak")).n > 0:
        stroke_parts.append(score("swing_consistency", _std(stats, "swing_peak")))
    # posture
    post_parts = []
    if counts.get("posture", 0) > 0:
        if RunningStat.from_dict(stats.get("knee")).n > 0:
            post_parts.append(score("knee", _mean(stats, "knee")))
        if RunningStat.from_dict(stats.get("stance")).n > 0:
            post_parts.append(score("stance", _mean(stats, "stance")))
        if RunningStat.from_dict(stats.get("knee")).n > 0:
            post_parts.append(score("posture_consistency", _std(stats, "knee")))

    families = {
        "move": sum(move_parts) / len(move_parts) if move_parts else None,
        "accuracy": sum(acc_parts) / len(acc_parts) if acc_parts else None,
        "stroke": sum(stroke_parts) / len(stroke_parts) if stroke_parts else None,
        "posture": sum(post_parts) / len(post_parts) if post_parts else None,
    }

    def _families_out():
        return {k: (round(v, 1) if v is not None else None)
                for k, v in families.items()}

    # ---- min-data gate ----
    if shots < cfg["min_shots"]:
        return {"composite": None, "tier": "Unranked",
                "families": _families_out(), "breakdown": breakdown}

    # ---- data-weighted composite (redistribute thin families) ----
    fam_min = cfg["family_min"]
    total_w = 0.0
    acc = 0.0
    for fam, fam_score in families.items():
        if fam_score is None:
            continue
        if fam == "accuracy":
            if shots < fam_min.get("accuracy", 0):
                continue
        elif counts.get(fam, 0) < fam_min.get(fam, 0):
            continue
        w = cfg["weights"].get(fam, 0.0)
        acc += w * fam_score
        total_w += w

    if total_w == 0:
        return {"composite": None, "tier": "Unranked",
                "families": _families_out(), "breakdown": breakdown}

    composite = acc / total_w
    tier = "Beginner"
    for cutoff, name in cfg["tiers"]:
        if composite >= cutoff:
            tier = name
            break

    return {"composite": round(composite, 1), "tier": tier,
            "families": _families_out(), "breakdown": breakdown}
