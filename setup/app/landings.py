"""Count NEW shuttle landings once each, and call them against the traced lines.

WHY TRACK IDS RATHER THAN POSITION MATCHING
The first version decided "new shuttle" by asking whether a detection sat within
25 px of one already seen. That conflates three different things and inflates
the count badly:

  * a genuinely new shuttle arriving
  * the detector NEWLY NOTICING one that was already lying there
  * box jitter moving a detection more than the radius, so one shuttle
    registers as two

Measured on lines.mp4: 60 frames with ~3.5 boxes per frame contain only 5-6
distinct ByteTrack ids. The tracker already solves the association problem that
radius matching was failing at, so a landing is now keyed on a track id being
seen for the first time.

WHY THERE IS STILL A POSITION GUARD
A tracker that loses a shuttle for long enough will re-acquire it under a new
id, which would double-count. A new id whose position sits on top of one already
counted is therefore rejected. Track ids do the work; the radius is a backstop.

WHY SEEDING MATTERS MOST
`lines.mp4` already has shuttles on the floor when it starts, and a feeder
session always will after the first few shots. Without a seed period every one
of them counts as a landing the moment the model first finds it - which is what
made the old total meaningless. Everything present during the first
`seed_frames` is registered as pre-existing and scores nothing.

THE FOUR ZONES
Each landing is judged against BOTH traced lines independently, which is why a
shuttle inside both increments two counters:

    above the green (inside) line  -> green_outside, else green_inside
    above the red  (outside) line  -> red_outside,   else red_inside

"Above" means a smaller y: further from the camera, past the line.
"""
from __future__ import annotations

MATCH_RADIUS = 30.0      # px: backstop against a re-acquired track id
STABLE_FRAMES = 3        # sightings of a new id before it scores
SEED_FRAMES = 60         # ~2 s at 30 fps: whatever is already on the floor


class LandingCounter:
    """Count each arriving shuttle exactly once, and call it against two lines."""

    def __init__(self, lines: dict, match_radius: float = MATCH_RADIUS,
                 stable_frames: int = STABLE_FRAMES, seed_frames: int = SEED_FRAMES):
        self.lines = lines or {}
        self.match_radius = match_radius
        self.stable_frames = stable_frames
        self.seed_frames = seed_frames

        self.seen: dict[int, int] = {}          # track id -> sightings so far
        self.counted: set[int] = set()          # track ids already scored
        self.preexisting: set[int] = set()      # on the floor before we started
        self.positions: list[tuple[float, float]] = []   # counted + seeded
        self.frames = 0

        self.total = 0
        self.counts = {"green_inside": 0, "green_outside": 0,
                       "red_inside": 0, "red_outside": 0}
        # The shuttle that landed most recently. Exactly one is active: when the
        # next lands it takes over. With dozens on the court the useful question
        # is never "which of these" but "which just arrived".
        self.active: dict | None = None

    # ── helpers ──────────────────────────────────────────────
    @property
    def seeding(self) -> bool:
        return self.frames < self.seed_frames

    def _near_counted(self, point) -> bool:
        px, py = point
        r2 = self.match_radius ** 2
        return any((px - x) ** 2 + (py - y) ** 2 <= r2 for x, y in self.positions)

    def classify(self, point) -> dict:
        """Which side of each traced line this landing fell on."""
        x, y = point
        out: dict[str, str] = {}
        for role, colour in (("inside", "green"), ("outside", "red")):
            fit = self.lines.get(role)
            if not fit:
                continue
            out[colour] = "outside" if y < fit["a"] * x + fit["b"] else "inside"
        return out

    # ── main entry ───────────────────────────────────────────
    def update(self, tracks, frame_idx: int | None = None) -> list[dict]:
        """Feed this frame's [(track_id, (x, y))]; return newly counted landings.

        `track_id` None means the tracker produced no id for that box, which is
        treated as not-yet-trackable rather than as a new shuttle.
        """
        self.frames += 1
        idx = self.frames if frame_idx is None else frame_idx
        confirmed: list[dict] = []

        for item in tracks:
            # Callers pass (track_id, point) or (track_id, point, box); the box
            # is only for drawing, so take the first two and ignore the rest.
            tid, point = item[0], item[1]
            bbox = item[2] if len(item) > 2 else None
            if tid is None:
                continue

            if self.seeding:
                # Already on the floor before we started watching.
                self.preexisting.add(tid)
                if not self._near_counted(point):
                    self.positions.append(point)
                continue

            if tid in self.preexisting or tid in self.counted:
                continue

            self.seen[tid] = self.seen.get(tid, 0) + 1
            if self.seen[tid] < self.stable_frames:
                continue

            # A re-acquired id lands on a shuttle already counted.
            if self._near_counted(point):
                self.counted.add(tid)
                continue

            self.counted.add(tid)
            self.positions.append(point)
            self.total += 1
            sides = self.classify(point)
            for colour, side in sides.items():
                self.counts[f"{colour}_{side}"] += 1
            event = {"id": self.total, "track": tid, "frame": idx,
                     "px": [round(point[0], 1), round(point[1], 1)],
                     # Kept so the active box can be drawn on frames where the
                     # detector does not find this shuttle - det=0 between
                     # inferences is common, and a box that blinks out cannot
                     # be used to judge a call.
                     "box": list(bbox) if bbox else None, **sides}
            self.active = event
            confirmed.append(event)

        return confirmed

    def state_of(self, tid) -> str:
        """"active" | "counted" | "pending" for one track id."""
        if self.active is not None and tid == self.active.get("track"):
            return "active"
        if tid in self.counted or tid in self.preexisting:
            return "counted"
        return "pending"

    def reset(self) -> None:
        """New rally: forget everything, including the seed period."""
        self.seen.clear()
        self.counted.clear()
        self.preexisting.clear()
        self.positions.clear()
        self.frames = 0
        self.total = 0
        self.active = None
        for k in self.counts:
            self.counts[k] = 0
