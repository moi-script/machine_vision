"""Count NEW shuttle landings once each, and call them against the traced lines.

WHY A REGISTRY RATHER THAN A PER-FRAME COUNT
Measured over lines.mp4: 1 detection at frame 300, 210 by frame 500, 5140 by
frame 1140. A feeder session buries the court, so "how many shuttles are in this
frame" says nothing about scoring. What matters is which shuttle just arrived,
so every detection is matched against shuttles already seen; a match inside
MATCH_RADIUS is old and is ignored from then on. Over the whole clip this turned
5140 detections into 47 landings.

WHY A NEW LANDING MUST PERSIST
The landed-shuttle detector's test precision is 0.752 - roughly one detection in
four on unseen frames is spurious - so a single-frame flicker would otherwise
score a point. A candidate has to hold still in about the same place for
CONFIRM_FRAMES consecutive frames before it counts.

THE FOUR ZONES
Each landing is judged against BOTH traced lines independently, which is why a
shuttle inside both increments two counters:

    above the green (inside) line  -> green_outside, else green_inside
    above the red  (outside) line  -> red_outside,   else red_inside

"Above" means a smaller y, i.e. further from the camera and past the line.
"""
from __future__ import annotations

MATCH_RADIUS = 25.0      # px: this close to a known shuttle IS that shuttle
CONFIRM_FRAMES = 4       # consecutive sightings before a candidate scores


class LandingCounter:
    """Retire counted shuttles; classify each new one against two lines."""

    def __init__(self, lines: dict, match_radius: float = MATCH_RADIUS,
                 confirm_frames: int = CONFIRM_FRAMES):
        # {"inside": {"a":..,"b":..}, "outside": {...}} from the calibration.
        self.lines = lines or {}
        self.match_radius = match_radius
        self.confirm_frames = confirm_frames
        self.retired: list[tuple[float, float]] = []
        self.pending: list[dict] = []
        self.total = 0
        self.counts = {"green_inside": 0, "green_outside": 0,
                       "red_inside": 0, "red_outside": 0}
        # The shuttle that landed most recently. Exactly one is "active" at a
        # time: when the next one lands it takes over and this one becomes just
        # another retired position. That is what makes scoring readable once
        # dozens of shuttles are lying on the court - the question is never
        # "which of these 50" but "which one just arrived".
        self.active: dict | None = None

    def _near(self, entries, point) -> bool:
        px, py = point
        return any((px - ex) ** 2 + (py - ey) ** 2 <= self.match_radius ** 2
                   for ex, ey in entries)

    def classify(self, point) -> dict:
        """Which side of each line this landing fell on."""
        x, y = point
        out: dict[str, str] = {}
        for role, colour in (("inside", "green"), ("outside", "red")):
            fit = self.lines.get(role)
            if not fit:
                continue
            line_y = fit["a"] * x + fit["b"]
            out[colour] = "outside" if y < line_y else "inside"
        return out

    def update(self, points, frame_idx: int) -> list[dict]:
        """Feed this frame's landing points; return newly confirmed landings."""
        confirmed: list[dict] = []

        for point in points:
            if self._near(self.retired, point):
                continue                      # already counted, never again

            cand = None
            for c in self.pending:
                if (c["point"][0] - point[0]) ** 2 + (c["point"][1] - point[1]) ** 2 \
                        <= self.match_radius ** 2:
                    cand = c
                    break
            if cand is None:
                self.pending.append({"point": point, "hits": 1, "last": frame_idx})
                continue

            cand["hits"] += 1
            cand["last"] = frame_idx
            cand["point"] = point             # a settling shuttle drifts a little

            if cand["hits"] >= self.confirm_frames:
                self.pending.remove(cand)
                self.retired.append(point)
                self.total += 1
                sides = self.classify(point)
                for colour, side in sides.items():
                    self.counts[f"{colour}_{side}"] += 1
                event = {"id": self.total, "frame": frame_idx,
                         "px": [round(point[0], 1), round(point[1], 1)],
                         **sides}
                # Hand over: the previous active shuttle is now just history.
                self.active = event
                confirmed.append(event)

        # A candidate that stopped being detected was noise, not a landing.
        self.pending = [c for c in self.pending
                        if frame_idx - c["last"] <= self.confirm_frames]
        return confirmed

    def state_of(self, point) -> str:
        """"active" | "counted" | "pending" — what to draw for this detection.

        Checked active-first: the active shuttle is also in `retired`, since it
        has been counted, and reporting it as merely counted would leave nothing
        highlighted.
        """
        if self.active is not None:
            ax, ay = self.active["px"]
            if (ax - point[0]) ** 2 + (ay - point[1]) ** 2 <= self.match_radius ** 2:
                return "active"
        if self._near(self.retired, point):
            return "counted"
        return "pending"

    def reset(self) -> None:
        self.retired.clear()
        self.pending.clear()
        self.active = None
        self.total = 0
        for k in self.counts:
            self.counts[k] = 0
