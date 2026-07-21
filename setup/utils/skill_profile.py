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
