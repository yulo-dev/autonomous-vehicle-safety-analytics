"""Shared detector types and helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Event:
    event_type: str
    scenario_id: str
    start_step: int
    end_step: int
    start_t_s: float
    end_t_s: float
    severity: str
    primary_track_id: int
    counterpart_track_id: int | None = None
    counterpart_type: str | None = None
    dq_confidence: float = 1.0
    metrics: dict = field(default_factory=dict)

    def event_id(self, detector_version: str) -> str:
        key = "|".join(str(x) for x in (
            self.scenario_id, self.event_type, self.primary_track_id,
            self.counterpart_track_id, self.start_step, detector_version,
        ))
        return hashlib.sha1(key.encode()).hexdigest()[:16]


def segments(mask: np.ndarray, min_len: int, merge_gap: int) -> list[tuple[int, int]]:
    """Inclusive (start, end) index ranges of True runs.

    Runs separated by at most `merge_gap` False samples are merged first, then
    runs shorter than `min_len` are dropped. Merging before the length filter
    means a brief threshold dip doesn't split one braking action into two
    events or erase it.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = list(np.flatnonzero(edges == 1))
    ends = list(np.flatnonzero(edges == -1) - 1)
    merged: list[list[int]] = [[starts[0], ends[0]]]
    for s, e in zip(starts[1:], ends[1:], strict=True):
        if s - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(int(s), int(e)) for s, e in merged if e - s + 1 >= min_len]


def steps_for(seconds: float, dt: float) -> int:
    return max(1, int(round(seconds / dt)))


def severity_descending(value: float, bins: list[float]) -> str:
    """For metrics where smaller is worse (TTC, signed decel): bins ordered worse-ward."""
    s1_s2, s2_s3 = bins
    if value < s2_s3:
        return "S3"
    if value < s1_s2:
        return "S2"
    return "S1"
