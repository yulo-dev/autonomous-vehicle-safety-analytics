"""Derived kinematics from tracked states.

Acceleration comes from a Savitzky-Golay derivative of the reported velocity,
fitted separately on each contiguous run of valid states. Runs shorter than
the smoothing window get no derived values (NaN) rather than a noisy guess,
and smoothing never crosses an invalid gap.

Longitudinal acceleration is the acceleration vector projected onto the
direction of travel (smoothed velocity), falling back to box heading when the
agent is nearly stopped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

FEATURE_COLUMNS = ["vx_s", "vy_s", "speed", "ax", "ay", "a_lon"]


def window_steps(window_s: float, dt: float) -> int:
    n = int(round(window_s / dt))
    return n if n % 2 == 1 else n + 1


def _runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of contiguous True runs."""
    edges = np.diff(np.concatenate(([0], valid.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def track_features(track: pd.DataFrame, window: int, polyorder: int, dt: float) -> pd.DataFrame:
    """Compute features for one track (rows sorted by step)."""
    n = len(track)
    out = {c: np.full(n, np.nan) for c in FEATURE_COLUMNS}
    vx = track["vx"].to_numpy(dtype=float)
    vy = track["vy"].to_numpy(dtype=float)
    heading = track["heading"].to_numpy(dtype=float)
    for start, end in _runs(track["valid"].to_numpy()):
        if end - start < window:
            continue
        sl = slice(start, end)
        out["vx_s"][sl] = savgol_filter(vx[sl], window, polyorder)
        out["vy_s"][sl] = savgol_filter(vy[sl], window, polyorder)
        out["ax"][sl] = savgol_filter(vx[sl], window, polyorder, deriv=1, delta=dt)
        out["ay"][sl] = savgol_filter(vy[sl], window, polyorder, deriv=1, delta=dt)

    speed = np.hypot(out["vx_s"], out["vy_s"])
    moving = speed > 0.5
    ux = np.where(moving, out["vx_s"] / np.where(moving, speed, 1.0), np.cos(heading))
    uy = np.where(moving, out["vy_s"] / np.where(moving, speed, 1.0), np.sin(heading))
    out["speed"] = speed
    out["a_lon"] = out["ax"] * ux + out["ay"] * uy
    return track.assign(**out)


def scenario_features(states: pd.DataFrame, smoothing: dict, dt: float = 0.1) -> pd.DataFrame:
    """Features for every track in one scenario's states."""
    window = window_steps(smoothing["window_s"], dt)
    polyorder = smoothing["polyorder"]
    parts = [
        track_features(g.sort_values("step"), window, polyorder, dt)
        for _, g in states.groupby("track_id", sort=True)
    ]
    return pd.concat(parts, ignore_index=True) if parts else states.assign(**{c: [] for c in FEATURE_COLUMNS})
