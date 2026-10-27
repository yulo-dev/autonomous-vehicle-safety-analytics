"""Low time-to-collision conflicts between the SDC and other road users.

Geometry: each agent's box is covered by three discs spaced along its heading
(a standard approximation that tracks elongated vehicles far better than one
circle). TTC is the earliest time, under constant velocity and heading, at
which any SDC disc touches any counterpart disc.

Boxes already overlapping at t=0 are treated as tracking artifacts and skipped:
WOMD has essentially no contact events, and a real collision needs its own
detector rather than a TTC of zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from avsafety.detect.base import Event, segments, severity_descending, steps_for

EVENT_TYPE = "low_ttc_conflict"
_OFFSETS = np.array([-1.0, 0.0, 1.0]) / 3.0  # disc centers at -L/3, 0, +L/3


def discs(x, y, heading, length, width):
    """Disc centers (..., 3, 2) and radii (...,) for boxes broadcast over leading dims."""
    x, y, heading, length, width = (np.asarray(a, dtype=float) for a in (x, y, heading, length, width))
    off = length[..., None] * _OFFSETS
    cx = x[..., None] + off * np.cos(heading)[..., None]
    cy = y[..., None] + off * np.sin(heading)[..., None]
    radius = np.sqrt((length / 6.0) ** 2 + (width / 2.0) ** 2)
    return np.stack([cx, cy], axis=-1), radius


def ttc_grid(sdc: dict, others: dict, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    """TTC from the SDC to N counterparts over T steps.

    `sdc` fields have shape (T,), `others` fields (T, N); NaN marks a missing
    state. Returns (ttc, overlap), both (T, N): ttc is inf when there is no
    contact within the horizon or data is missing, and nan where the boxes
    already overlap at t=0.
    """
    c_s, r_s = discs(sdc["x"], sdc["y"], sdc["heading"], sdc["length"], sdc["width"])      # (T,3,2), (T,)
    c_o, r_o = discs(others["x"], others["y"], others["heading"], others["length"], others["width"])  # (T,N,3,2)
    w = np.stack([others["vx"] - sdc["vx"][:, None], others["vy"] - sdc["vy"][:, None]], axis=-1)  # (T,N,2)

    rel = c_o[:, :, None, :, :] - c_s[:, None, :, None, :]   # (T,N,3,3,2): every disc pair
    big_r = (r_s[:, None] + r_o)[..., None, None]            # (T,N,1,1)
    a = (w * w).sum(-1)[..., None, None]                     # (T,N,1,1)
    b = 2.0 * (rel * w[:, :, None, None, :]).sum(-1)         # (T,N,3,3)
    c = (rel * rel).sum(-1) - big_r**2                       # (T,N,3,3)

    with np.errstate(invalid="ignore", divide="ignore"):
        overlap = (c <= 0).any(axis=(-2, -1))
        disc = b * b - 4.0 * a * c
        t = (-b - np.sqrt(disc)) / (2.0 * a)
        hit = (c > 0) & (b < 0) & (disc >= 0) & (a > 1e-9) & (t >= 0) & (t <= horizon)
    t = np.where(hit, t, np.inf).min(axis=(-2, -1))
    t[overlap] = np.nan
    return t, overlap


def ttc(sdc: dict, others: dict, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-step convenience wrapper: scalar SDC state vs N counterparts."""
    s = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in sdc.items()}
    o = {k: np.asarray(v, dtype=float)[None, :] for k, v in others.items()}
    t, overlap = ttc_grid(s, o, horizon)
    return t[0], overlap[0]


_FIELDS = ["x", "y", "heading", "length", "width", "vx_s", "vy_s", "speed"]


def detect(features: pd.DataFrame, sdc_track_id: int, scenario_id: str, cfg: dict, dt: float,
           excluded_tracks: set[int] | None = None) -> list[Event]:
    """`features` holds all tracks of one scenario with kinematic features."""
    excluded_tracks = excluded_tracks or set()
    f = features[np.isfinite(features["speed"].to_numpy(dtype=float))]
    sdc = f[f["track_id"] == sdc_track_id]
    others = f[(f["track_id"] != sdc_track_id)
               & f["object_type"].isin(cfg["counterpart_types"])
               & ~f["track_id"].isin(excluded_tracks)]
    if sdc.empty or others.empty:
        return []

    n_steps = int(features["step"].max()) + 1
    steps = np.arange(n_steps)
    sdc_frame = sdc.set_index("step")[_FIELDS].reindex(steps)
    track_ids = np.sort(others["track_id"].unique())
    oth = {k: others.pivot(index="step", columns="track_id", values=k).reindex(index=steps, columns=track_ids)
         .to_numpy(dtype=float) for k in _FIELDS}
    s = {k: sdc_frame[k].to_numpy(dtype=float) for k in _FIELDS}

    dx, dy = oth["x"] - s["x"][:, None], oth["y"] - s["y"][:, None]
    dist = np.hypot(dx, dy)
    with np.errstate(invalid="ignore"):
        active = (s["speed"][:, None] >= cfg["min_sdc_speed_mps"]) & (dist <= cfg["max_range_m"])
    rvx, rvy = oth["vx_s"] - s["vx_s"][:, None], oth["vy_s"] - s["vy_s"][:, None]
    closing = -(dx * rvx + dy * rvy) / np.maximum(dist, 1e-6)

    grid, _ = ttc_grid(
        {"x": s["x"], "y": s["y"], "heading": s["heading"], "length": s["length"], "width": s["width"],
         "vx": s["vx_s"], "vy": s["vy_s"]},
        {"x": oth["x"], "y": oth["y"], "heading": oth["heading"], "length": oth["length"], "width": oth["width"],
         "vx": oth["vx_s"], "vy": oth["vy_s"]},
        cfg["horizon_s"],
    )
    grid = np.where(active, grid, np.inf)

    sdc_steps = np.flatnonzero(np.isfinite(s["speed"]))
    first, last = int(sdc_steps.min()), int(sdc_steps.max())
    min_len = steps_for(cfg["min_duration_s"], dt)
    gap = steps_for(cfg["merge_gap_s"], dt)
    otype = others.drop_duplicates("track_id").set_index("track_id")["object_type"]
    t_by_step = features.drop_duplicates("step").set_index("step")["t_s"]
    events = []
    for j, tid in enumerate(track_ids):
        series = grid[:, j]
        with np.errstate(invalid="ignore"):
            mask = series < cfg["ttc_threshold_s"]
        for s0, e0 in segments(mask, min_len, gap):
            window = np.where(mask[s0:e0 + 1], series[s0:e0 + 1], np.nan)
            k = s0 + int(np.nanargmin(window))
            min_ttc = float(series[k])
            events.append(Event(
                event_type=EVENT_TYPE,
                scenario_id=scenario_id,
                start_step=s0,
                end_step=e0,
                start_t_s=float(t_by_step.get(s0, s0 * dt)),
                end_t_s=float(t_by_step.get(e0, e0 * dt)),
                severity=severity_descending(min_ttc, cfg["severity_bins_s"]),
                primary_track_id=int(sdc_track_id),
                counterpart_track_id=int(tid),
                counterpart_type=str(otype[tid]),
                metrics={
                    "min_ttc_s": round(min_ttc, 3),
                    "step_at_min": k,
                    "sdc_speed_at_min_mps": round(float(s["speed"][k]), 3),
                    "closing_speed_at_min_mps": round(float(closing[k, j]), 3),
                    "min_center_distance_m": round(float(np.nanmin(dist[s0:e0 + 1, j])), 3),
                    "duration_s": round((e0 - s0 + 1) * dt, 3),
                    "truncated_start": s0 <= first,
                    "truncated_end": e0 >= last,
                },
            ))
    return events


def pair_ttc_series(features: pd.DataFrame, sdc_track_id: int, counterpart_id: int, cfg: dict) -> pd.DataFrame:
    """Per-step TTC between the SDC and one counterpart, for drill-down plots.

    Applies the same gating as `detect` (SDC speed, range), so the plotted
    series is exactly what the detector thresholded.
    """
    f = features[np.isfinite(features["speed"].to_numpy(dtype=float))]
    steps = np.arange(int(features["step"].max()) + 1)
    s = f[f["track_id"] == sdc_track_id].set_index("step")[_FIELDS].reindex(steps)
    o = f[f["track_id"] == counterpart_id].set_index("step")[_FIELDS].reindex(steps)
    grid, overlap = ttc_grid(
        {"x": s["x"].to_numpy(), "y": s["y"].to_numpy(), "heading": s["heading"].to_numpy(),
         "length": s["length"].to_numpy(), "width": s["width"].to_numpy(),
         "vx": s["vx_s"].to_numpy(), "vy": s["vy_s"].to_numpy()},
        {k: o[src].to_numpy()[:, None] for k, src in
         [("x", "x"), ("y", "y"), ("heading", "heading"), ("length", "length"), ("width", "width"),
          ("vx", "vx_s"), ("vy", "vy_s")]},
        cfg["horizon_s"],
    )
    dist = np.hypot(o["x"].to_numpy() - s["x"].to_numpy(), o["y"].to_numpy() - s["y"].to_numpy())
    with np.errstate(invalid="ignore"):
        active = (s["speed"].to_numpy() >= cfg["min_sdc_speed_mps"]) & (dist <= cfg["max_range_m"])
    ttc_vals = np.where(active, grid[:, 0], np.inf)
    return pd.DataFrame({"step": steps, "ttc_s": ttc_vals, "overlap": overlap[:, 0]})
