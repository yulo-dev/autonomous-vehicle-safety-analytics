"""Hard braking: sustained longitudinal deceleration of the SDC."""

from __future__ import annotations

import numpy as np
import pandas as pd

from avsafety.detect.base import Event, segments, severity_descending, steps_for

EVENT_TYPE = "hard_braking"


def detect(sdc: pd.DataFrame, scenario_id: str, cfg: dict, dt: float) -> list[Event]:
    """`sdc` holds one track's states with kinematic features, one row per step."""
    sdc = sdc.sort_values("step").reset_index(drop=True)
    a = sdc["a_lon"].to_numpy(dtype=float)
    v = sdc["speed"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        mask = (a < cfg["decel_threshold_mps2"]) & (v > cfg["min_speed_mps"])

    have = np.flatnonzero(np.isfinite(a))
    first, last = (int(have[0]), int(have[-1])) if len(have) else (0, -1)
    events = []
    for s, e in segments(mask, steps_for(cfg["min_duration_s"], dt), steps_for(cfg["merge_gap_s"], dt)):
        window = a[s:e + 1]
        k = s + int(np.nanargmin(window))
        peak = float(a[k])
        events.append(Event(
            event_type=EVENT_TYPE,
            scenario_id=scenario_id,
            start_step=int(sdc.at[s, "step"]),
            end_step=int(sdc.at[e, "step"]),
            start_t_s=float(sdc.at[s, "t_s"]),
            end_t_s=float(sdc.at[e, "t_s"]),
            severity=severity_descending(peak, cfg["severity_bins_mps2"]),
            primary_track_id=int(sdc.at[s, "track_id"]),
            metrics={
                "peak_decel_mps2": round(peak, 3),
                "step_at_peak": int(sdc.at[k, "step"]),
                "speed_start_mps": round(float(v[s]), 3),
                "speed_end_mps": round(float(v[e]), 3),
                "delta_v_mps": round(float(v[s] - v[e]), 3),
                "duration_s": round((e - s + 1) * dt, 3),
                "truncated_start": s <= first,
                "truncated_end": e >= last,
            },
        ))
    return events
