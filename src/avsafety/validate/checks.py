"""Data quality checks over ingested Parquet tables.

Each check returns failures only; the runner adds one summary row per check so
that "passed with 0 failures" is recorded explicitly, not inferred from silence.

Scenario-level errors make a scenario ineligible for detection and exposure.
Track-level errors exclude that track from detection; if the track is the SDC,
the failure escalates to the scenario (`sdc_track_error`). Warnings never
exclude data; they lower event dq_confidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Failure:
    check_name: str
    entity_type: str  # scenario | track
    scenario_id: str
    track_id: int | None
    level: str  # error | warn
    details: dict = field(default_factory=dict)

    @property
    def entity_id(self) -> str:
        return self.scenario_id if self.track_id is None else f"{self.scenario_id}:{self.track_id}"


@dataclass(frozen=True)
class CheckSpec:
    name: str
    entity_type: str
    level: str
    description: str


CHECKS: dict[str, CheckSpec] = {c.name: c for c in [
    CheckSpec("duplicate_scenario_id", "scenario", "error", "scenario_id appears more than once"),
    CheckSpec("step_count_mismatch", "scenario", "error", "a track's state count differs from n_steps"),
    CheckSpec("timestamps_irregular", "scenario", "error", "timestamps missing, non-monotonic, or off 10 Hz"),
    CheckSpec("sdc_low_validity", "scenario", "error", "SDC valid-frame fraction below threshold"),
    CheckSpec("sdc_track_error", "scenario", "error", "SDC track failed an error-level track check"),
    CheckSpec("sdc_stationary", "scenario", "warn", "SDC barely moves; contributes ~0 distance exposure"),
    CheckSpec("missing_values", "track", "error", "NaN/inf in a valid state"),
    CheckSpec("implausible_dimensions", "track", "error", "box length/width outside physical range"),
    CheckSpec("implausible_speed", "track", "error", "reported speed above physical limit"),
    CheckSpec("position_jump", "track", "error", "consecutive positions imply speed above limit"),
    CheckSpec("speed_inconsistency", "track", "warn", "reported speed disagrees with position-derived speed"),
    CheckSpec("heading_out_of_range", "track", "warn", "heading outside [-pi, pi]"),
    CheckSpec("unknown_object_type", "track", "warn", "object type unset or unknown"),
    CheckSpec("tracking_gaps", "track", "warn", "invalid states between first and last valid state"),
]}


def _track_failures(
    name: str, mask: pd.Series, states: pd.DataFrame, value_col: str | None = None
) -> list[Failure]:
    """Aggregate per-step failures into one Failure per track."""
    bad = states.loc[mask]
    if bad.empty:
        return []
    spec = CHECKS[name]
    out = []
    agg = {"n_steps": ("step", "size"), "first_step": ("step", "min")}
    if value_col is not None:
        agg["max_value"] = (value_col, "max")
    for (sid, tid), row in bad.groupby(["scenario_id", "track_id"], observed=True).agg(**agg).iterrows():
        details = {k: (round(float(v), 3) if isinstance(v, float) else int(v)) for k, v in row.items()}
        out.append(Failure(name, "track", str(sid), int(tid), spec.level, details))
    return out


# ---------------------------------------------------------------- scenario checks

def check_duplicate_ids(scenarios: pd.DataFrame, cfg: dict) -> list[Failure]:
    dup = scenarios[scenarios["scenario_id"].duplicated(keep=False)]
    return [
        Failure("duplicate_scenario_id", "scenario", sid, None, "error",
                {"occurrences": len(g), "sources": sorted(set(g["source_file"] + ":" + g["record_index"].astype(str)))})
        for sid, g in dup.groupby("scenario_id")
    ]


def check_step_counts(scenarios: pd.DataFrame, states: pd.DataFrame, cfg: dict) -> list[Failure]:
    counts = states.groupby(["scenario_id", "track_id"], observed=True).size().rename("n").reset_index()
    expected = scenarios.drop_duplicates("scenario_id").set_index("scenario_id")["n_steps"]
    counts["expected"] = counts["scenario_id"].astype(str).map(expected)
    bad = counts[counts["n"] != counts["expected"]]
    return [
        Failure("step_count_mismatch", "scenario", str(sid), None, "error",
                {"n_tracks_affected": len(g), "expected": int(g["expected"].iloc[0]),
                 "observed": sorted(int(x) for x in g["n"].unique())})
        for sid, g in bad.groupby("scenario_id", observed=True)
    ]


def check_timestamps(states: pd.DataFrame, cfg: dict) -> list[Failure]:
    sdc = states[states["is_sdc"]].sort_values(["scenario_id", "step"])
    out = []
    lo = cfg["expected_dt_s"] - cfg["dt_tolerance_s"]
    hi = cfg["expected_dt_s"] + cfg["dt_tolerance_s"]
    for sid, g in sdc.groupby("scenario_id", observed=True):
        t = g["t_s"].to_numpy()
        dt = np.diff(t)
        n_nan = int(np.isnan(t).sum())
        n_bad = int(((dt < lo) | (dt > hi)).sum()) if len(dt) else 0
        if n_nan or n_bad:
            out.append(Failure("timestamps_irregular", "scenario", str(sid), None, "error",
                               {"n_nan": n_nan, "n_bad_intervals": n_bad,
                                "min_dt": float(np.nanmin(dt)) if len(dt) else None}))
    return out


def check_sdc_validity(scenarios: pd.DataFrame, cfg: dict) -> list[Failure]:
    out = []
    t = cfg["sdc_valid_frac"]
    for _, r in scenarios.drop_duplicates("scenario_id").iterrows():
        frac = float(r["sdc_valid_frac"])
        if frac < t["warn_below"]:
            level = "error" if frac < t["error_below"] else "warn"
            out.append(Failure("sdc_low_validity", "scenario", r["scenario_id"], None, level,
                               {"sdc_valid_frac": round(frac, 4)}))
        if float(r["sdc_distance_m"]) < cfg["sdc_stationary_distance_m"]:
            out.append(Failure("sdc_stationary", "scenario", r["scenario_id"], None, "warn",
                               {"sdc_distance_m": round(float(r["sdc_distance_m"]), 3)}))
    return out


# ---------------------------------------------------------------- track checks

def check_track_values(states: pd.DataFrame, cfg: dict) -> list[Failure]:
    s = states.sort_values(["scenario_id", "track_id", "step"]).reset_index(drop=True)
    valid = s["valid"].to_numpy()
    v = s[valid]
    out: list[Failure] = []

    numeric = ["x", "y", "vx", "vy", "heading", "length", "width"]
    nonfinite = ~np.isfinite(v[numeric].to_numpy(dtype=float)).all(axis=1)
    out += _track_failures("missing_values", pd.Series(nonfinite, index=v.index), v)

    d = cfg["dimensions"]
    dims_bad = (
        (v["length"] < d["min_m"]) | (v["length"] > d["max_length_m"])
        | (v["width"] < d["min_m"]) | (v["width"] > d["max_width_m"])
    )
    out += _track_failures("implausible_dimensions", dims_bad, v, "length")

    speed = np.hypot(v["vx"].astype(float), v["vy"].astype(float))
    vv = v.assign(speed=speed)
    out += _track_failures("implausible_speed", vv["speed"] > cfg["max_speed_mps"], vv, "speed")

    out += _track_failures("heading_out_of_range", v["heading"].abs() > np.pi + 1e-3, v, "heading")

    # Consecutive valid-valid pairs within a track.
    same_track = (s["scenario_id"] == s["scenario_id"].shift()) & (s["track_id"] == s["track_id"].shift())
    pair = same_track & (s["step"] - s["step"].shift() == 1) & s["valid"] & s["valid"].shift(fill_value=False)
    dt = (s["t_s"] - s["t_s"].shift()).where(pair)
    disp = np.hypot(s["x"] - s["x"].shift(), s["y"] - s["y"].shift()).where(pair)
    implied = (disp / dt).where(dt > 0)
    rep = np.hypot(s["vx"].astype(float), s["vy"].astype(float))
    rep_mid = 0.5 * (rep + rep.shift())
    ss = s.assign(implied_speed=implied, mismatch=(implied - rep_mid).abs())

    out += _track_failures("position_jump", ss["implied_speed"] > cfg["max_speed_mps"], ss, "implied_speed")

    mism = _track_failures("speed_inconsistency", ss["mismatch"] > cfg["max_speed_mismatch_mps"], ss, "mismatch")
    out += [f for f in mism if f.details["n_steps"] >= cfg["speed_mismatch_min_steps"]]

    otype = s.drop_duplicates(["scenario_id", "track_id"])
    unknown = otype[otype["object_type"].isin(["unset", "unknown"])]
    out += [Failure("unknown_object_type", "track", str(r.scenario_id), int(r.track_id), "warn",
                    {"object_type": r.object_type}) for r in unknown.itertuples()]

    # Gaps: invalid states strictly between first and last valid state.
    vs = s[s["valid"]].groupby(["scenario_id", "track_id"], observed=True)["step"].agg(["min", "max", "size"])
    vs["gap_steps"] = (vs["max"] - vs["min"] + 1) - vs["size"]
    for (sid, tid), r in vs[vs["gap_steps"] > 0].iterrows():
        out.append(Failure("tracking_gaps", "track", str(sid), int(tid), "warn",
                           {"gap_steps": int(r["gap_steps"]), "valid_steps": int(r["size"])}))
    return out


def escalate_sdc(failures: list[Failure], states: pd.DataFrame) -> list[Failure]:
    sdc_ids = (
        states.loc[states["is_sdc"], ["scenario_id", "track_id"]]
        .drop_duplicates()
        .astype({"scenario_id": str})
    )
    sdc = set(zip(sdc_ids["scenario_id"], sdc_ids["track_id"].astype(int), strict=True))
    by_scenario: dict[str, list[str]] = {}
    for f in failures:
        if f.entity_type == "track" and f.level == "error" and (f.scenario_id, f.track_id) in sdc:
            by_scenario.setdefault(f.scenario_id, []).append(f.check_name)
    return [Failure("sdc_track_error", "scenario", sid, None, "error", {"checks": sorted(set(names))})
            for sid, names in by_scenario.items()]


STATE_CHECKS: list[Callable[[pd.DataFrame, dict], list[Failure]]] = [check_timestamps, check_track_values]


def run_all(scenarios: pd.DataFrame, states: pd.DataFrame, cfg: dict) -> list[Failure]:
    """Run every check on one batch (typically one shard)."""
    failures = check_sdc_validity(scenarios, cfg) + check_step_counts(scenarios, states, cfg)
    for fn in STATE_CHECKS:
        failures += fn(states, cfg)
    failures += escalate_sdc(failures, states)
    return failures
