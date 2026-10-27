"""Detection stage: eligible scenarios -> features -> events in the store."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from avsafety import db
from avsafety.detect import hard_braking, low_ttc
from avsafety.detect.base import Event
from avsafety.features.kinematics import scenario_features
from avsafety.validate.runner import (
    latest_validate_run,
    read_states,
    scenario_warnings,
    shard_dirs,
    track_flags,
)


class ConfigVersionError(RuntimeError):
    pass


def _guard_version(conn: sqlite3.Connection, version: str, cfg_hash: str) -> None:
    row = conn.execute(
        "SELECT config_hash FROM pipeline_runs WHERE stage = 'detect' AND version = ? "
        "AND config_hash != ? LIMIT 1", (version, cfg_hash)
    ).fetchone()
    if row:
        raise ConfigVersionError(
            f"detector version {version} already ran with a different config "
            f"({row['config_hash']} vs {cfg_hash}); bump `version` in detectors.yaml"
        )


def _dq_confidence(feats: pd.DataFrame, ev: Event, warn_tracks: set, warn_scenario: bool) -> float:
    """Data-quality confidence, not P(event is real).

    Base = share of valid states for the involved tracks within +/-1 s of the
    event; x0.8 if an involved track has a DQ warning; x0.9 if the scenario does.
    """
    lo, hi = ev.start_step - 10, ev.end_step + 10
    tracks = [ev.primary_track_id] + ([ev.counterpart_track_id] if ev.counterpart_track_id is not None else [])
    win = feats[feats["track_id"].isin(tracks) & feats["step"].between(lo, hi)]
    conf = float(win["valid"].mean()) if len(win) else 0.0
    if any((ev.scenario_id, t) in warn_tracks for t in tracks):
        conf *= 0.8
    if warn_scenario:
        conf *= 0.9
    return round(conf, 3)


def detect_scenario(states: pd.DataFrame, sdc_track_id: int, scenario_id: str, cfg: dict,
                    excluded_tracks: set[int]) -> tuple[list[Event], pd.DataFrame]:
    dt = float(np.nanmedian(np.diff(np.sort(states["t_s"].unique())))) or 0.1
    feats = scenario_features(states, cfg["smoothing"], dt)
    sdc = feats[feats["track_id"] == sdc_track_id]
    events = hard_braking.detect(sdc, scenario_id, cfg["hard_braking"], dt)
    events += low_ttc.detect(feats, sdc_track_id, scenario_id, cfg["low_ttc_conflict"], dt, excluded_tracks)
    return events, feats


def _insert(conn: sqlite3.Connection, run_id: str, version: str, events: list[Event]) -> int:
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO events (event_id, scenario_id, run_id, detector_version, event_type, "
        "start_step, end_step, start_t_s, end_t_s, severity, dq_confidence, primary_track_id, "
        "counterpart_track_id, counterpart_type, metrics) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(e.event_id(version), e.scenario_id, run_id, version, e.event_type, e.start_step, e.end_step,
          e.start_t_s, e.end_t_s, e.severity, e.dq_confidence, e.primary_track_id,
          e.counterpart_track_id, e.counterpart_type, json.dumps(e.metrics)) for e in events],
    )
    return conn.total_changes - before


def run_detection(processed_root: str | Path, conn: sqlite3.Connection, cfg: dict) -> dict:
    validate_run = latest_validate_run(conn)
    if validate_run is None:
        raise RuntimeError("no completed validation run; run `avsafety validate` first")

    version, cfg_hash = str(cfg["version"]), db.config_hash(cfg)
    _guard_version(conn, version, cfg_hash)
    run_id = db.start_run(conn, "detect", cfg_hash, version=version)

    eligible = {r["scenario_id"] for r in conn.execute(
        "SELECT scenario_id FROM scenario_eligibility WHERE eligible = 1")}
    sdc_ids = dict(conn.execute("SELECT scenario_id, sdc_track_id FROM scenarios").fetchall())
    error_tracks, warn_tracks = track_flags(conn, validate_run)
    warn_scenarios = scenario_warnings(conn, validate_run)

    counts: Counter[str] = Counter()
    n_scenarios = inserted = 0
    for shard in shard_dirs(processed_root):
        states = read_states(shard)
        states = states[states["scenario_id"].isin(eligible)]
        for sid, g in states.groupby("scenario_id", observed=True):
            sid = str(sid)
            excluded = {t for (s, t) in error_tracks if s == sid}
            events, feats = detect_scenario(g, int(sdc_ids[sid]), sid, cfg, excluded)
            for ev in events:
                ev.dq_confidence = _dq_confidence(feats, ev, warn_tracks, sid in warn_scenarios)
                counts[ev.event_type] += 1
            inserted += _insert(conn, run_id, version, events)
            n_scenarios += 1
        conn.commit()

    summary = {
        "detector_version": version,
        "config": cfg,
        "validate_run_id": validate_run,
        "scenarios_processed": n_scenarios,
        "events_detected": dict(counts),
        "events_inserted": inserted,
        "events_already_present": sum(counts.values()) - inserted,
    }
    db.finish_run(conn, run_id, summary)
    return {"run_id": run_id, **summary}
