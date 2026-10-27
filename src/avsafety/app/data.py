"""Data access for the dashboard, kept free of Streamlit so it can be tested."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from avsafety.config import load_config
from avsafety.detect.low_ttc import pair_ttc_series
from avsafety.features.kinematics import scenario_features
from avsafety.validate.runner import STATE_COLUMNS


def run_config(conn: sqlite3.Connection, run_id: str, fallback: str) -> dict:
    """Config recorded on a pipeline run, so drill-downs use the rules that produced the event."""
    row = conn.execute("SELECT summary FROM pipeline_runs WHERE run_id = ?", (run_id,)).fetchone()
    if row and row[0]:
        cfg = json.loads(row[0]).get("config")
        if cfg:
            return cfg
    return load_config(fallback)


def scenario_states(processed_root: str | Path, source_file: str, scenario_id: str) -> pd.DataFrame:
    path = Path(processed_root) / source_file / "agent_states.parquet"
    df = pd.read_parquet(path, columns=STATE_COLUMNS, filters=[("scenario_id", "==", scenario_id)])
    return df.astype({"scenario_id": str, "object_type": str})


def event_timeseries(states: pd.DataFrame, event: pd.Series, cfg: dict) -> pd.DataFrame:
    """Kinematics (and TTC for interaction events) for the event's tracks, one row per step and track."""
    dt = 0.1
    feats = scenario_features(states, cfg["smoothing"], dt)
    tracks = [int(event["primary_track_id"])]
    if pd.notna(event["counterpart_track_id"]):
        tracks.append(int(event["counterpart_track_id"]))
    ts = feats[feats["track_id"].isin(tracks)].copy()
    ts["role"] = ts["track_id"].map(lambda t: "SDC" if t == tracks[0] else "counterpart")
    if event["event_type"] == "low_ttc_conflict" and len(tracks) == 2:
        ttc = pair_ttc_series(feats, tracks[0], tracks[1], cfg["low_ttc_conflict"])
        ts = ts.merge(ttc, on="step", how="left")
    return ts


def dq_failures(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT check_name, level, entity_type, entity_id, scenario_id, details FROM dq_check_results "
        "WHERE run_id = (SELECT run_id FROM latest_validate_run) AND entity_type != 'check' AND passed = 0 "
        "ORDER BY level, check_name, entity_id", conn)


def scenario_dq(conn: sqlite3.Connection, scenario_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT check_name, level, entity_id, details FROM dq_check_results "
        "WHERE run_id = (SELECT run_id FROM latest_validate_run) AND scenario_id = ? AND passed = 0 "
        "ORDER BY level, check_name", conn, params=(scenario_id,))
