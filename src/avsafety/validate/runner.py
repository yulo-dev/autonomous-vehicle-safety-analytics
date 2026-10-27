"""Validation stage: load scenarios into the event store and record DQ results."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd

from avsafety import db
from avsafety.validate.checks import CHECKS, Failure, check_duplicate_ids, run_all

STATE_COLUMNS = [
    "scenario_id", "track_id", "object_type", "is_sdc", "step", "t_s",
    "x", "y", "vx", "vy", "heading", "length", "width", "valid",
]

SCENARIO_COLUMNS = [
    "scenario_id", "source_file", "record_index", "n_steps", "duration_s", "sdc_track_id",
    "sdc_distance_m", "sdc_valid_frac", "n_vehicles", "n_pedestrians", "n_cyclists", "ingest_run_id",
]


def shard_dirs(processed_root: str | Path) -> list[Path]:
    return sorted(p.parent for p in Path(processed_root).glob("*/scenarios.parquet"))


def read_states(shard: Path, columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(shard / "agent_states.parquet", columns=columns or STATE_COLUMNS)
    for col in ("scenario_id", "object_type"):
        if col in df:
            df[col] = df[col].astype("category")
    return df


def _upsert_scenarios(conn: sqlite3.Connection, scenarios: pd.DataFrame) -> int:
    # Duplicates are flagged by DQ; the store keeps the first occurrence as a
    # placeholder so the flag has a row to attach to.
    first = scenarios.drop_duplicates("scenario_id", keep="first")
    cols = ", ".join(SCENARIO_COLUMNS)
    marks = ", ".join("?" for _ in SCENARIO_COLUMNS)
    updates = ", ".join(f"{c} = excluded.{c}" for c in SCENARIO_COLUMNS if c != "scenario_id")
    conn.executemany(
        f"INSERT INTO scenarios ({cols}) VALUES ({marks}) "
        f"ON CONFLICT (scenario_id) DO UPDATE SET {updates}",
        first[SCENARIO_COLUMNS].itertuples(index=False, name=None),
    )
    return len(first)


def _write_results(conn: sqlite3.Connection, run_id: str, failures: list[Failure], n_scenarios: int,
                   n_tracks: int) -> None:
    conn.executemany(
        "INSERT INTO dq_check_results (run_id, check_name, entity_type, entity_id, scenario_id, "
        "level, passed, details) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
        [(run_id, f.check_name, f.entity_type, f.entity_id, f.scenario_id, f.level, json.dumps(f.details))
         for f in failures],
    )
    counts = Counter(f.check_name for f in failures)
    for spec in CHECKS.values():
        evaluated = n_scenarios if spec.entity_type == "scenario" else n_tracks
        conn.execute(
            "INSERT INTO dq_check_results (run_id, check_name, entity_type, entity_id, scenario_id, "
            "level, passed, details) VALUES (?, ?, 'check', ?, NULL, ?, ?, ?)",
            (run_id, spec.name, spec.name, spec.level, int(counts[spec.name] == 0),
             json.dumps({"n_failed": counts[spec.name], "n_evaluated": evaluated,
                         "entity_type": spec.entity_type, "description": spec.description})),
        )


def run_validation(processed_root: str | Path, conn: sqlite3.Connection, cfg: dict) -> dict:
    shards = shard_dirs(processed_root)
    if not shards:
        raise FileNotFoundError(f"no ingested shards under {processed_root}")

    run_id = db.start_run(conn, "validate", db.config_hash(cfg))
    all_scenarios = []
    failures: list[Failure] = []
    n_tracks = 0
    for shard in shards:
        sc = pd.read_parquet(shard / "scenarios.parquet")
        states = read_states(shard)
        all_scenarios.append(sc)
        failures += run_all(sc, states, cfg)
        n_tracks += states.groupby(["scenario_id", "track_id"], observed=True).ngroups

    scenarios = pd.concat(all_scenarios, ignore_index=True)
    failures += check_duplicate_ids(scenarios, cfg)

    n_loaded = _upsert_scenarios(conn, scenarios)
    _write_results(conn, run_id, failures, scenarios["scenario_id"].nunique(), n_tracks)

    excluded = {f.scenario_id for f in failures if f.entity_type == "scenario" and f.level == "error"}
    summary = {
        "shards": len(shards),
        "scenarios": n_loaded,
        "scenarios_excluded": len(excluded),
        "tracks": n_tracks,
        "failures_by_check": dict(Counter(f.check_name for f in failures)),
        "config": cfg,
    }
    db.finish_run(conn, run_id, summary)
    return {"run_id": run_id, **summary}


def latest_validate_run(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT run_id FROM latest_validate_run").fetchone()
    return row["run_id"] if row else None


def track_flags(conn: sqlite3.Connection, run_id: str) -> tuple[set, set]:
    """Return ({(scenario, track)} with errors, {(scenario, track)} with warnings)."""
    errors, warns = set(), set()
    for r in conn.execute(
        "SELECT entity_id, level FROM dq_check_results "
        "WHERE run_id = ? AND entity_type = 'track' AND passed = 0", (run_id,)
    ):
        sid, tid = r["entity_id"].rsplit(":", 1)
        (errors if r["level"] == "error" else warns).add((sid, int(tid)))
    return errors, warns


def scenario_warnings(conn: sqlite3.Connection, run_id: str) -> set[str]:
    return {r["scenario_id"] for r in conn.execute(
        "SELECT DISTINCT scenario_id FROM dq_check_results "
        "WHERE run_id = ? AND entity_type = 'scenario' AND level = 'warn' AND passed = 0", (run_id,)
    )}
