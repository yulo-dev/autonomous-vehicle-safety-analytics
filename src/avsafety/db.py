"""SQLite connection, schema bootstrap, and pipeline-run bookkeeping."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import uuid
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

DEFAULT_DB = Path("data/avsafety.db")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def connect(path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(resources.files("avsafety.sql").joinpath("schema.sql").read_text())
    return conn


def config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def start_run(conn: sqlite3.Connection, stage: str, cfg_hash: str, version: str | None = None) -> str:
    run_id = f"{stage}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    conn.execute(
        "INSERT INTO pipeline_runs (run_id, stage, version, config_hash, git_sha, started_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, stage, version, cfg_hash, git_sha(), utc_now()),
    )
    conn.commit()
    return run_id


def finish_run(conn: sqlite3.Connection, run_id: str, summary: dict) -> None:
    conn.execute(
        "UPDATE pipeline_runs SET finished_at = ?, summary = ? WHERE run_id = ?",
        (utc_now(), json.dumps(summary, default=str), run_id),
    )
    conn.commit()
