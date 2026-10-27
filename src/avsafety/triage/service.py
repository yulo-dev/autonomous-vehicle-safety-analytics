"""Human triage: append-only review decisions with full history.

A decision is never edited. Changing one's mind means recording a new decision
with its own rationale, so the audit trail shows who decided what, when, and why.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

from avsafety.db import utc_now

DECISIONS = ("true_issue", "false_positive", "needs_review")


class TriageError(ValueError):
    pass


@dataclass(frozen=True)
class Decision:
    decision_id: int
    event_id: str
    reviewer: str
    decision: str
    rationale: str
    decided_at: str


def record_decision(conn: sqlite3.Connection, event_id: str, reviewer: str, decision: str,
                    rationale: str) -> Decision:
    reviewer, rationale = reviewer.strip(), rationale.strip()
    if decision not in DECISIONS:
        raise TriageError(f"decision must be one of {DECISIONS}, got {decision!r}")
    if not reviewer:
        raise TriageError("reviewer is required")
    if not rationale:
        raise TriageError("rationale is required")
    if conn.execute("SELECT 1 FROM events WHERE event_id = ?", (event_id,)).fetchone() is None:
        raise TriageError(f"unknown event_id {event_id!r}")

    decided_at = utc_now()
    cur = conn.execute(
        "INSERT INTO triage_decisions (event_id, reviewer, decision, rationale, decided_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (event_id, reviewer, decision, rationale, decided_at),
    )
    conn.commit()
    return Decision(cur.lastrowid, event_id, reviewer, decision, rationale, decided_at)


def history(conn: sqlite3.Connection, event_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT decision_id, decided_at, reviewer, decision, rationale FROM triage_decisions "
        "WHERE event_id = ? ORDER BY decision_id DESC",
        conn, params=(event_id,),
    )


def current_status(conn: sqlite3.Connection, event_id: str) -> str:
    row = conn.execute("SELECT review_status FROM event_status WHERE event_id = ?", (event_id,)).fetchone()
    if row is None:
        raise TriageError(f"unknown event_id {event_id!r}")
    return row[0]
