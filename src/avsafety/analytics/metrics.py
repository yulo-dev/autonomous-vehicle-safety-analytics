"""Exposure-normalized safety metrics.

Every rate states its denominator explicitly. Numerators and denominators both
come from scenarios eligible under the latest validation run, so excluding a
scenario for bad data removes its events and its exposure together.

Confidence intervals are exact (Garwood) Poisson intervals for rates and
Wilson intervals for triage precision.
"""

from __future__ import annotations

import sqlite3
from importlib import resources

import numpy as np
import pandas as pd
from scipy import stats


def _sql(name: str) -> str:
    return resources.files("avsafety.sql.analytics").joinpath(f"{name}.sql").read_text()


def query(conn: sqlite3.Connection, name: str, **params) -> pd.DataFrame:
    return pd.read_sql_query(_sql(name), conn, params=params)


def latest_detector_version(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT version FROM pipeline_runs WHERE stage = 'detect' AND finished_at IS NOT NULL "
        "ORDER BY started_at DESC, rowid DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def detector_versions(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT version FROM pipeline_runs WHERE stage = 'detect' AND finished_at IS NOT NULL "
        "GROUP BY version ORDER BY MAX(started_at) DESC")]


def poisson_ci(k: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Exact Garwood interval for a Poisson count."""
    k = np.asarray(k, dtype=float)
    lo = np.where(k > 0, stats.chi2.ppf(alpha / 2, 2 * k) / 2, 0.0)
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / 2
    return lo, hi


def wilson_ci(successes: np.ndarray, n: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    s, n = np.asarray(successes, dtype=float), np.asarray(n, dtype=float)
    z = stats.norm.ppf(1 - alpha / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = s / n
        center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return np.where(n > 0, center - half, np.nan), np.where(n > 0, center + half, np.nan)


def exposure(conn: sqlite3.Connection) -> dict:
    row = query(conn, "exposure").iloc[0]
    excluded = conn.execute("SELECT COUNT(*) FROM scenario_eligibility WHERE eligible = 0").fetchone()[0]
    return {
        "n_scenarios": int(row["n_scenarios"] or 0),
        "sdc_km": float(row["sdc_km"] or 0.0),
        "sdc_hours": float(row["sdc_hours"] or 0.0),
        "n_excluded": int(excluded),
    }


def event_rates(conn: sqlite3.Connection, version: str) -> pd.DataFrame:
    """Per event type: counts, severity mix, and rates per 1k scenarios / 100 km / hour with 95% CIs."""
    exp = exposure(conn)
    df = query(conn, "event_counts", version=version)
    if df.empty:
        return df
    lo, hi = poisson_ci(df["n_events"].to_numpy())
    denominators = {
        "per_1k_scenarios": exp["n_scenarios"] / 1000.0,
        "per_100km": exp["sdc_km"] / 100.0,
        "per_hour": exp["sdc_hours"],
    }
    for name, denom in denominators.items():
        if denom > 0:
            df[name] = df["n_events"] / denom
            df[f"{name}_lo"] = lo / denom
            df[f"{name}_hi"] = hi / denom
        else:
            df[name] = df[f"{name}_lo"] = df[f"{name}_hi"] = np.nan
    df["share_of_scenarios"] = df["n_scenarios_with_event"] / max(exp["n_scenarios"], 1)
    return df


def review_summary(conn: sqlite3.Connection, version: str) -> pd.DataFrame:
    """Triage outcomes and precision = true_issue / (true_issue + false_positive)."""
    df = query(conn, "review_summary", version=version)
    if df.empty:
        return df
    adjudicated = df["true_issue"] + df["false_positive"]
    with np.errstate(invalid="ignore", divide="ignore"):
        df["precision"] = np.where(adjudicated > 0, df["true_issue"] / adjudicated, np.nan)
    df["precision_lo"], df["precision_hi"] = wilson_ci(df["true_issue"], adjudicated)
    df["review_coverage"] = df["touched"] / df["n_events"]
    return df


def events_detail(conn: sqlite3.Connection, version: str) -> pd.DataFrame:
    return query(conn, "events_detail", version=version)


def counterpart_breakdown(conn: sqlite3.Connection, version: str) -> pd.DataFrame:
    return query(conn, "counterpart_breakdown", version=version)


def dq_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    return query(conn, "dq_summary")
