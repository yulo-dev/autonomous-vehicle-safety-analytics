"""Summarize what was ingested: volume, exposure, and parse health."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_table(processed_root: str | Path, table: str) -> pd.DataFrame:
    files = sorted(Path(processed_root).glob(f"*/{table}.parquet"))
    if not files:
        raise FileNotFoundError(f"no {table}.parquet under {processed_root}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def load_manifests(processed_root: str | Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(processed_root).glob("*/manifest.json"))]


def summarize(processed_root: str | Path) -> dict:
    sc = load_table(processed_root, "scenarios")
    manifests = load_manifests(processed_root)
    failures = sum(len(m["failures"]) for m in manifests)
    framing = [m["source_file"] for m in manifests if m["framing_error"]]

    return {
        "shards": len(manifests),
        "scenarios": len(sc),
        "duplicate_scenario_ids": int(sc["scenario_id"].duplicated().sum()),
        "record_parse_failures": failures,
        "shards_with_framing_errors": framing,
        "crc_verified_all": all(m["crc_verified"] for m in manifests),
        "steps_per_scenario": sc["n_steps"].value_counts().sort_index().to_dict(),
        "sdc_km_total": round(sc["sdc_distance_m"].sum() / 1000, 2),
        "sdc_hours_total": round(sc["duration_s"].sum() / 3600, 3),
        "sdc_valid_frac_p05": round(sc["sdc_valid_frac"].quantile(0.05), 4),
        "sdc_stationary_scenarios": int((sc["sdc_distance_m"] < 1.0).sum()),
        "agents_mean": {
            "vehicles": round(sc["n_vehicles"].mean(), 1),
            "pedestrians": round(sc["n_pedestrians"].mean(), 1),
            "cyclists": round(sc["n_cyclists"].mean(), 1),
        },
        "scenarios_with_signals": int((sc["n_signal_lanes"] > 0).sum()),
    }
