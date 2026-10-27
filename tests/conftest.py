from pathlib import Path

import pandas as pd
import pytest

from avsafety import db
from avsafety.config import load_config
from avsafety.ingest.normalize import normalize_scenario
from avsafety.ingest.pipeline import ingest_shard
from avsafety.ingest.tfrecord import write_records


def to_frames(scenarios) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize protos straight to (scenarios, agent_states) DataFrames."""
    sc, st = [], []
    for i, s in enumerate(scenarios):
        t = normalize_scenario(s.SerializeToString(), "test.tfrecord", i, "run-test")
        sc.append(pd.DataFrame(t["scenarios"]))
        st.append(pd.DataFrame(t["agent_states"]))
    return pd.concat(sc, ignore_index=True), pd.concat(st, ignore_index=True)


@pytest.fixture
def validation_cfg():
    return load_config("validation")


@pytest.fixture
def detector_cfg():
    return load_config("detectors")


@pytest.fixture
def build_store(tmp_path):
    """Write scenarios to a shard, ingest, and return (processed_root, conn)."""

    def _build(scenarios, extra_payloads=()):
        raw = tmp_path / "raw" / "t.tfrecord-00000-of-00001"
        raw.parent.mkdir(parents=True, exist_ok=True)
        write_records(raw, [s.SerializeToString() for s in scenarios] + list(extra_payloads))
        processed = tmp_path / "processed"
        ingest_shard(raw, processed)
        conn = db.connect(tmp_path / "t.db")
        return Path(processed), conn

    return _build
