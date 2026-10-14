import json

import pandas as pd

from avsafety.ingest.inventory import summarize
from avsafety.ingest.pipeline import ingest_shard
from avsafety.ingest.tfrecord import write_records
from avsafety.synthetic import N_STEPS, demo_scenarios


def _shard(tmp_path, payloads, name="demo.tfrecord-00000-of-00001"):
    path = tmp_path / name
    write_records(path, payloads)
    return path


def test_clean_shard(tmp_path):
    scenarios = demo_scenarios()
    shard = _shard(tmp_path, [s.SerializeToString() for s in scenarios])
    m = ingest_shard(shard, tmp_path / "out")

    assert m.scenarios_written == len(scenarios)
    assert m.failures == [] and m.framing_error is None
    out = tmp_path / "out" / shard.name
    sc = pd.read_parquet(out / "scenarios.parquet")
    assert list(sc["scenario_id"]) == [s.scenario_id for s in scenarios]
    states = pd.read_parquet(out / "agent_states.parquet")
    assert len(states) == sum(len(s.tracks) for s in scenarios) * N_STEPS
    assert json.loads((out / "manifest.json").read_text())["scenarios_written"] == len(scenarios)


def test_bad_record_is_logged_and_others_survive(tmp_path):
    good = [s.SerializeToString() for s in demo_scenarios()[:2]]
    shard = _shard(tmp_path, [good[0], b"\xff\xff not a proto", good[1]])
    m = ingest_shard(shard, tmp_path / "out")

    assert m.scenarios_written == 2
    assert len(m.failures) == 1
    assert m.failures[0].record_index == 1


def test_truncated_shard_keeps_prefix(tmp_path):
    payloads = [s.SerializeToString() for s in demo_scenarios()]
    shard = _shard(tmp_path, payloads)
    shard.write_bytes(shard.read_bytes()[:-100])
    m = ingest_shard(shard, tmp_path / "out")

    assert m.framing_error is not None
    assert m.scenarios_written == len(payloads) - 1


def test_max_scenarios_and_rerun_is_idempotent(tmp_path):
    shard = _shard(tmp_path, [s.SerializeToString() for s in demo_scenarios()])
    ingest_shard(shard, tmp_path / "out", max_scenarios=2)
    m = ingest_shard(shard, tmp_path / "out", max_scenarios=2)
    sc = pd.read_parquet(tmp_path / "out" / shard.name / "scenarios.parquet")
    assert m.scenarios_written == 2 and len(sc) == 2


def test_inventory(tmp_path):
    shard = _shard(tmp_path, [s.SerializeToString() for s in demo_scenarios()])
    ingest_shard(shard, tmp_path / "out")
    inv = summarize(tmp_path / "out")
    assert inv["scenarios"] == 4
    assert inv["duplicate_scenario_ids"] == 0
    assert inv["record_parse_failures"] == 0
    assert inv["steps_per_scenario"] == {N_STEPS: 4}
    assert inv["scenarios_with_signals"] == 1
