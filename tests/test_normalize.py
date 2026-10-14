import pyarrow as pa
import pytest

from avsafety.ingest.normalize import (
    AGENT_STATE_SCHEMA,
    SCENARIO_SCHEMA,
    normalize_scenario,
)
from avsafety.protos import scenario_pb2
from avsafety.synthetic import N_STEPS, AgentSpec, build_scenario


def _norm(sc):
    return normalize_scenario(sc.SerializeToString(), "test.tfrecord", 0, "run-test")


def test_row_counts_and_schema_conformance():
    sc = build_scenario("s1", [AgentSpec(0), AgentSpec(1, x0=30.0)])
    out = _norm(sc)
    assert len(out["agent_states"]["step"]) == 2 * N_STEPS
    # Building typed tables proves every column matches its declared schema.
    pa.Table.from_pydict(out["agent_states"], schema=AGENT_STATE_SCHEMA)
    pa.Table.from_pydict(out["scenarios"], schema=SCENARIO_SCHEMA)


def test_sdc_flag_and_exposure_for_constant_speed():
    sc = build_scenario("s1", [AgentSpec(0, v0=12.0), AgentSpec(1, x0=60.0)])
    out = _norm(sc)
    s = out["scenarios"]
    assert s["sdc_track_id"] == [0]
    assert s["sdc_distance_m"][0] == pytest.approx(12.0 * (N_STEPS - 1) * 0.1)
    assert s["duration_s"][0] == pytest.approx(9.0)
    sdc_rows = sum(out["agent_states"]["is_sdc"])
    assert sdc_rows == N_STEPS


def test_invalid_gap_is_not_bridged_in_distance():
    gap = tuple(range(40, 46))
    sc = build_scenario("s1", [AgentSpec(0, v0=10.0, invalid_steps=gap)])
    s = _norm(sc)["scenarios"]
    valid_pairs = (40 - 1) + (N_STEPS - 46 - 1)  # 0..39 and 46..90
    assert s["sdc_distance_m"][0] == pytest.approx(valid_pairs * 1.0)
    assert s["sdc_valid_frac"][0] == pytest.approx((N_STEPS - len(gap)) / N_STEPS)


def test_invalid_states_are_kept_not_dropped():
    sc = build_scenario("s1", [AgentSpec(0, invalid_steps=(5, 6))])
    st = _norm(sc)["agent_states"]
    assert len(st["valid"]) == N_STEPS
    assert st["valid"].count(False) == 2


def test_signal_states_and_map_features():
    sc = build_scenario("s1", [AgentSpec(0)], with_signal=True)
    out = _norm(sc)
    sig = out["signal_states"]
    assert len(sig["step"]) == N_STEPS
    assert set(sig["state"]) == {"go", "stop"}
    assert sig["stop_x"][0] == 120.0
    mf = out["map_features"]
    assert mf["feature_type"] == ["lane"]
    assert mf["subtype"] == ["TYPE_SURFACE_STREET"]
    assert len(mf["xs"][0]) == 31


def test_object_types_counted():
    ped = scenario_pb2.Track.TYPE_PEDESTRIAN
    cyc = scenario_pb2.Track.TYPE_CYCLIST
    sc = build_scenario("s1", [AgentSpec(0), AgentSpec(1, object_type=ped), AgentSpec(2, object_type=cyc)])
    s = _norm(sc)["scenarios"]
    assert (s["n_vehicles"][0], s["n_pedestrians"][0], s["n_cyclists"][0]) == (1, 1, 1)


@pytest.mark.parametrize("mutate,msg", [
    (lambda sc: sc.ClearField("scenario_id"), "scenario_id"),
    (lambda sc: sc.ClearField("timestamps_seconds"), "timestamps"),
    (lambda sc: setattr(sc, "sdc_track_index", 7), "out of range"),
])
def test_structural_errors_raise(mutate, msg):
    sc = build_scenario("s1", [AgentSpec(0)])
    mutate(sc)
    with pytest.raises(ValueError, match=msg):
        _norm(sc)
