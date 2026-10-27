import json

import pytest

from avsafety.protos import scenario_pb2
from avsafety.synthetic import AgentSpec, build_scenario
from avsafety.validate.checks import CHECKS, check_duplicate_ids, run_all
from avsafety.validate.runner import run_validation, track_flags
from tests.conftest import to_frames


def _failures(scenarios, cfg):
    sc, st = to_frames(scenarios)
    return run_all(sc, st, cfg)


def _names(failures):
    return {(f.check_name, f.entity_type, f.track_id) for f in failures}


def test_clean_scenario_has_no_failures(validation_cfg):
    sc = build_scenario("clean", [AgentSpec(0, v0=10.0), AgentSpec(1, x0=40.0)])
    assert _failures([sc], validation_cfg) == []


def test_counterpart_teleport_is_track_error_not_scenario(validation_cfg):
    sc = build_scenario("s", [AgentSpec(0, v0=10.0), AgentSpec(1, x0=40.0, spike_steps=(50,))])
    names = _names(_failures([sc], validation_cfg))
    assert ("position_jump", "track", 1) in names
    assert not any(n[0] == "sdc_track_error" for n in names)


def test_sdc_track_error_escalates_to_scenario(validation_cfg):
    sc = build_scenario("s", [AgentSpec(0, v0=10.0, spike_steps=(50,))])
    f = _failures([sc], validation_cfg)
    esc = [x for x in f if x.check_name == "sdc_track_error"]
    assert len(esc) == 1 and esc[0].level == "error"
    assert "position_jump" in esc[0].details["checks"]


@pytest.mark.parametrize("n_invalid,level", [(10, "warn"), (60, "error")])
def test_sdc_validity_levels(validation_cfg, n_invalid, level):
    sc = build_scenario("s", [AgentSpec(0, invalid_steps=tuple(range(20, 20 + n_invalid)))])
    hits = [f for f in _failures([sc], validation_cfg) if f.check_name == "sdc_low_validity"]
    assert len(hits) == 1 and hits[0].level == level


def test_implausible_speed(validation_cfg):
    sc = build_scenario("s", [AgentSpec(0, v0=10.0), AgentSpec(1, v0=90.0)])
    assert ("implausible_speed", "track", 1) in _names(_failures([sc], validation_cfg))


def test_speed_inconsistency_is_warning(validation_cfg):
    sc = build_scenario("s", [AgentSpec(0), AgentSpec(1, x0=30.0, vel_glitch=(40, 6, -7.0))])
    hits = [f for f in _failures([sc], validation_cfg) if f.check_name == "speed_inconsistency"]
    assert len(hits) == 1 and hits[0].level == "warn" and hits[0].track_id == 1


def test_unknown_type_and_gaps_are_warnings(validation_cfg):
    sc = build_scenario("s", [
        AgentSpec(0),
        AgentSpec(1, object_type=scenario_pb2.Track.TYPE_UNSET, x0=30.0),
        AgentSpec(2, x0=60.0, invalid_steps=tuple(range(30, 36))),
    ])
    f = {(x.check_name, x.track_id): x for x in _failures([sc], validation_cfg)}
    assert f[("unknown_object_type", 1)].level == "warn"
    assert f[("tracking_gaps", 2)].details["gap_steps"] == 6


def test_duplicate_ids(validation_cfg):
    a = build_scenario("dup", [AgentSpec(0)])
    sc, _ = to_frames([a, a])
    f = check_duplicate_ids(sc, validation_cfg)
    assert len(f) == 1 and f[0].details["occurrences"] == 2


def test_runner_records_summaries_and_eligibility(build_store, validation_cfg):
    good = build_scenario("good", [AgentSpec(0), AgentSpec(1, x0=40.0, spike_steps=(50,))])
    bad = build_scenario("bad", [AgentSpec(0, invalid_steps=tuple(range(5, 85)))])
    processed, conn = build_store([good, bad])
    out = run_validation(processed, conn, validation_cfg)

    assert out["scenarios"] == 2 and out["scenarios_excluded"] == 1
    summaries = conn.execute(
        "SELECT check_name, passed, details FROM dq_check_results WHERE run_id = ? AND entity_type = 'check'",
        (out["run_id"],)).fetchall()
    assert {r["check_name"] for r in summaries} == set(CHECKS)  # every check reported, pass or fail
    by_name = {r["check_name"]: r for r in summaries}
    assert by_name["missing_values"]["passed"] == 1
    assert json.loads(by_name["position_jump"]["details"])["n_failed"] == 1

    elig = dict(conn.execute("SELECT scenario_id, eligible FROM scenario_eligibility").fetchall())
    assert elig == {"good": 1, "bad": 0}
    errors, _ = track_flags(conn, out["run_id"])
    assert ("good", 1) in errors


def test_eligibility_follows_latest_run(build_store, validation_cfg):
    sc = build_scenario("s", [AgentSpec(0, invalid_steps=tuple(range(20, 30)))])  # warn only
    processed, conn = build_store([sc])
    run_validation(processed, conn, validation_cfg)
    strict = {**validation_cfg, "sdc_valid_frac": {"warn_below": 0.95, "error_below": 0.95}}
    run_validation(processed, conn, strict)
    assert conn.execute("SELECT eligible FROM scenario_eligibility").fetchone()[0] == 0
