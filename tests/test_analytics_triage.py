import sqlite3

import numpy as np
import pytest

from avsafety.analytics import metrics
from avsafety.detect.runner import run_detection
from avsafety.synthetic import AgentSpec, build_scenario
from avsafety.triage.service import TriageError, current_status, history, record_decision
from avsafety.validate.runner import run_validation


@pytest.fixture
def store(build_store, validation_cfg, detector_cfg):
    scenarios = [
        build_scenario("brake_a", [AgentSpec(0, v0=15.0, accel=-5.0, brake_start_s=3.0)]),
        build_scenario("brake_b", [AgentSpec(0, v0=12.0, accel=-7.0, brake_start_s=2.0)]),
        build_scenario("cruise", [AgentSpec(0, v0=10.0)]),
        build_scenario("excluded_brake", [AgentSpec(0, v0=15.0, accel=-5.0, brake_start_s=3.0,
                                                    invalid_steps=tuple(range(40, 91)))]),
    ]
    processed, conn = build_store(scenarios)
    run_validation(processed, conn, validation_cfg)
    run_detection(processed, conn, detector_cfg)
    return conn


def test_poisson_ci_known_values():
    lo, hi = metrics.poisson_ci(np.array([0, 10]))
    assert lo[0] == 0.0 and hi[0] == pytest.approx(3.689, abs=1e-3)
    assert lo[1] == pytest.approx(4.795, abs=1e-3) and hi[1] == pytest.approx(18.390, abs=1e-3)


def test_exposure_uses_only_eligible_scenarios(store):
    exp = metrics.exposure(store)
    assert exp["n_scenarios"] == 3 and exp["n_excluded"] == 1
    assert exp["sdc_hours"] == pytest.approx(3 * 9.0 / 3600)


def test_rates_have_consistent_denominators(store):
    rates = metrics.event_rates(store, "1.0.0")
    hb = rates.set_index("event_type").loc["hard_braking"]
    exp = metrics.exposure(store)
    assert hb["n_events"] == 2  # excluded scenario's event never counted
    assert hb["per_1k_scenarios"] == pytest.approx(2 / 3 * 1000)
    assert hb["per_100km"] == pytest.approx(2 / (exp["sdc_km"] / 100))
    assert hb["per_1k_scenarios_lo"] < hb["per_1k_scenarios"] < hb["per_1k_scenarios_hi"]
    assert {hb["s1"], hb["s2"], hb["s3"]} == {0, 1}  # -5 -> S2, -7 -> S3


def test_triage_is_append_only_with_history(store):
    eid = store.execute("SELECT event_id FROM events WHERE scenario_id = 'brake_a'").fetchone()[0]
    assert current_status(store, eid) == "needs_review"

    record_decision(store, eid, "alice", "false_positive", "looks like sensor noise")
    record_decision(store, eid, "bob", "true_issue", "lead vehicle braked hard; reaction was correct")
    assert current_status(store, eid) == "true_issue"

    h = history(store, eid)
    assert list(h["reviewer"]) == ["bob", "alice"]  # newest first, nothing overwritten

    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store.execute("UPDATE triage_decisions SET decision = 'needs_review'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store.execute("DELETE FROM triage_decisions")


@pytest.mark.parametrize("kwargs,msg", [
    ({"decision": "maybe"}, "decision must be"),
    ({"reviewer": "  "}, "reviewer"),
    ({"rationale": ""}, "rationale"),
    ({"event_id": "nope"}, "unknown event_id"),
])
def test_triage_validation(store, kwargs, msg):
    eid = store.execute("SELECT event_id FROM events LIMIT 1").fetchone()[0]
    args = {"event_id": eid, "reviewer": "alice", "decision": "true_issue", "rationale": "ok", **kwargs}
    with pytest.raises(TriageError, match=msg):
        record_decision(store, **args)


def test_review_summary_precision(store):
    ids = [r[0] for r in store.execute("SELECT event_id FROM events WHERE event_type = 'hard_braking'")]
    record_decision(store, ids[0], "alice", "true_issue", "real")
    record_decision(store, ids[1], "alice", "false_positive", "glitch")
    s = metrics.review_summary(store, "1.0.0").set_index("event_type").loc["hard_braking"]
    assert s["precision"] == pytest.approx(0.5)
    assert s["review_coverage"] == pytest.approx(2 / s["n_events"])
    assert 0 <= s["precision_lo"] <= s["precision"] <= s["precision_hi"] <= 1


def test_dq_summary_lists_all_checks(store):
    dq = metrics.dq_summary(store)
    assert len(dq) == 14
    assert dq.loc[dq["check_name"] == "sdc_low_validity", "passed"].item() == 0
