import math

import numpy as np
import pandas as pd
import pytest

from avsafety.config import load_config
from avsafety.detect import hard_braking, low_ttc
from avsafety.detect.base import segments, severity_descending
from avsafety.detect.runner import ConfigVersionError, detect_scenario, run_detection
from avsafety.features.kinematics import scenario_features
from avsafety.protos import scenario_pb2
from avsafety.synthetic import AgentSpec, build_scenario, demo_fleet
from avsafety.validate.runner import run_validation
from tests.conftest import to_frames

CFG = load_config("detectors")


def _events(agents, excluded=frozenset()):
    _, st = to_frames([build_scenario("s", agents)])
    events, _ = detect_scenario(st, 0, "s", CFG, set(excluded))
    return events


# ------------------------------------------------------------------ helpers

def test_segments_merge_then_filter():
    m = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 0], dtype=bool)
    assert segments(m, min_len=3, merge_gap=1) == [(0, 4)]
    assert segments(m, min_len=1, merge_gap=0) == [(0, 1), (3, 4), (8, 8)]
    assert segments(np.zeros(5, bool), 1, 1) == []


@pytest.mark.parametrize("value,expected", [(-3.2, "S1"), (-5.0, "S2"), (-7.0, "S3")])
def test_severity_bins(value, expected):
    assert severity_descending(value, [-4.5, -6.0]) == expected


# ------------------------------------------------------------------ hard braking

def test_hard_braking_known_answer():
    ev = [e for e in _events([AgentSpec(0, v0=15.0, accel=-5.0, brake_start_s=3.0)])
          if e.event_type == "hard_braking"]
    assert len(ev) == 1
    e = ev[0]
    assert e.severity == "S2"
    assert e.metrics["peak_decel_mps2"] == pytest.approx(-5.0, abs=0.05)
    assert e.start_t_s == pytest.approx(3.0, abs=0.3)
    assert e.end_t_s == pytest.approx(5.8, abs=0.3)  # until speed drops below min_speed (1 m/s)
    assert e.metrics["delta_v_mps"] > 12


def test_mild_braking_is_not_an_event():
    assert not [e for e in _events([AgentSpec(0, v0=15.0, accel=-2.5, brake_start_s=2.0)])
                if e.event_type == "hard_braking"]


def test_standstill_jitter_ignored():
    sdc = pd.DataFrame({"step": range(20), "t_s": np.arange(20) * 0.1, "track_id": 0,
                        "a_lon": -5.0, "speed": 0.5})
    assert hard_braking.detect(sdc, "s", CFG["hard_braking"], 0.1) == []


# ------------------------------------------------------------------ TTC geometry

def _box(x, y=0.0, vx=0.0, vy=0.0, heading=0.0, length=4.8, width=2.0):
    return {"x": x, "y": y, "vx": vx, "vy": vy, "heading": heading, "length": length, "width": width}


def _arr(boxes):
    return {k: np.array([b[k] for b in boxes]) for k in boxes[0]}


def test_ttc_closed_form_rear_end():
    r = math.hypot(4.8 / 6, 1.0)
    expected = ((30 - 1.6) - 1.6 - 2 * r) / 10.0  # front disc to rear disc
    t, _ = low_ttc.ttc(_box(0, vx=10.0), _arr([_box(30.0)]), horizon=5.0)
    assert t[0] == pytest.approx(expected, abs=1e-9)


def test_ttc_no_conflict_cases():
    t, overlap = low_ttc.ttc(_box(0, vx=10.0), _arr([
        _box(0.0, y=3.7, vx=10.0),   # parallel lane, same speed
        _box(30.0, vx=15.0),         # pulling away
        _box(200.0),                 # beyond horizon
    ]), horizon=5.0)
    assert np.isinf(t).all() and not overlap.any()


def test_ttc_initial_overlap_is_nan():
    t, overlap = low_ttc.ttc(_box(0, vx=10.0), _arr([_box(2.0)]), horizon=5.0)
    assert overlap[0] and np.isnan(t[0])


# ------------------------------------------------------------------ TTC detector

def test_low_ttc_event_approaching_stopped_vehicle():
    ev = [e for e in _events([AgentSpec(0, v0=15.0), AgentSpec(1, x0=60.0, v0=0.0)])
          if e.event_type == "low_ttc_conflict"]
    assert len(ev) == 1
    e = ev[0]
    assert e.counterpart_track_id == 1 and e.counterpart_type == "vehicle"
    assert e.severity == "S3" and e.metrics["min_ttc_s"] < 1.0
    assert e.metrics["closing_speed_at_min_mps"] == pytest.approx(15.0, abs=0.5)


def test_low_ttc_respects_excluded_tracks_and_types():
    agents = [AgentSpec(0, v0=15.0), AgentSpec(1, x0=60.0, v0=0.0)]
    assert not [e for e in _events(agents, excluded={1}) if e.event_type == "low_ttc_conflict"]
    other = [AgentSpec(0, v0=15.0), AgentSpec(1, object_type=scenario_pb2.Track.TYPE_OTHER, x0=60.0, v0=0.0)]
    assert not [e for e in _events(other) if e.event_type == "low_ttc_conflict"]


def test_pedestrian_crossing_conflict():
    ped = scenario_pb2.Track.TYPE_PEDESTRIAN
    agents = [AgentSpec(0, v0=10.0),
              AgentSpec(1, object_type=ped, x0=40.0, y0=-6.0, v0=0.0, vy=1.4, length=0.8, width=0.8)]
    ev = [e for e in _events(agents) if e.event_type == "low_ttc_conflict"]
    assert len(ev) == 1 and ev[0].counterpart_type == "pedestrian"


def test_feature_window_matches_detector_dt():
    _, st = to_frames([build_scenario("s", [AgentSpec(0)])])
    f = scenario_features(st, CFG["smoothing"], 0.1)
    assert f["a_lon"].notna().sum() == 91


# ------------------------------------------------------------------ stage behaviour

def test_detection_idempotent_and_version_guarded(build_store, validation_cfg, detector_cfg):
    scenarios = [build_scenario("brake", [AgentSpec(0, v0=15.0, accel=-5.0, brake_start_s=3.0)])]
    processed, conn = build_store(scenarios)
    run_validation(processed, conn, validation_cfg)

    first = run_detection(processed, conn, detector_cfg)
    assert first["events_inserted"] == 1
    again = run_detection(processed, conn, detector_cfg)
    assert again["events_inserted"] == 0 and again["events_already_present"] == 1

    changed = {**detector_cfg, "hard_braking": {**detector_cfg["hard_braking"], "decel_threshold_mps2": -2.0}}
    with pytest.raises(ConfigVersionError):
        run_detection(processed, conn, changed)
    bumped = run_detection(processed, conn, {**changed, "version": "1.1.0"})
    assert bumped["events_inserted"] == 1  # same event, new version, new id


def test_dq_warning_lowers_confidence(build_store, validation_cfg, detector_cfg):
    clean = build_scenario("clean", [AgentSpec(0, v0=15.0), AgentSpec(1, x0=60.0, v0=0.0)])
    gappy = build_scenario("gappy", [AgentSpec(0, v0=15.0),
                                     AgentSpec(1, x0=60.0, v0=0.0, invalid_steps=tuple(range(15, 25)))])
    processed, conn = build_store([clean, gappy])
    run_validation(processed, conn, validation_cfg)
    run_detection(processed, conn, detector_cfg)
    conf = dict(conn.execute(
        "SELECT scenario_id, dq_confidence FROM events WHERE event_type = 'low_ttc_conflict'").fetchall())
    assert conf["clean"] == 1.0
    assert conf["gappy"] < 0.8


def test_demo_fleet_ground_truth(build_store, validation_cfg, detector_cfg):
    """Templates with no hazard produce no events; glitches fool only hard braking."""
    scenarios, truth = demo_fleet(n=40, seed=3)
    processed, conn = build_store(scenarios)
    run_validation(processed, conn, validation_cfg)
    run_detection(processed, conn, detector_cfg)
    ev = pd.read_sql("SELECT scenario_id, event_type FROM events", conn)
    ev["kind"] = ev["scenario_id"].map(truth)
    assert not ev["kind"].isin(["cruise", "cyclist_pass"]).any()
    glitch = ev[ev["kind"] == "sensor_glitch"]
    assert set(glitch["event_type"]) <= {"hard_braking"}
    assert ev["kind"].str.startswith("dq_sdc").sum() == 0  # excluded scenario never reaches detection
