import numpy as np
import pytest

from avsafety.features.kinematics import scenario_features, window_steps
from avsafety.synthetic import AgentSpec, build_scenario
from tests.conftest import to_frames

SMOOTH = {"window_s": 0.7, "polyorder": 2}


def _feats(agents, rng=None):
    _, st = to_frames([build_scenario("s", agents, rng=rng)])
    return scenario_features(st, SMOOTH)


def test_window_is_odd():
    assert window_steps(0.7, 0.1) == 7
    assert window_steps(0.6, 0.1) == 7


def test_constant_deceleration_recovered():
    f = _feats([AgentSpec(0, v0=20.0, accel=-4.0)])
    interior = f[(f["step"] > 5) & (f["step"] < 40)]  # stops at t=5.0 s
    assert interior["a_lon"].to_numpy() == pytest.approx(-4.0, abs=1e-6)
    assert interior["speed"].to_numpy() == pytest.approx(20.0 - 4.0 * interior["t_s"].to_numpy(), abs=1e-6)


def test_smoothing_never_crosses_invalid_gap():
    f = _feats([AgentSpec(0, invalid_steps=(40, 41))])
    assert f.loc[f["step"].isin([40, 41]), "a_lon"].isna().all()
    assert f.loc[f["step"].isin([39, 42]), "a_lon"].notna().all()


def test_short_valid_run_gets_no_features():
    invalid = tuple(range(0, 40)) + tuple(range(45, 91))  # 5-step valid run < window of 7
    f = _feats([AgentSpec(0, invalid_steps=invalid)])
    assert f["a_lon"].isna().all()


def test_noise_does_not_create_large_acceleration():
    f = _feats([AgentSpec(0, v0=12.0, noise_pos=0.05, noise_vel=0.1)], rng=np.random.default_rng(0))
    assert f["a_lon"].abs().max() < 2.0


def test_longitudinal_axis_follows_travel_direction():
    # Pure +y motion at 10 m/s: speed comes from vy and a_lon stays ~0.
    f = _feats([AgentSpec(0, v0=0.0, vy=0.0, lane_change=(0.0, 100.0, 1000.0))])  # pure +y at 10 m/s
    assert f["a_lon"].abs().max() < 1e-6
    assert f["speed"].dropna().to_numpy() == pytest.approx(10.0)
