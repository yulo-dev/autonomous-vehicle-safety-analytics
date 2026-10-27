"""Build synthetic WOMD-format scenarios with known ground truth.

Used by tests and by `avsafety synth` so the pipeline runs end to end without
Waymo data access. Geometry is deliberately simple: straight roads along +x,
constant or piecewise-constant acceleration, so expected outputs can be
computed by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from avsafety.protos import map_pb2, scenario_pb2

DT = 0.1
N_STEPS = 91
CURRENT_INDEX = 10


@dataclass
class AgentSpec:
    track_id: int
    object_type: int = scenario_pb2.Track.TYPE_VEHICLE
    x0: float = 0.0
    y0: float = 0.0
    v0: float = 10.0
    accel: float = 0.0
    brake_start_s: float | None = None  # accel applies only after this time if set
    invalid_steps: tuple[int, ...] = ()
    length: float = 4.8
    width: float = 2.0
    vy: float = 0.0                                          # constant lateral velocity
    lane_change: tuple[float, float, float] | None = None    # (start_s, duration_s, dy_m)
    vel_glitch: tuple[int, int, float] | None = None         # (start_step, n_steps, dv) sensor glitch
    spike_steps: tuple[int, ...] = ()                        # +30 m position teleports
    noise_pos: float = 0.0                                   # gaussian sigma, m
    noise_vel: float = 0.0                                   # gaussian sigma, m/s


def _lateral(spec: AgentSpec, t: float) -> tuple[float, float]:
    """Return (y, vy): constant drift plus an optional linear lane change."""
    y, vy = spec.y0 + spec.vy * t, spec.vy
    if spec.lane_change is not None:
        t0, dur, dy = spec.lane_change
        frac = min(max((t - t0) / dur, 0.0), 1.0)
        y += dy * frac
        if t0 <= t < t0 + dur:
            vy += dy / dur
    return y, vy


def _kinematics(spec: AgentSpec, t: float) -> tuple[float, float]:
    """Return (x, v) along +x; velocity is floored at zero (no reversing)."""
    if spec.brake_start_s is None or spec.accel == 0.0:
        t0, a = 0.0, spec.accel
    else:
        t0, a = spec.brake_start_s, spec.accel
    if t <= t0:
        return spec.x0 + spec.v0 * t, spec.v0
    x_t0 = spec.x0 + spec.v0 * t0
    tau = t - t0
    if a < 0:
        t_stop = spec.v0 / -a
        if tau >= t_stop:
            return x_t0 + spec.v0 * t_stop + 0.5 * a * t_stop**2, 0.0
    return x_t0 + spec.v0 * tau + 0.5 * a * tau**2, spec.v0 + a * tau


def build_scenario(
    scenario_id: str,
    agents: list[AgentSpec],
    sdc_index: int = 0,
    n_steps: int = N_STEPS,
    with_signal: bool = False,
    rng: np.random.Generator | None = None,
) -> scenario_pb2.Scenario:
    sc = scenario_pb2.Scenario(
        scenario_id=scenario_id,
        current_time_index=CURRENT_INDEX,
        sdc_track_index=sdc_index,
    )
    sc.timestamps_seconds.extend(round(i * DT, 6) for i in range(n_steps))
    for spec in agents:
        tr = sc.tracks.add(id=spec.track_id, object_type=spec.object_type)
        for step in range(n_steps):
            t = step * DT
            x, v = _kinematics(spec, t)
            y, vy = _lateral(spec, t)
            heading = math.atan2(vy, v) if math.hypot(v, vy) > 0.1 else 0.0
            vx_rep, vy_rep = v, vy
            if spec.vel_glitch and spec.vel_glitch[0] <= step < spec.vel_glitch[0] + spec.vel_glitch[1]:
                vx_rep += spec.vel_glitch[2]
            if rng is not None and (spec.noise_pos or spec.noise_vel):
                x += rng.normal(0, spec.noise_pos)
                y += rng.normal(0, spec.noise_pos)
                vx_rep += rng.normal(0, spec.noise_vel)
                vy_rep += rng.normal(0, spec.noise_vel)
            if step in spec.spike_steps:
                x += 30.0
            tr.states.add(
                center_x=x, center_y=y, center_z=0.0,
                length=spec.length, width=spec.width, height=1.6,
                heading=heading, velocity_x=vx_rep, velocity_y=vy_rep,
                valid=step not in spec.invalid_steps,
            )

    lane = sc.map_features.add(id=100)
    lane.lane.type = map_pb2.LaneCenter.TYPE_SURFACE_STREET
    lane.lane.speed_limit_mph = 35.0
    for x in range(-50, 251, 10):
        lane.lane.polyline.add(x=float(x), y=0.0, z=0.0)

    for step in range(n_steps):
        dms = sc.dynamic_map_states.add()
        if with_signal:
            state = (map_pb2.TrafficSignalLaneState.LANE_STATE_GO if step < 50
                     else map_pb2.TrafficSignalLaneState.LANE_STATE_STOP)
            ls = dms.lane_states.add(lane=100, state=state)
            ls.stop_point.x, ls.stop_point.y = 120.0, 0.0
    return sc


def demo_scenarios() -> list[scenario_pb2.Scenario]:
    """A small mixed set: cruising, a hard brake behind a lead, a pedestrian."""
    ped = scenario_pb2.Track.TYPE_PEDESTRIAN
    return [
        build_scenario("synth_cruise", [AgentSpec(0, v0=12.0), AgentSpec(1, x0=60.0, v0=12.0)]),
        build_scenario(
            "synth_hard_brake",
            [
                AgentSpec(0, v0=15.0, accel=-5.0, brake_start_s=3.0),
                AgentSpec(1, x0=25.0, v0=15.0, accel=-6.0, brake_start_s=2.5),
            ],
            with_signal=True,
        ),
        build_scenario(
            "synth_ped",
            [AgentSpec(0, v0=8.0), AgentSpec(1, object_type=ped, x0=40.0, y0=1.5, v0=0.0)],
        ),
        build_scenario("synth_gap", [AgentSpec(0, v0=10.0, invalid_steps=tuple(range(40, 46)))]),
    ]


# --------------------------------------------------------------------- demo fleet

VEH = scenario_pb2.Track.TYPE_VEHICLE
PED = scenario_pb2.Track.TYPE_PEDESTRIAN
CYC = scenario_pb2.Track.TYPE_CYCLIST


def _sid(rng: np.random.Generator) -> str:
    return f"{int(rng.integers(0, 16**12)):012x}"


def _traffic(rng: np.random.Generator, start_id: int, n: int) -> list[AgentSpec]:
    """Background vehicles in an adjacent lane, far enough not to interact."""
    return [
        AgentSpec(start_id + i, x0=float(rng.uniform(-40, 120)), y0=3.7 * float(rng.choice([-1, 1])),
                  v0=float(rng.uniform(5, 14)), noise_pos=0.05, noise_vel=0.1)
        for i in range(n)
    ]


def demo_fleet(n: int = 60, seed: int = 7) -> tuple[list[scenario_pb2.Scenario], dict[str, str]]:
    """Randomized scenarios covering events, near-misses, false positives and DQ issues.

    Returns the scenarios plus a {scenario_id: template} map, which acts as the
    hidden ground truth when judging detector output on synthetic data.
    """
    rng = np.random.default_rng(seed)
    noise = {"noise_pos": 0.05, "noise_vel": 0.1}
    templates = ["cruise", "lead_brake", "cut_in", "ped_crossing", "sensor_glitch", "cyclist_pass"]
    weights = [0.30, 0.22, 0.15, 0.13, 0.12, 0.08]
    out: list[scenario_pb2.Scenario] = []
    truth: dict[str, str] = {}

    for _ in range(n):
        kind = str(rng.choice(templates, p=weights))
        sid = _sid(rng)
        v = float(rng.uniform(8, 15))
        bg = _traffic(rng, 10, int(rng.integers(2, 6)))
        if kind == "cruise":
            agents = [AgentSpec(0, v0=v, **noise), AgentSpec(1, x0=float(rng.uniform(40, 80)), v0=v, **noise)]
        elif kind == "lead_brake":
            gap = float(rng.uniform(14, 30))
            t_lead = float(rng.uniform(2.0, 4.0))
            lead_accel = float(rng.uniform(-7.0, -4.0))
            agents = [
                AgentSpec(0, v0=v, accel=lead_accel * float(rng.uniform(0.75, 1.15)),
                          brake_start_s=t_lead + float(rng.uniform(0.4, 1.0)), **noise),
                AgentSpec(1, x0=gap, v0=v, accel=lead_accel, brake_start_s=t_lead, **noise),
            ]
        elif kind == "cut_in":
            t_cut = float(rng.uniform(2.5, 4.0))
            agents = [
                AgentSpec(0, v0=v, accel=float(rng.uniform(-4.5, -2.5)), brake_start_s=t_cut + 0.8, **noise),
                AgentSpec(1, x0=v * t_cut * 0.55 + float(rng.uniform(8, 16)), y0=3.6, v0=v * 0.55,
                          lane_change=(t_cut, 1.5, -3.6), **noise),
            ]
        elif kind == "ped_crossing":
            agents = [
                AgentSpec(0, v0=v, accel=float(rng.uniform(-6.0, -2.5)), brake_start_s=float(rng.uniform(0.6, 1.6)),
                          **noise),
                AgentSpec(1, object_type=PED, x0=float(rng.uniform(28, 40)), y0=-6.0, v0=0.0, vy=1.4,
                          length=0.8, width=0.8, noise_pos=0.05),
            ]
        elif kind == "sensor_glitch":
            agents = [
                AgentSpec(0, v0=v, vel_glitch=(int(rng.integers(20, 70)), 4, -3.5), **noise),
                AgentSpec(1, x0=float(rng.uniform(40, 80)), v0=v, **noise),
            ]
        else:  # cyclist_pass: cyclist in a bike lane, passed with margin
            agents = [
                AgentSpec(0, v0=v, **noise),
                AgentSpec(1, object_type=CYC, x0=float(rng.uniform(15, 30)), y0=-2.4, v0=5.0,
                          length=1.8, width=0.7, **noise),
            ]
        out.append(build_scenario(sid, agents + bg, with_signal=bool(rng.random() < 0.3), rng=rng))
        truth[sid] = kind

    # Deliberate data quality problems, one of each.
    bad = [
        ("dq_teleport_counterpart",
         [AgentSpec(0, v0=10.0, **noise), AgentSpec(1, x0=40.0, v0=10.0, spike_steps=(50,))]),
        ("dq_sdc_mostly_invalid", [AgentSpec(0, v0=10.0, invalid_steps=tuple(range(10, 80)))]),
        ("dq_parked_sdc", [AgentSpec(0, v0=0.0), AgentSpec(1, x0=20.0, v0=8.0, y0=3.7, **noise)]),
        ("dq_unknown_type", [AgentSpec(0, v0=10.0, **noise),
                             AgentSpec(1, object_type=scenario_pb2.Track.TYPE_UNSET, x0=50.0, v0=10.0)]),
    ]
    for kind, agents in bad:
        sid = _sid(rng)
        out.append(build_scenario(sid, agents, rng=rng))
        truth[sid] = kind
    # A duplicate scenario_id: re-emit the first scenario.
    out.append(out[0])
    return out, truth
