"""Build synthetic WOMD-format scenarios with known ground truth.

Used by tests and by `avsafety synth` so the pipeline runs end to end without
Waymo data access. Geometry is deliberately simple: straight roads along +x,
constant or piecewise-constant acceleration, so expected outputs can be
computed by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

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
            x, v = _kinematics(spec, step * DT)
            tr.states.add(
                center_x=x, center_y=spec.y0, center_z=0.0,
                length=spec.length, width=spec.width, height=1.6,
                heading=0.0, velocity_x=v, velocity_y=0.0,
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
