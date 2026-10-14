"""Flatten a WOMD Scenario proto into columnar tables.

Output tables (one row per ...):
    scenarios      scenario
    agent_states   (scenario, track, step); invalid states are kept with valid=False
    map_features   (scenario, map feature); geometry as x/y/z lists
    signal_states  (scenario, step, controlled lane)

No filtering or repair happens here. Data quality is the validation layer's job,
so this module stays a faithful, lossless-enough projection of the source.
"""

from __future__ import annotations

import math
from collections import Counter

import pyarrow as pa

from avsafety.protos import map_pb2, scenario_pb2

OBJECT_TYPES = {
    scenario_pb2.Track.TYPE_UNSET: "unset",
    scenario_pb2.Track.TYPE_VEHICLE: "vehicle",
    scenario_pb2.Track.TYPE_PEDESTRIAN: "pedestrian",
    scenario_pb2.Track.TYPE_CYCLIST: "cyclist",
    scenario_pb2.Track.TYPE_OTHER: "other",
}

SIGNAL_STATES = {
    v: k.removeprefix("LANE_STATE_").lower()
    for k, v in map_pb2.TrafficSignalLaneState.State.items()
}

SCENARIO_SCHEMA = pa.schema([
    ("scenario_id", pa.string()),
    ("source_file", pa.string()),
    ("record_index", pa.int32()),
    ("n_steps", pa.int32()),
    ("current_time_index", pa.int32()),
    ("duration_s", pa.float64()),
    ("dt_median_s", pa.float64()),
    ("n_tracks", pa.int32()),
    ("n_vehicles", pa.int32()),
    ("n_pedestrians", pa.int32()),
    ("n_cyclists", pa.int32()),
    ("n_map_features", pa.int32()),
    ("n_signal_lanes", pa.int32()),
    ("sdc_track_index", pa.int32()),
    ("sdc_track_id", pa.int32()),
    ("sdc_valid_frac", pa.float64()),
    ("sdc_distance_m", pa.float64()),
    ("ingest_run_id", pa.string()),
])

AGENT_STATE_SCHEMA = pa.schema([
    ("scenario_id", pa.string()),
    ("track_id", pa.int32()),
    ("object_type", pa.string()),
    ("is_sdc", pa.bool_()),
    ("step", pa.int16()),
    ("t_s", pa.float64()),
    ("x", pa.float64()),
    ("y", pa.float64()),
    ("z", pa.float64()),
    ("length", pa.float32()),
    ("width", pa.float32()),
    ("height", pa.float32()),
    ("heading", pa.float32()),
    ("vx", pa.float32()),
    ("vy", pa.float32()),
    ("valid", pa.bool_()),
])

MAP_FEATURE_SCHEMA = pa.schema([
    ("scenario_id", pa.string()),
    ("feature_id", pa.int64()),
    ("feature_type", pa.string()),
    ("subtype", pa.string()),
    ("speed_limit_mph", pa.float64()),
    ("controlled_lanes", pa.list_(pa.int64())),
    ("xs", pa.list_(pa.float64())),
    ("ys", pa.list_(pa.float64())),
    ("zs", pa.list_(pa.float64())),
])

SIGNAL_STATE_SCHEMA = pa.schema([
    ("scenario_id", pa.string()),
    ("step", pa.int16()),
    ("t_s", pa.float64()),
    ("lane_id", pa.int64()),
    ("state", pa.string()),
    ("stop_x", pa.float64()),
    ("stop_y", pa.float64()),
])


def _median(values: list[float]) -> float:
    if not values:
        return math.nan
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])


def _path_length(states) -> float:
    """Distance over consecutive valid-valid step pairs only.

    Pairs that straddle an invalid state are skipped instead of bridged, so a
    tracking gap never becomes a straight-line teleport in the exposure total.
    """
    total = 0.0
    for a, b in zip(states, states[1:], strict=False):
        if a.valid and b.valid:
            total += math.hypot(b.center_x - a.center_x, b.center_y - a.center_y)
    return total


def _map_feature_row(scenario_id: str, f) -> dict:
    kind = f.WhichOneof("feature_data")
    row = {
        "scenario_id": scenario_id,
        "feature_id": f.id,
        "feature_type": kind or "unset",
        "subtype": None,
        "speed_limit_mph": None,
        "controlled_lanes": None,
        "xs": [],
        "ys": [],
        "zs": [],
    }
    if kind is None:
        return row
    msg = getattr(f, kind)
    if kind == "lane":
        row["subtype"] = map_pb2.LaneCenter.LaneType.Name(msg.type)
        row["speed_limit_mph"] = msg.speed_limit_mph if msg.HasField("speed_limit_mph") else None
        pts = msg.polyline
    elif kind in ("road_line", "road_edge"):
        enum = map_pb2.RoadLine.RoadLineType if kind == "road_line" else map_pb2.RoadEdge.RoadEdgeType
        row["subtype"] = enum.Name(msg.type)
        pts = msg.polyline
    elif kind == "stop_sign":
        row["controlled_lanes"] = list(msg.lane)
        pts = [msg.position]
    else:  # crosswalk, speed_bump, driveway
        pts = msg.polygon
    row["xs"] = [p.x for p in pts]
    row["ys"] = [p.y for p in pts]
    row["zs"] = [p.z for p in pts]
    return row


def normalize_scenario(
    raw: bytes, source_file: str, record_index: int, run_id: str
) -> dict[str, dict[str, list]]:
    """Parse one serialized Scenario into column dicts keyed by table name.

    Raises google.protobuf.message.DecodeError on undecodable bytes and
    ValueError on structural problems that make the record unusable
    (no scenario_id, no timestamps, SDC index out of range).
    """
    sc = scenario_pb2.Scenario()
    sc.ParseFromString(raw)

    sid = sc.scenario_id
    ts = list(sc.timestamps_seconds)
    if not sid:
        raise ValueError("missing scenario_id")
    if not ts:
        raise ValueError(f"{sid}: no timestamps")
    if not 0 <= sc.sdc_track_index < len(sc.tracks):
        raise ValueError(f"{sid}: sdc_track_index {sc.sdc_track_index} out of range")

    n_steps = len(ts)
    sdc = sc.tracks[sc.sdc_track_index]

    # agent_states, column-major to keep memory flat
    cols: dict[str, list] = {name: [] for name in AGENT_STATE_SCHEMA.names}
    type_counts: Counter[str] = Counter()
    for ti, track in enumerate(sc.tracks):
        otype = OBJECT_TYPES.get(track.object_type, "unknown")
        type_counts[otype] += 1
        is_sdc = ti == sc.sdc_track_index
        for step, s in enumerate(track.states):
            cols["scenario_id"].append(sid)
            cols["track_id"].append(track.id)
            cols["object_type"].append(otype)
            cols["is_sdc"].append(is_sdc)
            cols["step"].append(step)
            # Track/timestamp length mismatch is a DQ finding, not a parse error.
            cols["t_s"].append(ts[step] if step < n_steps else math.nan)
            cols["x"].append(s.center_x)
            cols["y"].append(s.center_y)
            cols["z"].append(s.center_z)
            cols["length"].append(s.length)
            cols["width"].append(s.width)
            cols["height"].append(s.height)
            cols["heading"].append(s.heading)
            cols["vx"].append(s.velocity_x)
            cols["vy"].append(s.velocity_y)
            cols["valid"].append(s.valid)

    # signal_states
    sig: dict[str, list] = {name: [] for name in SIGNAL_STATE_SCHEMA.names}
    signal_lanes: set[int] = set()
    for step, dms in enumerate(sc.dynamic_map_states):
        for ls in dms.lane_states:
            signal_lanes.add(ls.lane)
            sig["scenario_id"].append(sid)
            sig["step"].append(step)
            sig["t_s"].append(ts[step] if step < n_steps else math.nan)
            sig["lane_id"].append(ls.lane)
            sig["state"].append(SIGNAL_STATES.get(ls.state, "unknown"))
            has_stop = ls.HasField("stop_point")
            sig["stop_x"].append(ls.stop_point.x if has_stop else None)
            sig["stop_y"].append(ls.stop_point.y if has_stop else None)

    # map_features
    mf_rows = [_map_feature_row(sid, f) for f in sc.map_features]
    mf = {name: [r[name] for r in mf_rows] for name in MAP_FEATURE_SCHEMA.names}

    sdc_states = list(sdc.states)
    dts = [b - a for a, b in zip(ts, ts[1:], strict=False)]
    scen = {
        "scenario_id": [sid],
        "source_file": [source_file],
        "record_index": [record_index],
        "n_steps": [n_steps],
        "current_time_index": [sc.current_time_index],
        "duration_s": [ts[-1] - ts[0]],
        "dt_median_s": [_median(dts)],
        "n_tracks": [len(sc.tracks)],
        "n_vehicles": [type_counts["vehicle"]],
        "n_pedestrians": [type_counts["pedestrian"]],
        "n_cyclists": [type_counts["cyclist"]],
        "n_map_features": [len(sc.map_features)],
        "n_signal_lanes": [len(signal_lanes)],
        "sdc_track_index": [sc.sdc_track_index],
        "sdc_track_id": [sdc.id],
        "sdc_valid_frac": [sum(s.valid for s in sdc_states) / len(sdc_states) if sdc_states else 0.0],
        "sdc_distance_m": [_path_length(sdc_states)],
        "ingest_run_id": [run_id],
    }

    return {"scenarios": scen, "agent_states": cols, "map_features": mf, "signal_states": sig}
