"""Safety event review dashboard.

Run with `avsafety app` (or `streamlit run` on this file with AVSAFETY_DB and
AVSAFETY_PROCESSED set).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from avsafety import db
from avsafety.analytics import metrics
from avsafety.app import data
from avsafety.triage.service import DECISIONS, TriageError, history, record_decision

DB_PATH = Path(os.environ.get("AVSAFETY_DB", "data/avsafety.db"))
PROCESSED = Path(os.environ.get("AVSAFETY_PROCESSED", "data/processed"))

# Reference palette: categorical slots 1-2 for event type and roles; one-hue
# ordinal blue ramp for severity; neutral gray for "not yet reviewed".
TYPE_COLORS = {"hard_braking": "#2a78d6", "low_ttc_conflict": "#eb6834"}
SEVERITY_COLORS = {"S1": "#86b6ef", "S2": "#2a78d6", "S3": "#104281"}
STATUS_COLORS = {"true_issue": "#2a78d6", "false_positive": "#eb6834", "needs_review": "#8c8b86"}
ROLE_COLORS = {"SDC": "#2a78d6", "counterpart": "#eb6834", "other": "#b8b7b1"}
PAGES = ["Overview", "Event queue", "Review", "Data quality"]

st.set_page_config(page_title="AV Safety Event Review", layout="wide")


def conn():
    return db.connect(DB_PATH)


def scale(mapping: dict) -> alt.Scale:
    return alt.Scale(domain=list(mapping), range=list(mapping.values()))


def go_review(event_id: str) -> None:
    st.session_state["event_id"] = event_id
    st.session_state["page"] = "Review"


# ------------------------------------------------------------------ sidebar

if not DB_PATH.exists():
    st.error(f"No event store at {DB_PATH}. Run `avsafety validate` and `avsafety detect` first.")
    st.stop()

c = conn()
versions = metrics.detector_versions(c)
if not versions:
    st.warning("No completed detection runs yet. Run `avsafety detect`.")
    st.stop()

st.session_state.setdefault("page", "Overview")
with st.sidebar:
    st.title("AV Safety Review")
    page = st.radio("Page", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    version = st.selectbox("Detector version", versions)
    reviewer = st.text_input("Reviewer", key="reviewer", placeholder="your name")

events = metrics.events_detail(c, version)
events["metrics_d"] = events["metrics"].map(json.loads)

with st.sidebar:
    st.subheader("Filters")
    f_type = st.multiselect("Event type", sorted(events["event_type"].unique()))
    f_sev = st.multiselect("Severity", ["S3", "S2", "S1"])
    f_status = st.multiselect("Review status", list(DECISIONS))
    f_cp = st.multiselect("Counterpart", sorted(events["counterpart_type"].dropna().unique()))
    f_conf = st.slider("Min DQ confidence", 0.0, 1.0, 0.0, 0.05)

mask = events["dq_confidence"] >= f_conf
if f_type:
    mask &= events["event_type"].isin(f_type)
if f_sev:
    mask &= events["severity"].isin(f_sev)
if f_status:
    mask &= events["review_status"].isin(f_status)
if f_cp:
    mask &= events["counterpart_type"].isin(f_cp)
view = events[mask]


# ------------------------------------------------------------------ overview

def page_overview() -> None:
    st.header("Overview")
    exp = metrics.exposure(c)
    rev = metrics.review_summary(c, version)
    reviewed = int(rev["touched"].sum()) if not rev.empty else 0
    adjudicated = int((rev["true_issue"] + rev["false_positive"]).sum()) if not rev.empty else 0
    precision = rev["true_issue"].sum() / adjudicated if adjudicated else None

    k = st.columns(6)
    k[0].metric("Eligible scenarios", f"{exp['n_scenarios']:,}", help=f"{exp['n_excluded']} excluded by DQ")
    k[1].metric("SDC distance", f"{exp['sdc_km']:.1f} km")
    k[2].metric("SDC time", f"{exp['sdc_hours'] * 60:.1f} min")
    k[3].metric("Events", f"{len(events):,}")
    k[4].metric("Reviewed", f"{reviewed}/{len(events)}")
    k[5].metric("Precision", "n/a" if precision is None else f"{precision:.0%}",
                help="true_issue / (true_issue + false_positive), all detectors")

    st.caption("All rates use eligible scenarios only: a scenario excluded for data quality "
               "removes both its events and its exposure.")
    rates = metrics.event_rates(c, version)
    if rates.empty:
        st.info("No events for this detector version.")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Events per 100 SDC-km")
        pts = alt.Chart(rates).encode(
            y=alt.Y("event_type:N", title=None),
            color=alt.Color("event_type:N", scale=scale(TYPE_COLORS), legend=None),
        )
        ci = pts.mark_rule(strokeWidth=2).encode(
            x=alt.X("per_100km_lo:Q", title="Events per 100 km (95% exact Poisson CI)"), x2="per_100km_hi:Q")
        dot = pts.mark_circle(size=120, opacity=1).encode(
            x="per_100km:Q",
            tooltip=[alt.Tooltip("event_type:N", title="Type"), alt.Tooltip("n_events:Q", title="Events"),
                     alt.Tooltip("per_100km:Q", title="Rate", format=".1f"),
                     alt.Tooltip("per_100km_lo:Q", title="CI low", format=".1f"),
                     alt.Tooltip("per_100km_hi:Q", title="CI high", format=".1f")])
        st.altair_chart((ci + dot).properties(height=160), width="stretch")

    with right:
        st.subheader("Severity mix")
        sev = events.groupby(["event_type", "severity"]).size().rename("n").reset_index()
        bars = alt.Chart(sev).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("event_type:N", title=None),
            x=alt.X("n:Q", title="Events", stack="zero"),
            color=alt.Color("severity:N", scale=scale(SEVERITY_COLORS), legend=alt.Legend(orient="top")),
            order=alt.Order("severity:N"),
            tooltip=[alt.Tooltip("event_type:N", title="Type"), alt.Tooltip("severity:N", title="Severity"),
                     alt.Tooltip("n:Q", title="Events")])
        st.altair_chart(bars.properties(height=160), width="stretch")

    table = rates[["event_type", "n_events", "s1", "s2", "s3", "per_1k_scenarios", "per_100km",
                   "per_100km_lo", "per_100km_hi", "per_hour", "mean_dq_confidence"]]
    st.dataframe(table.round(2), hide_index=True, width="stretch")

    st.subheader("Triage progress")
    status = events.groupby(["event_type", "review_status"]).size().rename("n").reset_index()
    sbar = alt.Chart(status).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("event_type:N", title=None),
        x=alt.X("n:Q", title="Events", stack="zero"),
        color=alt.Color("review_status:N", scale=scale(STATUS_COLORS), legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("event_type:N", title="Type"), alt.Tooltip("review_status:N", title="Status"),
                 alt.Tooltip("n:Q", title="Events")])
    st.altair_chart(sbar.properties(height=140), width="stretch")
    if not rev.empty:
        st.dataframe(rev[["event_type", "n_events", "true_issue", "false_positive", "needs_review",
                          "precision", "precision_lo", "precision_hi", "review_coverage"]].round(3),
                     hide_index=True, width="stretch")


# ------------------------------------------------------------------ queue

def page_queue() -> None:
    st.header("Event queue")
    st.caption(f"{len(view)} of {len(events)} events match the filters. Select a row to review it.")
    if view.empty:
        return
    table = view.assign(
        key_metric=view.apply(lambda r: (f"min TTC {r.metrics_d['min_ttc_s']:.2f} s"
                                         if r.event_type == "low_ttc_conflict"
                                         else f"peak {r.metrics_d['peak_decel_mps2']:.1f} m/s²"), axis=1),
    )[["event_id", "event_type", "severity", "key_metric", "counterpart_type", "dq_confidence",
       "review_status", "scenario_id", "start_t_s", "end_t_s", "last_reviewer"]].reset_index(drop=True)

    def on_select():
        rows = st.session_state["queue"].selection.rows
        if rows:
            go_review(table.loc[rows[0], "event_id"])

    st.dataframe(table, key="queue", on_select=on_select, selection_mode="single-row",
                 hide_index=True, width="stretch",
                 column_config={"dq_confidence": st.column_config.ProgressColumn(
                     "DQ confidence", min_value=0.0, max_value=1.0, format="%.2f")})
    pending = table[table["review_status"] == "needs_review"]
    if not pending.empty:
        st.button("Review next unreviewed event", type="primary",
                  on_click=go_review, args=(pending.iloc[0]["event_id"],))


# ------------------------------------------------------------------ review

def _window(ev: pd.Series) -> alt.Chart:
    return alt.Chart(pd.DataFrame({"start": [ev["start_t_s"]], "end": [ev["end_t_s"]]})).mark_rect(
        opacity=0.12, color="#8c8b86").encode(x="start:Q", x2="end:Q")


def _line(ts: pd.DataFrame, y: str, title: str, ev: pd.Series, rule: float | None = None,
          clip: float | None = None) -> alt.Chart:
    d = ts.dropna(subset=[y]).copy()
    if clip is not None:
        d[y] = d[y].clip(upper=clip)
    base = alt.Chart(d).mark_line(strokeWidth=2).encode(
        x=alt.X("t_s:Q", title="Time (s)"),
        y=alt.Y(f"{y}:Q", title=title),
        color=alt.Color("role:N", scale=scale({k: ROLE_COLORS[k] for k in ("SDC", "counterpart")
                                               if k in set(d["role"])}),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("role:N"), alt.Tooltip("t_s:Q", title="t (s)", format=".1f"),
                 alt.Tooltip(f"{y}:Q", title=title, format=".2f")])
    layers = [_window(ev), base]
    if rule is not None:
        layers.append(alt.Chart(pd.DataFrame({"v": [rule]})).mark_rule(strokeDash=[4, 4], color="#52514e")
                      .encode(y="v:Q"))
    return alt.layer(*layers).properties(height=200)


def _trajectory(states: pd.DataFrame, ev: pd.Series) -> alt.Chart:
    s = states[states["valid"]].copy()
    cp = ev["counterpart_track_id"]
    s["role"] = np.where(s["track_id"] == ev["primary_track_id"], "SDC",
                         np.where(s["track_id"] == cp, "counterpart", "other"))
    s["in_event"] = s["step"].between(ev["start_step"], ev["end_step"])
    cx, cy = s.loc[s["role"] != "other", "x"].mean(), s.loc[s["role"] != "other", "y"].mean()
    half = max(np.ptp(s.loc[s["role"] != "other", "x"]), np.ptp(s.loc[s["role"] != "other", "y"]), 20) / 2 + 10
    s = s[s["x"].between(cx - half, cx + half) & s["y"].between(cy - half, cy + half)]
    xs, ys = alt.Scale(domain=[cx - half, cx + half]), alt.Scale(domain=[cy - half, cy + half])
    path = alt.Chart(s).mark_line(strokeWidth=2).encode(
        x=alt.X("x:Q", scale=xs, title="x (m)"), y=alt.Y("y:Q", scale=ys, title="y (m)"),
        detail="track_id:N", order="step:Q",
        color=alt.Color("role:N", scale=scale(ROLE_COLORS), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("role:N"), alt.Tooltip("track_id:N"), alt.Tooltip("object_type:N")])
    hot = alt.Chart(s[s["in_event"] & (s["role"] != "other")]).mark_circle(size=60, opacity=1).encode(
        x=alt.X("x:Q", scale=xs), y=alt.Y("y:Q", scale=ys),
        color=alt.Color("role:N", scale=scale(ROLE_COLORS), legend=None),
        tooltip=[alt.Tooltip("role:N"), alt.Tooltip("step:Q"), alt.Tooltip("t_s:Q", format=".1f")])
    return (path + hot).properties(height=420)


def page_review() -> None:
    st.header("Review")
    if events.empty:
        st.info("No events.")
        return
    ids = list(view["event_id"]) or list(events["event_id"])
    current = st.session_state.get("event_id")
    if current not in set(events["event_id"]):
        current = ids[0]
    options = ids if current in ids else [current] + ids
    label = events.set_index("event_id")
    event_id = st.selectbox(
        "Event", options, index=options.index(current),
        format_func=lambda e: f"{e} · {label.at[e, 'event_type']} · {label.at[e, 'severity']} · "
                              f"{label.at[e, 'review_status']}")
    st.session_state["event_id"] = event_id
    ev = label.loc[event_id].copy()
    ev["event_id"] = event_id
    m = ev["metrics_d"]

    cols = st.columns(5)
    cols[0].metric("Type", ev["event_type"])
    cols[1].metric("Severity", ev["severity"])
    cols[2].metric("Status", ev["review_status"])
    cols[3].metric("DQ confidence", f"{ev['dq_confidence']:.2f}")
    cols[4].metric("Window", f"{ev['start_t_s']:.1f} to {ev['end_t_s']:.1f} s")
    st.caption(f"Scenario `{ev['scenario_id']}` · source `{ev['source_file']}` · "
               f"detector {ev['detector_version']} · run `{ev['run_id']}`")

    cfg = data.run_config(c, ev["run_id"], "detectors")
    try:
        states = data.scenario_states(PROCESSED, ev["source_file"], ev["scenario_id"])
    except FileNotFoundError:
        st.error(f"Agent states not found under {PROCESSED}/{ev['source_file']}.")
        states = None

    left, right = st.columns([3, 2])
    with left:
        if states is not None:
            ts = data.event_timeseries(states, ev, cfg)
            st.altair_chart(_trajectory(states, ev), width="stretch")
            st.altair_chart(_line(ts, "speed", "Speed (m/s)", ev), width="stretch")
            if ev["event_type"] == "hard_braking":
                st.altair_chart(_line(ts[ts["role"] == "SDC"], "a_lon", "Longitudinal accel (m/s²)", ev,
                                      rule=cfg["hard_braking"]["decel_threshold_mps2"]), width="stretch")
            elif "ttc_s" in ts:
                horizon = cfg["low_ttc_conflict"]["horizon_s"]
                st.altair_chart(_line(ts[ts["role"] == "SDC"], "ttc_s", f"TTC (s, clipped at {horizon:g})", ev,
                                      rule=cfg["low_ttc_conflict"]["ttc_threshold_s"], clip=horizon),
                                width="stretch")
    with right:
        st.subheader("Metrics")
        st.dataframe(pd.DataFrame({"metric": list(m), "value": [str(v) for v in m.values()]}),
                     hide_index=True, width="stretch")
        dq = data.scenario_dq(c, ev["scenario_id"])
        if not dq.empty:
            st.subheader("Data quality flags")
            st.dataframe(dq, hide_index=True, width="stretch")

        st.subheader("Decision")
        with st.form("triage", clear_on_submit=True):
            decision = st.radio("Decision", DECISIONS, horizontal=True, key="decision")
            rationale = st.text_area("Rationale", key="rationale", placeholder="What in the data supports this call?")
            submitted = st.form_submit_button("Record decision", type="primary")
        if submitted:
            try:
                d = record_decision(c, event_id, reviewer or "", decision, rationale)
                st.success(f"Recorded {d.decision} at {d.decided_at}")
                st.rerun()
            except TriageError as e:
                st.error(str(e))

        st.subheader("Audit history")
        h = history(c, event_id)
        if h.empty:
            st.caption("No decisions yet.")
        else:
            st.dataframe(h, hide_index=True, width="stretch")


# ------------------------------------------------------------------ data quality

def page_dq() -> None:
    st.header("Data quality")
    exp = metrics.exposure(c)
    st.caption(f"Latest validation run. {exp['n_excluded']} scenario(s) excluded from detection and "
               "exposure; warnings are kept and lower event DQ confidence.")
    summary = metrics.dq_summary(c)
    st.dataframe(summary, hide_index=True, width="stretch",
                 column_config={"passed": st.column_config.CheckboxColumn("passed")})
    failures = data.dq_failures(c)
    st.subheader(f"Failures ({len(failures)})")
    if not failures.empty:
        st.dataframe(failures, hide_index=True, width="stretch")
    excluded = pd.read_sql_query(
        "SELECT scenario_id, reasons FROM scenario_eligibility WHERE eligible = 0", c)
    st.subheader(f"Excluded scenarios ({len(excluded)})")
    if not excluded.empty:
        st.dataframe(excluded, hide_index=True, width="stretch")


{"Overview": page_overview, "Event queue": page_queue, "Review": page_review,
 "Data quality": page_dq}[page]()
