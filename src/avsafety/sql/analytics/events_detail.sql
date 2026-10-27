-- One row per event with review status, for listing and drill-down.
SELECT
    ev.event_id, ev.scenario_id, ev.event_type, ev.severity, ev.dq_confidence,
    ev.start_step, ev.end_step, ev.start_t_s, ev.end_t_s,
    ev.primary_track_id, ev.counterpart_track_id, ev.counterpart_type,
    ev.metrics, ev.detector_version, ev.run_id,
    st.review_status, st.last_reviewer, st.last_decided_at, st.n_decisions,
    s.source_file
FROM events ev
JOIN event_status st USING (event_id)
JOIN scenarios s USING (scenario_id)
WHERE ev.detector_version = :version
ORDER BY ev.severity DESC, ev.scenario_id, ev.start_step;
