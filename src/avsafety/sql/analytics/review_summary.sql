-- Triage outcomes per event type (current status = latest decision).
SELECT
    ev.event_type,
    COUNT(*)                                        AS n_events,
    SUM(st.review_status = 'true_issue')            AS true_issue,
    SUM(st.review_status = 'false_positive')        AS false_positive,
    SUM(st.review_status = 'needs_review')          AS needs_review,
    SUM(st.n_decisions > 0)                         AS touched
FROM events ev
JOIN event_status st USING (event_id)
WHERE ev.detector_version = :version
GROUP BY ev.event_type
ORDER BY ev.event_type;
