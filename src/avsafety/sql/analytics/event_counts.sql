-- Event counts per type for one detector version, restricted to eligible
-- scenarios so numerator and denominator cover the same population.
SELECT
    ev.event_type,
    COUNT(*)                         AS n_events,
    COUNT(DISTINCT ev.scenario_id)   AS n_scenarios_with_event,
    SUM(ev.severity = 'S1')          AS s1,
    SUM(ev.severity = 'S2')          AS s2,
    SUM(ev.severity = 'S3')          AS s3,
    AVG(ev.dq_confidence)            AS mean_dq_confidence
FROM events ev
JOIN scenario_eligibility e ON e.scenario_id = ev.scenario_id AND e.eligible = 1
WHERE ev.detector_version = :version
GROUP BY ev.event_type
ORDER BY ev.event_type;
