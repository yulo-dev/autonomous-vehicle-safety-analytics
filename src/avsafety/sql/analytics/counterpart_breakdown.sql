-- Interaction events by counterpart type and severity.
SELECT
    ev.event_type,
    COALESCE(ev.counterpart_type, 'none')           AS counterpart_type,
    ev.severity,
    COUNT(*)                                        AS n_events,
    MIN(json_extract(ev.metrics, '$.min_ttc_s'))    AS worst_min_ttc_s
FROM events ev
JOIN scenario_eligibility e ON e.scenario_id = ev.scenario_id AND e.eligible = 1
WHERE ev.detector_version = :version
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;
