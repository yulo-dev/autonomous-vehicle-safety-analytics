-- Per-check outcome for the latest validation run.
SELECT
    check_name,
    level,
    passed,
    json_extract(details, '$.entity_type')  AS entity_type,
    json_extract(details, '$.n_failed')     AS n_failed,
    json_extract(details, '$.n_evaluated')  AS n_evaluated,
    json_extract(details, '$.description')  AS description
FROM dq_check_results
WHERE run_id = (SELECT run_id FROM latest_validate_run) AND entity_type = 'check'
ORDER BY passed, level, check_name;
