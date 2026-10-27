-- Exposure over scenarios eligible under the latest validation run.
SELECT
    COUNT(*)                         AS n_scenarios,
    SUM(s.sdc_distance_m) / 1000.0   AS sdc_km,
    SUM(s.duration_s) / 3600.0       AS sdc_hours
FROM scenarios s
JOIN scenario_eligibility e USING (scenario_id)
WHERE e.eligible = 1;
