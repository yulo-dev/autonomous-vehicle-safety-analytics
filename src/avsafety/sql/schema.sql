-- Event store schema. SQLite for the MVP; types and constraints are kept
-- portable to PostgreSQL.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id       TEXT PRIMARY KEY,
    stage        TEXT NOT NULL CHECK (stage IN ('validate', 'detect')),
    version      TEXT,              -- detector version for detect runs
    config_hash  TEXT NOT NULL,
    git_sha      TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    summary      TEXT               -- JSON
);

CREATE TABLE IF NOT EXISTS scenarios (
    scenario_id      TEXT PRIMARY KEY,
    source_file      TEXT NOT NULL,
    record_index     INTEGER NOT NULL,
    n_steps          INTEGER NOT NULL,
    duration_s       REAL NOT NULL,
    sdc_track_id     INTEGER NOT NULL,
    sdc_distance_m   REAL NOT NULL,
    sdc_valid_frac   REAL NOT NULL,
    n_vehicles       INTEGER NOT NULL,
    n_pedestrians    INTEGER NOT NULL,
    n_cyclists       INTEGER NOT NULL,
    ingest_run_id    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dq_check_results (
    result_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL REFERENCES pipeline_runs (run_id),
    check_name   TEXT NOT NULL,
    entity_type  TEXT NOT NULL CHECK (entity_type IN ('check', 'scenario', 'track')),
    entity_id    TEXT NOT NULL,
    scenario_id  TEXT,              -- set for scenario and track entities
    level        TEXT NOT NULL CHECK (level IN ('error', 'warn')),
    passed       INTEGER NOT NULL CHECK (passed IN (0, 1)),
    details      TEXT               -- JSON
);
CREATE INDEX IF NOT EXISTS ix_dq_run_scenario ON dq_check_results (run_id, scenario_id);

CREATE TABLE IF NOT EXISTS events (
    event_id              TEXT PRIMARY KEY,
    scenario_id           TEXT NOT NULL REFERENCES scenarios (scenario_id),
    run_id                TEXT NOT NULL REFERENCES pipeline_runs (run_id),
    detector_version      TEXT NOT NULL,
    event_type            TEXT NOT NULL,
    start_step            INTEGER NOT NULL,
    end_step              INTEGER NOT NULL,
    start_t_s             REAL NOT NULL,
    end_t_s               REAL NOT NULL,
    severity              TEXT NOT NULL CHECK (severity IN ('S1', 'S2', 'S3')),
    dq_confidence         REAL NOT NULL CHECK (dq_confidence BETWEEN 0 AND 1),
    primary_track_id      INTEGER NOT NULL,
    counterpart_track_id  INTEGER,
    counterpart_type      TEXT,
    metrics               TEXT NOT NULL,  -- JSON
    CHECK (end_step >= start_step)
);
CREATE INDEX IF NOT EXISTS ix_events_version_type ON events (detector_version, event_type);

-- Append-only: corrections are new rows, never edits.
CREATE TABLE IF NOT EXISTS triage_decisions (
    decision_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     TEXT NOT NULL REFERENCES events (event_id),
    reviewer     TEXT NOT NULL CHECK (length(trim(reviewer)) > 0),
    decision     TEXT NOT NULL CHECK (decision IN ('true_issue', 'false_positive', 'needs_review')),
    rationale    TEXT NOT NULL CHECK (length(trim(rationale)) > 0),
    decided_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_triage_event ON triage_decisions (event_id, decision_id);

CREATE TRIGGER IF NOT EXISTS triage_no_update
BEFORE UPDATE ON triage_decisions
BEGIN
    SELECT RAISE(ABORT, 'triage_decisions is append-only');
END;

CREATE TRIGGER IF NOT EXISTS triage_no_delete
BEFORE DELETE ON triage_decisions
BEGIN
    SELECT RAISE(ABORT, 'triage_decisions is append-only');
END;

-- Current review status = latest decision; untouched events are needs_review.
CREATE VIEW IF NOT EXISTS event_status AS
SELECT e.event_id,
       COALESCE(d.decision, 'needs_review') AS review_status,
       d.reviewer AS last_reviewer,
       d.decided_at AS last_decided_at,
       COALESCE(n.n_decisions, 0) AS n_decisions
FROM events e
LEFT JOIN (
    SELECT event_id, decision, reviewer, decided_at,
           ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY decision_id DESC) AS rn
    FROM triage_decisions
) d ON d.event_id = e.event_id AND d.rn = 1
LEFT JOIN (
    SELECT event_id, COUNT(*) AS n_decisions FROM triage_decisions GROUP BY event_id
) n ON n.event_id = e.event_id;

-- Latest validation run.
CREATE VIEW IF NOT EXISTS latest_validate_run AS
SELECT run_id FROM pipeline_runs
WHERE stage = 'validate' AND finished_at IS NOT NULL
ORDER BY started_at DESC, rowid DESC
LIMIT 1;

-- Scenarios eligible for detection and exposure: no error-level failure
-- (scenario or track) in the latest validation run.
CREATE VIEW IF NOT EXISTS scenario_eligibility AS
SELECT s.scenario_id,
       CASE WHEN f.scenario_id IS NULL THEN 1 ELSE 0 END AS eligible,
       f.reasons
FROM scenarios s
LEFT JOIN (
    SELECT scenario_id, group_concat(DISTINCT check_name) AS reasons
    FROM dq_check_results
    WHERE run_id = (SELECT run_id FROM latest_validate_run)
      AND entity_type = 'scenario' AND level = 'error' AND passed = 0
    GROUP BY scenario_id
) f ON f.scenario_id = s.scenario_id;
