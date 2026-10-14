# Design

## Goals

Simulate an AV safety data reporting workflow end to end: ingest fleet logs, validate them,
derive safety-relevant events with reproducible rules, route events through human triage with
a full audit trail, and report rates and version regressions with explicit denominators.

## Known limitations (read first)

1. **WOMD is curated.** Scenarios were mined for interesting interactions, so event rates are
   inflated relative to real fleet driving. All rates are per curated-scenario exposure.
2. **No ground-truth safety labels.** Every event is a rule-based derivation with documented
   thresholds. `dq_confidence` reflects data quality (valid-frame coverage, signal noise), not
   the probability that an event is real.
3. **Software version is synthetic.** Versions are assigned by `hash(scenario_id)`. To validate
   the regression detector, one version can carry a documented, injected perturbation.
4. **Short windows.** Validation scenarios are 9.1 s at 10 Hz. Events that start before or end
   after the window are truncated.
5. **Perception-derived kinematics.** Tracks come from an offboard detector, so finite-difference
   acceleration is noisy and must be smoothed and gated on `valid`.

## Architecture

```
WOMD TFRecords
   │  ingest/      pure-Python TFRecord reader + protobuf parse
   ▼
Parquet: scenarios, agent_states, map_features, signal_states (+ manifest)
   │  validate/    checks write dq_check_results; records are flagged, never dropped
   ▼
features/          smoothing, acceleration, jerk, relative geometry
   │
   ▼
detect/            versioned rule-based detectors ──► events (SQLite)
   ├─► triage/     append-only decisions; current status is a view
   ├─► llm/        optional summaries with prompt/response lineage; no labeling authority
   └─► analytics/  SQL + pandas metrics, Poisson rate comparison across versions
          │
          ▼
       Streamlit dashboard
```

High-volume state data stays in Parquet. Events, checks, triage, and summaries live in SQLite
with a Postgres-compatible schema.

## Data model (event store)

```sql
scenarios        (scenario_id PK, source_file, sdc_track_id, n_steps, duration_s,
                  sdc_distance_m, software_version, ingest_run_id)
pipeline_runs    (run_id PK, stage, config_hash, git_sha, started_at, finished_at)
dq_check_results (result_id PK, run_id FK, check_name, entity_type, entity_id,
                  level CHECK IN ('error','warn'), passed, details JSON)
events           (event_id PK = hash(scenario, type, start_step, detector_version),
                  scenario_id FK, run_id FK, detector_version, event_type,
                  start_t_s, end_t_s, severity CHECK IN ('S1','S2','S3'),
                  dq_confidence, primary_track_id, counterpart_track_id,
                  counterpart_type, metrics JSON)
triage_decisions (decision_id PK, event_id FK, reviewer,
                  decision CHECK IN ('true_issue','false_positive','needs_review'),
                  rationale, decided_at)          -- append-only, enforced by triggers
event_status     VIEW: latest decision per event, default 'needs_review'
llm_summaries    (summary_id PK, event_id FK, model, prompt_version, prompt_text,
                  input_fields JSON, input_hash, response JSON, created_at)
```

## Event definitions (SDC-centric)

| Event | Rule | Severity basis |
|---|---|---|
| `hard_braking` | Smoothed longitudinal accel < -3.0 m/s² for >= 0.3 s | Peak decel: -3 / -4.5 / -6 |
| `close_following` | Lead vehicle ahead (heading diff < 20°, lateral offset < 1.75 m), time headway < 1.0 s for >= 0.5 s | Min headway |
| `low_ttc_conflict` | Constant-velocity TTC to any agent < 2.0 s | Min TTC: 2.0 / 1.5 / 1.0 s |
| `vru_proximity` | Pedestrian or cyclist within 2.0 m while SDC speed > 2 m/s | Min distance × speed |
| `signal_stop_violation` | SDC crosses a lane stop point while its signal is STOP | Speed at crossing |

Thresholds live in config and are versioned with the detector.

## Metrics

- Events per 1,000 scenarios, per 100 SDC-km, per SDC-hour
- Triage precision per detector: `true_issue / (true_issue + false_positive)`
- Version regression: per event type, exact conditional Poisson test on SDC-km exposure,
  reported as rate ratio with 95% CI

## Milestones

| # | Scope | Status |
|---|---|---|
| M0 | TFRecord to Parquet ingest, manifest, inventory, synthetic fixtures | Done |
| M1 | Data quality checks and `dq_check_results` | |
| M2 | Kinematics, `hard_braking` and `low_ttc_conflict`, known-answer tests | |
| M3 | Event store, analytics SQL, denominators | |
| M4 | Streamlit dashboard and append-only triage (MVP complete) | |
| M5 | Synthetic versions, injected perturbation, regression test | |
| M6 | LLM event summaries with lineage | |
| M7 | Remaining detectors, CI | |
