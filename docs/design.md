# Design

## Goals

Model an AV safety data reporting workflow end to end: ingest driving logs, validate them, derive
safety-relevant events with reproducible rules, route events through human triage with a full
audit trail, and report rates with explicit denominators.

## Known limitations (read first)

1. **WOMD is curated.** Scenarios were mined for interesting interactions, so event rates are
   inflated relative to ordinary driving. Rates are per curated-scenario exposure.
2. **No ground-truth safety labels.** Every event is a rule-based derivation with documented
   thresholds. `dq_confidence` reflects data quality, not the probability that an event is real.
3. **Short windows.** Validation scenarios are 9.1 s at 10 Hz. Events at the window edges are
   flagged `truncated_start` / `truncated_end` in their metrics.
4. **Perception-derived kinematics.** Tracks come from an offboard detector. Acceleration is
   smoothed and computed only on contiguous valid runs; sensor glitches can still produce
   hard-braking false positives, which is exactly what triage is for.
5. **No contact detector.** Boxes overlapping at t=0 are treated as tracking artifacts and
   skipped by the TTC detector.

## Architecture

```
WOMD TFRecords
   │  ingest/      pure-Python TFRecord reader + protobuf parse, per-shard manifest
   ▼
Parquet: scenarios, agent_states, map_features, signal_states
   │  validate/    14 checks → dq_check_results; flag, never drop
   ▼
features/          Savitzky-Golay velocity and acceleration per valid run
   │
   ▼
detect/            versioned rule-based detectors → events (SQLite)
   ├─► triage/     append-only decisions; current status is a view
   └─► analytics/  exposure-normalized rates, severity mix, triage precision
          │
          ▼
       Streamlit dashboard (overview, queue, drill-down + decision form, data quality)
```

High-volume state data stays in Parquet. Runs, checks, events, and triage live in SQLite with a
Postgres-compatible schema (`src/avsafety/sql/schema.sql`).

## Data model

| Table / view | Purpose |
|---|---|
| `pipeline_runs` | One row per validate or detect run: stage, detector version, config hash, full config, git SHA, timings, summary |
| `scenarios` | Scenario metadata and exposure (SDC distance, duration, SDC valid fraction) |
| `dq_check_results` | Per-entity failures plus one summary row per check per run, so "passed" is recorded, not inferred |
| `events` | Detected events: type, window, severity, dq_confidence, primary and counterpart tracks, metrics JSON |
| `triage_decisions` | Append-only review decisions; UPDATE and DELETE are blocked by triggers |
| `event_status` (view) | Latest decision per event; untouched events are `needs_review` |
| `scenario_eligibility` (view) | Scenarios with no scenario-level error in the latest validation run |

## Data quality checks

| Check | Entity | Level |
|---|---|---|
| `duplicate_scenario_id` | scenario | error |
| `step_count_mismatch` | scenario | error |
| `timestamps_irregular` (missing, non-monotonic, off 10 Hz) | scenario | error |
| `sdc_low_validity` (< 50% error, < 90% warn) | scenario | error / warn |
| `sdc_track_error` (SDC failed an error-level track check) | scenario | error |
| `sdc_stationary` | scenario | warn |
| `missing_values` (NaN/inf in a valid state) | track | error |
| `implausible_dimensions` | track | error |
| `implausible_speed` (> 70 m/s) | track | error |
| `position_jump` (implied speed > 70 m/s) | track | error |
| `speed_inconsistency` (reported vs position-derived > 5 m/s, ≥ 3 steps) | track | warn |
| `heading_out_of_range` | track | warn |
| `unknown_object_type` | track | warn |
| `tracking_gaps` | track | warn |

Scenario errors exclude the scenario from detection and exposure. Track errors exclude that track
from detection. Warnings exclude nothing and lower `dq_confidence`.

## Event definitions (detector version 1.0.0)

Thresholds live in `src/avsafety/configs/detectors.yaml`.

| Event | Rule | Severity (S1 / S2 / S3) |
|---|---|---|
| `hard_braking` | Smoothed longitudinal accel < −3.0 m/s² for ≥ 0.3 s while speed > 1 m/s; runs merged across gaps ≤ 0.2 s | peak decel [−4.5, −3.0) / [−6.0, −4.5) / < −6.0 m/s² |
| `low_ttc_conflict` | TTC to a vehicle, pedestrian, or cyclist < 2.0 s for ≥ 0.3 s, within 50 m, SDC speed > 1 m/s; one episode per counterpart, gaps ≤ 0.5 s merged | min TTC [1.5, 2.0) / [1.0, 1.5) / < 1.0 s |

**TTC geometry.** Each box is covered by three discs along its heading, radius
√((L/6)² + (W/2)²). TTC is the earliest time, under constant velocity and heading, at which any
SDC disc touches any counterpart disc, solved in closed form per disc pair.

**Kinematics.** Velocity and acceleration come from a Savitzky-Golay fit (0.7 s window, order 2)
of the reported velocity, applied separately to each contiguous valid run. Runs shorter than the
window get no derived values. Longitudinal acceleration projects onto the direction of travel,
falling back to box heading below 0.5 m/s.

**dq_confidence.** Share of valid states for the involved tracks within ±1 s of the event,
×0.8 if an involved track has a DQ warning, ×0.9 if the scenario does.

**Versioning.** Event IDs hash scenario, type, tracks, start step, and detector version.
Re-running a version is idempotent. Running an existing version with a different config hash is
refused until the version is bumped.

## Metrics

- Rates per 1,000 scenarios, per 100 SDC-km, and per SDC-hour, with exact (Garwood) Poisson 95% CIs
- Severity mix and counterpart breakdown
- Triage precision per detector, `true_issue / (true_issue + false_positive)`, with Wilson 95% CI
- Review coverage, the share of events with at least one decision

## Milestones

| # | Scope | Status |
|---|---|---|
| M0 | TFRecord to Parquet ingest, manifest, inventory, synthetic fixtures | Done |
| M1 | Data quality checks, `dq_check_results`, eligibility | Done |
| M2 | Kinematics, `hard_braking`, `low_ttc_conflict`, versioned config, known-answer tests | Done |
| M3 | Event store, exposure-normalized analytics, report CLI | Done |
| M4 | Streamlit dashboard and append-only triage with audit history | Done |
| M5 | Synthetic software versions and regression testing between them | |
| M6 | LLM event summaries with prompt/response lineage; no labeling authority | |
| M7 | `close_following`, `vru_proximity`, signal violation detectors | |
