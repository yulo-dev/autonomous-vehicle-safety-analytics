# Autonomous Vehicle Safety Analytics

Safety event detection, data quality validation, human triage, and exposure-normalized reporting on
the [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/) (WOMD). The project models the
workflow an AV safety team runs to turn raw driving logs into reviewed, reportable safety events:

```
TFRecord logs → Parquet → data quality checks → kinematics → safety event detectors
             → event store → exposure-normalized metrics → dashboard + human triage (audited)
```

**Status:** MVP complete (M0 to M4). See [docs/design.md](docs/design.md) for the full design and roadmap.

## Quick start (macOS / Linux, Python 3.11+)

```bash
make setup && source .venv/bin/activate
make demo        # synthetic data → ingest → validate → detect → report → dashboard
```

`make demo` runs end to end without Waymo access, on a seeded synthetic fleet in WOMD format that
includes real hazards (hard-braking leads, cut-ins, pedestrian crossings), detector bait
(velocity sensor glitches), and deliberate data defects (teleporting tracks, an SDC that is
mostly invalid, a duplicate scenario ID, a corrupted record).

With real data:

```bash
./scripts/download_subset.sh       # 1 validation shard; N_SHARDS=3 for more
avsafety ingest && avsafety validate && avsafety detect && avsafety report
avsafety app
```

Real data requires accepting the Waymo license at https://waymo.com/open and running
`gcloud auth login` with the same Google account. Override `WOMD_BUCKET` if the download page
lists a different dataset version.

## Pipeline

| Stage | Command | What it does |
|---|---|---|
| Ingest | `avsafety ingest` | Pure-Python TFRecord reader (no TensorFlow) with CRC checks; Scenario protos flattened to Parquet tables for scenarios, agent states, map features, and signal states. Every unparseable record is logged in a per-shard manifest. |
| Validate | `avsafety validate` | 14 data quality checks (schema, ranges, missingness, timestamp regularity, position jumps, speed consistency, tracking gaps). Results go to `dq_check_results`; nothing is silently dropped. Scenario-level errors make a scenario ineligible; track-level errors exclude only that track. |
| Detect | `avsafety detect` | Savitzky-Golay kinematics, then `hard_braking` and `low_ttc_conflict` detectors driven by a versioned config. Reusing a version with changed thresholds is refused. |
| Report | `avsafety report` | Event rates per 1k scenarios, per 100 SDC-km, and per SDC-hour with exact Poisson CIs; severity mix; triage precision with Wilson CIs; DQ summary. |
| Triage | `avsafety app` / `avsafety triage` | Streamlit review workflow: queue with filters, event drill-down (trajectory, speed, acceleration or TTC against the threshold), and `true_issue` / `false_positive` / `needs_review` decisions with reviewer, rationale, and timestamp. Decisions are append-only, enforced by database triggers. |

## Design choices worth noting

- **Same population on both sides of every rate.** A scenario excluded for data quality drops its
  events and its exposure together.
- **Exposure ignores tracking gaps.** SDC distance sums only consecutive valid steps, so a dropout
  never turns into a straight-line jump in the denominator.
- **Confidence means data quality, not probability.** `dq_confidence` reflects valid-frame
  coverage and DQ warnings around the event; it is not a claim that the event is real.
- **Every event traces to an exact rule set.** Event IDs are deterministic per detector version;
  each run stores its config and config hash.
- **The dashboard plots what the detector saw.** Drill-down charts reuse the stored run config
  and the detector's own gating, so the threshold line on screen is the one that fired.

## Repo layout

```
src/avsafety/
  ingest/        TFRecord reader, normalization, shard pipeline, inventory
  validate/      data quality checks and runner
  features/      smoothing and derived kinematics
  detect/        hard braking, low-TTC conflict, detection runner
  analytics/     exposure, rates, triage metrics
  triage/        append-only review service
  app/           Streamlit dashboard
  sql/           event store schema and analytics queries
  configs/       versioned validation and detector configs
  synthetic.py   WOMD-format scenarios with known ground truth
proto/           vendored Waymo schemas (Apache-2.0), trimmed of lidar/camera fields
tests/           64 tests: known-answer detector tests, DQ checks, triage audit, CLI and dashboard
```

## Limitations

WOMD scenarios are mined for interesting interactions, so rates are inflated relative to
ordinary driving and are only comparable within this dataset. Events are rule-based derivations,
not ground-truth incident labels. See [docs/design.md](docs/design.md#known-limitations-read-first).

## Data and license

Waymo Open Dataset content is licensed for non-commercial use and may not be redistributed.
Nothing under `data/` is committed. Code in this repository is MIT licensed; the vendored
`.proto` files retain their Apache-2.0 license.
