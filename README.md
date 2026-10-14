# Autonomous Vehicle Safety Analytics

Safety event detection, data quality validation, human triage, and regression reporting on the
[Waymo Open Motion Dataset](https://waymo.com/open/data/motion/) (WOMD). The goal is to model the
workflow an AV safety team uses to turn raw fleet logs into reviewed, reportable safety events.

**Status:** M0 (ingest) complete. See [docs/design.md](docs/design.md) for architecture and roadmap.

## Quick start (macOS / Linux, Python 3.11+)

No TensorFlow required. TFRecords are read with a small pure-Python reader and parsed with
protobuf bindings generated from Waymo's published `.proto` schemas.

```bash
make setup && source .venv/bin/activate

# Option A: run end to end on synthetic data (no dataset access needed)
avsafety synth && avsafety ingest && avsafety inventory

# Option B: real data
./scripts/download_subset.sh          # 1 validation shard; N_SHARDS=3 for more
avsafety ingest --max-scenarios 100   # quick dev run; drop the flag for the full shard
avsafety inventory
```

Real data requires accepting the Waymo license at https://waymo.com/open and running
`gcloud auth login` with the same Google account. The bucket name encodes the dataset version;
override `WOMD_BUCKET` if the download page shows a different one.

## What ingest produces

Per shard, under `data/processed/<shard>/`:

| Table | Grain | Notes |
|---|---|---|
| `scenarios` | scenario | Exposure (SDC distance, duration), agent counts, SDC valid-frame fraction |
| `agent_states` | scenario × track × step | All states kept, including `valid=False`; filtering is the validation layer's job |
| `map_features` | scenario × feature | Lanes, road lines/edges, crosswalks, stop signs; geometry as coordinate lists |
| `signal_states` | scenario × step × lane | Traffic signal state and stop point |
| `manifest.json` | shard | Run ID, CRC status, row counts, every record that failed to parse and why |

Design choices worth noting:

- **Failures are recorded, not dropped.** An undecodable record is logged with its index and
  offset; the rest of the shard still loads. A broken frame header stops the shard (later
  offsets can't be trusted) and keeps everything before it.
- **Exposure ignores tracking gaps.** SDC distance sums only consecutive valid-valid steps, so a
  dropout never becomes a straight-line jump in the denominator.
- **Idempotent per shard.** Re-ingesting a shard replaces its folder.

## Repo layout

```
proto/                 vendored Waymo schemas (Apache-2.0), trimmed of lidar/camera fields
src/avsafety/
  ingest/              TFRecord reader, normalization, shard pipeline, inventory
  protos/              generated bindings (`make protos` to regenerate)
  synthetic.py         WOMD-format scenarios with known ground truth
  cli.py
tests/
scripts/download_subset.sh
docs/design.md
```

## Data and license

Waymo Open Dataset content is licensed for non-commercial use and may not be redistributed.
Nothing under `data/` is committed. Code in this repository is MIT licensed; the vendored
`.proto` files retain their Apache-2.0 license.
