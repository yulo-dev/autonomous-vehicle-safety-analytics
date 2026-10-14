"""Shard-level ingest: TFRecord -> Parquet tables + run manifest.

Layout per shard:
    <out_dir>/<shard_name>/scenarios.parquet
                          /agent_states.parquet
                          /map_features.parquet
                          /signal_states.parquet
                          /manifest.json

Records that fail to parse are listed in the manifest with their index and
error, never silently skipped. Re-running a shard overwrites its folder, so
ingest is idempotent per shard.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from google.protobuf.message import DecodeError

from avsafety.ingest import normalize
from avsafety.ingest.tfrecord import TFRecordError, crc_available, read_records

TABLE_SCHEMAS = {
    "scenarios": normalize.SCENARIO_SCHEMA,
    "agent_states": normalize.AGENT_STATE_SCHEMA,
    "map_features": normalize.MAP_FEATURE_SCHEMA,
    "signal_states": normalize.SIGNAL_STATE_SCHEMA,
}


@dataclass
class RecordFailure:
    record_index: int
    offset: int | None
    error_type: str
    message: str


@dataclass
class ShardManifest:
    run_id: str
    source_file: str
    started_at: str
    finished_at: str | None = None
    crc_verified: bool = False
    max_scenarios: int | None = None
    records_read: int = 0
    scenarios_written: int = 0
    framing_error: str | None = None
    row_counts: dict[str, int] = field(default_factory=dict)
    failures: list[RecordFailure] = field(default_factory=list)


class _BatchedWriter:
    """Accumulate column dicts and flush to one Parquet file per table."""

    def __init__(self, out_dir: Path, flush_every: int = 50):
        self.out_dir = out_dir
        self.flush_every = flush_every
        self.buffers: dict[str, list[dict[str, list]]] = {t: [] for t in TABLE_SCHEMAS}
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.rows: dict[str, int] = {t: 0 for t in TABLE_SCHEMAS}
        self.pending = 0

    def add(self, tables: dict[str, dict[str, list]]) -> None:
        for name, cols in tables.items():
            self.buffers[name].append(cols)
        self.pending += 1
        if self.pending >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        for name, chunks in self.buffers.items():
            if not chunks:
                continue
            schema = TABLE_SCHEMAS[name]
            merged = {c: [v for chunk in chunks for v in chunk[c]] for c in schema.names}
            table = pa.Table.from_pydict(merged, schema=schema)
            if name not in self.writers:
                self.writers[name] = pq.ParquetWriter(
                    self.out_dir / f"{name}.parquet", schema, compression="zstd"
                )
            self.writers[name].write_table(table)
            self.rows[name] += table.num_rows
            self.buffers[name] = []
        self.pending = 0

    def close(self) -> None:
        self.flush()
        for name, schema in TABLE_SCHEMAS.items():
            if name in self.writers:
                self.writers[name].close()
            else:  # emit empty file so downstream reads never special-case
                pq.write_table(schema.empty_table(), self.out_dir / f"{name}.parquet")


def ingest_shard(
    tfrecord_path: str | Path,
    out_root: str | Path,
    max_scenarios: int | None = None,
    run_id: str | None = None,
) -> ShardManifest:
    src = Path(tfrecord_path)
    out_dir = Path(out_root) / src.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    run_id = run_id or f"ingest-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    manifest = ShardManifest(
        run_id=run_id,
        source_file=src.name,
        started_at=datetime.now(UTC).isoformat(),
        crc_verified=crc_available(),
        max_scenarios=max_scenarios,
    )
    writer = _BatchedWriter(out_dir)

    try:
        for rec in read_records(src):
            if max_scenarios is not None and manifest.scenarios_written >= max_scenarios:
                break
            manifest.records_read += 1
            try:
                tables = normalize.normalize_scenario(rec.data, src.name, rec.index, run_id)
            except (DecodeError, ValueError) as e:
                manifest.failures.append(
                    RecordFailure(rec.index, rec.offset, type(e).__name__, str(e))
                )
                continue
            writer.add(tables)
            manifest.scenarios_written += 1
    except TFRecordError as e:
        # Framing is broken; later offsets are untrustworthy. Keep what we have.
        manifest.framing_error = str(e)
    finally:
        writer.close()

    manifest.row_counts = dict(writer.rows)
    manifest.finished_at = datetime.now(UTC).isoformat()
    (out_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))
    return manifest
