"""Command-line entry point: `avsafety <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from avsafety.ingest.inventory import summarize
from avsafety.ingest.pipeline import ingest_shard
from avsafety.ingest.tfrecord import crc_available

DEFAULT_RAW = Path("data/raw")
DEFAULT_PROCESSED = Path("data/processed")


def _cmd_ingest(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.paths] or sorted(DEFAULT_RAW.glob("*.tfrecord*"))
    if not paths:
        print(f"no TFRecord files given or found in {DEFAULT_RAW}", file=sys.stderr)
        return 1
    if not crc_available():
        print("warning: google-crc32c not installed; records are read without CRC checks", file=sys.stderr)
    exit_code = 0
    for p in paths:
        m = ingest_shard(p, args.out, max_scenarios=args.max_scenarios)
        status = "ok"
        if m.failures or m.framing_error:
            status = "with errors"
            exit_code = 2
        print(
            f"{p.name}: {m.scenarios_written} scenarios, "
            f"{m.row_counts.get('agent_states', 0):,} agent states, "
            f"{len(m.failures)} failed records ({status})"
        )
        if m.framing_error:
            print(f"  framing error: {m.framing_error}", file=sys.stderr)
    return exit_code


def _cmd_inventory(args: argparse.Namespace) -> int:
    print(json.dumps(summarize(args.processed), indent=2, default=str))
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    from avsafety.ingest.tfrecord import write_records
    from avsafety.synthetic import demo_scenarios

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scenarios = demo_scenarios()
    write_records(args.out, [s.SerializeToString() for s in scenarios])
    print(f"wrote {len(scenarios)} synthetic scenarios to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avsafety")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="TFRecord shards -> Parquet")
    p.add_argument("paths", nargs="*", help=f"shard files (default: {DEFAULT_RAW}/*.tfrecord*)")
    p.add_argument("--out", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--max-scenarios", type=int, default=None, help="cap per shard, for quick dev runs")
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("inventory", help="summarize ingested data")
    p.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    p.set_defaults(func=_cmd_inventory)

    p = sub.add_parser("synth", help="write a synthetic WOMD-format shard (no Waymo access needed)")
    p.add_argument("--out", type=Path, default=DEFAULT_RAW / "synthetic.tfrecord-00000-of-00001")
    p.set_defaults(func=_cmd_synth)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
