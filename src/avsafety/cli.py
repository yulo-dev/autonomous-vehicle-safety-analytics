"""Command-line entry point: `avsafety <command>`."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from avsafety.ingest.inventory import summarize
from avsafety.ingest.pipeline import ingest_shard
from avsafety.ingest.tfrecord import crc_available

DEFAULT_RAW = Path("data/raw")
DEFAULT_PROCESSED = Path("data/processed")
DEFAULT_DB = Path("data/avsafety.db")


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
        print(
            f"{p.name}: {m.scenarios_written} scenarios, "
            f"{m.row_counts.get('agent_states', 0):,} agent states, "
            f"{len(m.failures)} failed records (see manifest.json)"
        )
        if m.framing_error:
            print(f"  framing error: {m.framing_error}", file=sys.stderr)
            exit_code = 2
        elif m.failures and args.strict:
            exit_code = 2
    return exit_code


def _cmd_inventory(args: argparse.Namespace) -> int:
    print(json.dumps(summarize(args.processed), indent=2, default=str))
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    from avsafety.ingest.tfrecord import write_records
    from avsafety.synthetic import demo_fleet

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scenarios, truth = demo_fleet(n=args.n, seed=args.seed)
    # One undecodable record on purpose, to exercise failure tracking.
    write_records(args.out, [s.SerializeToString() for s in scenarios] + [b"\xff\xff corrupted record"])
    truth_path = args.out.parent / "synthetic_truth.json"
    truth_path.write_text(json.dumps(truth, indent=2))
    print(f"wrote {len(scenarios) + 1} records to {args.out} (template labels: {truth_path})")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from avsafety import db
    from avsafety.config import load_config
    from avsafety.validate.runner import run_validation

    out = run_validation(args.processed, db.connect(args.db), load_config("validation", args.config))
    print(json.dumps(out, indent=2))
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    from avsafety import db
    from avsafety.config import load_config
    from avsafety.detect.runner import ConfigVersionError, run_detection

    try:
        out = run_detection(args.processed, db.connect(args.db), load_config("detectors", args.config))
    except ConfigVersionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    import pandas as pd

    from avsafety import db
    from avsafety.analytics import metrics

    conn = db.connect(args.db)
    version = args.version or metrics.latest_detector_version(conn)
    if version is None:
        print("no detection runs yet", file=sys.stderr)
        return 1
    pd.set_option("display.width", 160)
    exp = metrics.exposure(conn)
    print(f"Detector version {version}")
    print(f"Exposure: {exp['n_scenarios']} eligible scenarios ({exp['n_excluded']} excluded by DQ), "
          f"{exp['sdc_km']:.2f} SDC-km, {exp['sdc_hours']:.3f} SDC-hours\n")

    rates = metrics.event_rates(conn, version)
    if not rates.empty:
        cols = ["event_type", "n_events", "s1", "s2", "s3", "per_1k_scenarios", "per_100km",
                "per_100km_lo", "per_100km_hi", "per_hour"]
        print("Event rates (95% exact Poisson CI on per-100km)")
        print(rates[cols].round(2).to_string(index=False), "\n")

    rev = metrics.review_summary(conn, version)
    if not rev.empty:
        print("Triage")
        print(rev[["event_type", "n_events", "true_issue", "false_positive", "needs_review",
                   "precision", "review_coverage"]].round(3).to_string(index=False), "\n")

    dq = metrics.dq_summary(conn)
    print("Data quality (latest validation run)")
    print(dq[["check_name", "level", "passed", "n_failed", "n_evaluated"]].to_string(index=False))
    return 0


def _cmd_triage(args: argparse.Namespace) -> int:
    from avsafety import db
    from avsafety.triage.service import TriageError, record_decision

    try:
        d = record_decision(db.connect(args.db), args.event_id, args.reviewer, args.decision, args.rationale)
    except TriageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"recorded decision {d.decision_id}: {d.event_id} -> {d.decision} by {d.reviewer} at {d.decided_at}")
    return 0


def _cmd_app(args: argparse.Namespace) -> int:
    app = Path(__file__).parent / "app" / "streamlit_app.py"
    env = {**os.environ, "AVSAFETY_DB": str(args.db.resolve()), "AVSAFETY_PROCESSED": str(args.processed.resolve())}
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)], env=env)


def main(argv: list[str] | None = None) -> int:
    from avsafety.triage.service import DECISIONS

    parser = argparse.ArgumentParser(prog="avsafety")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_paths(p, processed=True, db=True):
        if processed:
            p.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
        if db:
            p.add_argument("--db", type=Path, default=DEFAULT_DB)

    p = sub.add_parser("ingest", help="TFRecord shards -> Parquet")
    p.add_argument("paths", nargs="*", help=f"shard files (default: {DEFAULT_RAW}/*.tfrecord*)")
    p.add_argument("--out", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--max-scenarios", type=int, default=None, help="cap per shard, for quick dev runs")
    p.add_argument("--strict", action="store_true", help="non-zero exit if any record fails to parse")
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("inventory", help="summarize ingested data")
    add_paths(p, db=False)
    p.set_defaults(func=_cmd_inventory)

    p = sub.add_parser("synth", help="write a synthetic WOMD-format shard (no Waymo access needed)")
    p.add_argument("--out", type=Path, default=DEFAULT_RAW / "synthetic.tfrecord-00000-of-00001")
    p.add_argument("-n", type=int, default=60, help="number of randomized scenarios")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=_cmd_synth)

    p = sub.add_parser("validate", help="load scenarios and run data quality checks")
    add_paths(p)
    p.add_argument("--config", type=Path, help="override validation.yaml")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("detect", help="run safety event detectors on eligible scenarios")
    add_paths(p)
    p.add_argument("--config", type=Path, help="override detectors.yaml")
    p.set_defaults(func=_cmd_detect)

    p = sub.add_parser("report", help="print exposure-normalized metrics")
    add_paths(p, processed=False)
    p.add_argument("--version", help="detector version (default: latest run)")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("triage", help="record a review decision for an event")
    add_paths(p, processed=False)
    p.add_argument("event_id")
    p.add_argument("--decision", required=True, choices=DECISIONS)
    p.add_argument("--reviewer", required=True)
    p.add_argument("--rationale", required=True)
    p.set_defaults(func=_cmd_triage)

    p = sub.add_parser("app", help="launch the Streamlit dashboard")
    add_paths(p)
    p.set_defaults(func=_cmd_app)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
