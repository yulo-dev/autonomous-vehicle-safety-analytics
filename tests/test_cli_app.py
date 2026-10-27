"""End-to-end: CLI pipeline on the synthetic fleet, then the dashboard on its output."""

import json
import sqlite3

import pytest

from avsafety.cli import main


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    raw = root / "raw" / "synthetic.tfrecord-00000-of-00001"
    processed, db_path = root / "processed", root / "avsafety.db"
    assert main(["synth", "--out", str(raw), "-n", "30", "--seed", "11"]) == 0
    assert main(["ingest", str(raw), "--out", str(processed)]) == 0
    assert main(["ingest", str(raw), "--out", str(processed), "--strict"]) == 2  # 1 corrupted record
    assert main(["validate", "--processed", str(processed), "--db", str(db_path)]) == 0
    assert main(["detect", "--processed", str(processed), "--db", str(db_path)]) == 0
    return root, processed, db_path


def test_pipeline_outputs(workspace, capsys):
    root, processed, db_path = workspace
    manifest = json.loads((processed / "synthetic.tfrecord-00000-of-00001" / "manifest.json").read_text())
    assert len(manifest["failures"]) == 1

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM scenario_eligibility WHERE eligible = 0").fetchone()[0] >= 2

    assert main(["report", "--db", str(db_path)]) == 0
    out = capsys.readouterr().out
    assert "Event rates" in out and "Data quality" in out


def test_triage_cli(workspace, capsys):
    _, _, db_path = workspace
    eid = sqlite3.connect(db_path).execute("SELECT event_id FROM events LIMIT 1").fetchone()[0]
    assert main(["triage", eid, "--db", str(db_path), "--decision", "false_positive",
                 "--reviewer", "cli-test", "--rationale", "velocity glitch"]) == 0
    assert main(["triage", "missing", "--db", str(db_path), "--decision", "true_issue",
                 "--reviewer", "x", "--rationale", "y"]) == 1


def test_dashboard_pages_render(workspace, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from avsafety import app as app_pkg

    _, processed, db_path = workspace
    monkeypatch.setenv("AVSAFETY_DB", str(db_path))
    monkeypatch.setenv("AVSAFETY_PROCESSED", str(processed))
    path = str(next(iter(app_pkg.__path__))) + "/streamlit_app.py"

    at = AppTest.from_file(path, default_timeout=60).run()
    assert not at.exception
    for page in ["Event queue", "Review", "Data quality", "Overview"]:
        at.radio(key="page").set_value(page).run()
        assert not at.exception, page

    at.radio(key="page").set_value("Review").run()
    at.text_input(key="reviewer").set_value("app-test").run()
    at.text_area(key="rationale").set_value("SDC braked for a hard-braking lead; appropriate response.")
    at.radio(key="decision").set_value("true_issue")
    next(b for b in at.button if b.label == "Record decision").click().run()
    assert not at.exception and not at.error
    n = sqlite3.connect(db_path).execute(
        "SELECT COUNT(*) FROM triage_decisions WHERE reviewer = 'app-test'").fetchone()[0]
    assert n == 1
