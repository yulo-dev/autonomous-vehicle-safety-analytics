.PHONY: setup protos download synth ingest validate detect report app demo test lint

setup:
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev]"

protos:
	python -m grpc_tools.protoc -I proto --python_out=src \
		proto/avsafety/protos/map.proto proto/avsafety/protos/scenario.proto

download:
	./scripts/download_subset.sh

synth:
	avsafety synth

ingest:
	avsafety ingest

validate:
	avsafety validate

detect:
	avsafety detect

report:
	avsafety report

app:
	avsafety app

# Full pipeline on synthetic data, then open the dashboard.
demo: synth ingest validate detect report app

test:
	pytest -q

lint:
	ruff check src tests
