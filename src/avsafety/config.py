"""Load YAML configs bundled with the package, or from an override path."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml


def load_config(name: str, override: str | Path | None = None) -> dict:
    if override is not None:
        return yaml.safe_load(Path(override).read_text())
    return yaml.safe_load(resources.files("avsafety.configs").joinpath(f"{name}.yaml").read_text())
