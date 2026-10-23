"""Load the YAML configuration (config/selection.yaml by default)."""
from __future__ import annotations

import pathlib
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "selection.yaml"


def load_config(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    path = pathlib.Path(path) if path else DEFAULT_CONFIG
    with open(path) as fh:
        return yaml.safe_load(fh)


def data_dir(sub: str = "") -> pathlib.Path:
    d = ROOT / "data" / sub
    d.mkdir(parents=True, exist_ok=True)
    return d
