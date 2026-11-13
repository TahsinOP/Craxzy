"""YAML configuration with dotted-key overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from batter import ROOT

DEFAULT = ROOT / "configs" / "default.yaml"


def load(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config; `overrides` is a list of ``a.b.c=value`` strings."""
    with open(path or DEFAULT) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg
