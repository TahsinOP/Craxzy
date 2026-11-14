"""YAML configuration -> the frozen dataclasses used across the package.

``load()`` returns a ``Config`` whose fields are the dataclasses of the modules
that own them; every script takes ``--set a.b=value`` overrides.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from clay_sculptor import ROOT
from clay_sculptor.control.primitives import PrimCfg
from clay_sculptor.envs.shape import MapCfg
from clay_sculptor.models.scene import ClayCfg, SimCfg

DEFAULT = ROOT / "configs" / "default.yaml"


def _coerce(value, default):
    """Cast a YAML value to the type of the dataclass default (PyYAML reads ``1.5e5``
    as a string; numbers written as ints must become floats)."""
    if isinstance(default, tuple):
        return tuple(_coerce(v, default[0]) for v in value)
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, (int, float, str)):
        return type(default)(value)
    return value


def _make(cls, d: dict[str, Any]):
    defaults = {f.name: f.default for f in fields(cls)}
    unknown = set(d) - set(defaults)
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: _coerce(v, defaults[k]) for k, v in d.items()})


@dataclass(frozen=True)
class Config:
    clay: ClayCfg
    sim: SimCfg
    prim: PrimCfg
    map: MapCfg
    raw: dict

    def env(self, nthread: int = 0, nsim: int = 64):
        from clay_sculptor.envs.clay_env import ClayEnv

        return ClayEnv(self.clay, self.sim, self.prim, self.map, nthread=nthread, nsim=nsim)


def load(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    with open(path or DEFAULT) as f:
        raw = copy.deepcopy(yaml.safe_load(f))
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return Config(_make(ClayCfg, raw.get("clay", {})), _make(SimCfg, raw.get("sim", {})),
                  _make(PrimCfg, raw.get("prim", {})), _make(MapCfg, raw.get("map", {})), raw)
