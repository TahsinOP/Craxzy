"""YAML configuration with dotted-key overrides and derived geometry."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from card_catcher import ROOT

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


@dataclass(frozen=True)
class Plane:
    """Catch plane {x : n.(x - c0) = 0} with an orthonormal in-plane basis (eu, ew).

    ``signed(x) = n.(x - c0)`` is positive on the thrower's side.  In-plane
    coordinates are ``(u, w) = (eu.(x - c0), ew.(x - c0))``.
    """

    c0: np.ndarray
    n: np.ndarray
    eu: np.ndarray
    ew: np.ndarray
    u_range: tuple[float, float]
    w_range: tuple[float, float]

    @staticmethod
    def from_cfg(cfg: dict[str, Any]) -> "Plane":
        p = cfg["plane"]
        th = np.deg2rad(p["tilt_deg"])
        n = np.array([np.cos(th), 0.0, np.sin(th)])
        eu = np.array([0.0, 1.0, 0.0])
        ew = np.cross(n, eu)  # (-sin th, 0, cos th): up-slope direction
        return Plane(np.asarray(p["c0"], float), n, eu, ew, tuple(p["u_range"]), tuple(p["w_range"]))

    def signed(self, x: np.ndarray) -> np.ndarray:
        return (x - self.c0) @ self.n

    def coords(self, x: np.ndarray) -> np.ndarray:
        d = x - self.c0
        return np.stack([d @ self.eu, d @ self.ew], axis=-1)

    def point(self, uw: np.ndarray) -> np.ndarray:
        uw = np.asarray(uw, float)
        return self.c0 + uw[..., :1] * self.eu + uw[..., 1:2] * self.ew

    def inside(self, x: np.ndarray, margin: float = 0.0) -> np.ndarray:
        c = self.coords(x)
        return (
            (c[..., 0] >= self.u_range[0] - margin)
            & (c[..., 0] <= self.u_range[1] + margin)
            & (c[..., 1] >= self.w_range[0] - margin)
            & (c[..., 1] <= self.w_range[1] + margin)
        )

    @property
    def basis(self) -> np.ndarray:
        """2x3 matrix E with rows (eu, ew): in-plane coordinates = E (x - c0)."""
        return np.stack([self.eu, self.ew])
