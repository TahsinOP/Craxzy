"""Two-arm ping-pong rally (stretch goal of P6).

Two Pandas with thin paddles keep a ping-pong ball going across a regulation table.  Each
arm plans in its own frame, where it is the batter's arm A: arm B sits at x = L facing -x,
and the map  (x, y, z) -> (L - x, -y, z)  (a rotation of pi about the vertical) takes the
world to B's frame.  Everything else (flight model, EKF, aiming, impact inversion, swing QP)
is the batter's, with two additions: the ball bounces on the table, and a return must clear
the net and arrive where the other arm can reach it.
"""
from __future__ import annotations

import copy

import yaml

from batter import ROOT

RALLY_CFG = ROOT / "configs" / "rally.yaml"


def load(cfg: dict, overrides: list[str] | None = None) -> dict:
    """The rally configuration: configs/rally.yaml merged over the batter's config."""
    out = copy.deepcopy(cfg)
    with open(RALLY_CFG) as f:
        extra = yaml.safe_load(f)
    for key, val in extra.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **val}
        else:
            out[key] = val
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    t = out["table"]
    t["L"] = t["length"] + 2 * t["gap"]                 # distance between the two robot bases
    return out
