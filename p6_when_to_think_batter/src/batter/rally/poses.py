"""Nominal strike posture and ready pose for the paddle (the rally's analogue of scripts/design_ready.py).

The nominal strike posture is, for a ball through the middle of the strike zone with the face
normal tilted ``elevation`` above horizontal, the one with the highest attainable paddle
speed along the normal within the joint speed limits, at least 0.4 rad from every joint limit
(random-restart IK).  The ready pose is that posture wound back along the stroke by 1.5 times
each joint's run-up distance at ``design_speed``.  Results are cached in data/rally_poses.json.
"""
from __future__ import annotations

import json

import numpy as np

from batter import DATA
from batter.control.arm import Arm
from batter.models.scene import QD_MAX, QDD_MAX

PATH = DATA / "rally_poses.json"
def _key(cfg: dict) -> dict:
    """What the poses depend on: the paddle's geometry and the strike zone (not the aim or the restitution)."""
    pd, rc = cfg["paddle"], cfg["rally"]
    return {"paddle": [pd["radius"], pd["thickness"], pd["handle"], pd["mass"]], "strike_x": rc["strike_x"], "zone_z": rc["zone_z"]}


def design(arm: Arm, cfg: dict, elevation_deg: float = 15.0, design_speed: float = 1.2, restarts: int = 600,
           seed: int = 0) -> dict:
    sx = float(np.mean(cfg["rally"]["strike_x"]))
    point = np.array([sx, 0.0, float(np.mean(cfg["rally"]["zone_z"]))])
    el = np.deg2rad(elevation_deg)
    n = np.array([np.cos(el), 0.0, np.sin(el)])
    pos = point - (cfg["ball"]["radius"] + cfg["paddle"]["thickness"] / 2) * n
    rng = np.random.default_rng(seed)
    best, best_s = None, 0.0
    for _ in range(restarts):
        s0 = rng.uniform(arm.q_min + 0.1, arm.q_max - 0.1)
        q, ep, er = arm.ik(pos, n, s0, s0, iters=200, null_gain=0.0)
        if ep > 1e-3 or er > 1e-2 or np.min(np.minimum(q - arm.q_min, arm.q_max - q)) < 0.4 or abs(q[0]) > 0.6:
            continue
        s_max = 1.0 / np.max(np.abs(arm.strike_velocity(q, n, 1.0)) / QD_MAX)
        if s_max > best_s:
            best, best_s = q, s_max
    if best is None:
        raise RuntimeError("no strike posture found")
    v = arm.strike_velocity(best, n, design_speed)
    ready = np.clip(best - 1.5 * v * np.abs(v) / (2 * 0.9 * QDD_MAX), arm.q_min + 0.1, arm.q_max - 0.1)
    return {"nominal": best.tolist(), "ready": ready.tolist(), "max_speed": float(best_s), "elevation_deg": elevation_deg}


def load(arm: Arm, cfg: dict) -> dict:
    """The cached poses, designed on first use."""
    key = _key(cfg)
    if PATH.exists():
        cached = json.loads(PATH.read_text())
        if cached.get("key") == json.loads(json.dumps(key)):
            return cached
    out = design(arm, cfg)
    out["key"] = key
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(out, indent=2))
    return out
