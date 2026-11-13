"""M0 design check: can the Panda face its palm towards a descending object?

For an upright (table) mount and an inverted (gantry) mount, sweep TCP positions in the
vertical plane in front of the base and palm elevations of 20, 40 and 60 degrees above
the horizontal (objects arrive at 45-85 degrees), solve IK, and record the smallest
distance to a joint limit.  A pose counts as usable if IK converges with >= 0.3 rad of
margin on every joint.  Writes results/m0_mount.json.
"""
from __future__ import annotations

import json

import numpy as np

from card_catcher import RESULTS, config
from card_catcher.control.arm import READY_SEED, Arm, hand_rotation
from card_catcher.models.scene import BASE_HEIGHT, TABLE_TOP, build_catch_model

EU = np.array([0.0, 1.0, 0.0])


def scan(mount: str) -> dict:
    arm = Arm(build_catch_model("ball", config.load()["sim"], mount=mount))
    base_z = BASE_HEIGHT if mount == "inverted" else TABLE_TOP
    sign = -1.0 if mount == "inverted" else 1.0          # workspace is below an inverted base
    xs, dzs = np.arange(0.35, 0.86, 0.05), np.arange(0.20, 1.01, 0.05)
    seeds = [READY_SEED, np.array([0.0, 0.2, 0.0, -1.7, 0.0, 3.3, 0.785]), np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.785])]
    out = {}
    for pitch in (20, 40, 60):
        th = np.deg2rad(pitch)
        n = np.array([np.cos(th), 0.0, np.sin(th)])
        R = hand_rotation(n, np.cross(n, EU))
        best = np.full((len(dzs), len(xs)), -1.0)
        for i, dz in enumerate(dzs):
            for j, x in enumerate(xs):
                for q0 in seeds:
                    q, ep, er = arm.ik(np.array([x, 0.0, base_z + sign * dz]), R, q0, q0, iters=150, null_gain=0.02)
                    if ep < 1e-3 and er < 1e-2:
                        best[i, j] = max(best[i, j], float(np.minimum(q - arm.q_min, arm.q_max - q).min()))
        out[pitch] = {"reachable": float((best >= 0).mean()), "usable_margin_0.3": float((best >= 0.3).mean()),
                      "best_margin": float(best.max())}
    return out


def main():
    res = {m: scan(m) for m in ("upright", "inverted")}
    (RESULTS / "m0_mount.json").write_text(json.dumps(res, indent=2))
    for m, r in res.items():
        for pitch, v in r.items():
            print(f"{m:9s} palm {pitch} deg up: IK reachable {v['reachable']*100:5.1f} % of the grid, usable (>= 0.3 rad margin) "
                  f"{v['usable_margin_0.3']*100:5.1f} %, best margin {v['best_margin']:.2f} rad")


if __name__ == "__main__":
    main()
