"""Design the nominal strike posture and the ready pose (constants in batter/control/arm.py).

Random-restart IK for a strike through the middle of the zone; keep the posture with the
highest attainable bat speed along the normal that stays 0.4 rad clear of every joint
limit; wind it back along the swing to get the ready pose.

    env -u PYTHONPATH .venv/bin/python scripts/design_ready.py
"""
from __future__ import annotations

import numpy as np

from batter import config
from batter.control.arm import Arm
from batter.models.scene import QD_MAX, QDD_MAX, build_cage_model

NORMAL_ELEVATION_DEG = 25.0
STRIKE_POINT = np.array([0.62, 0.0, 0.95])
DESIGN_SPEED = 1.8          # m/s
RUNUP = 1.5                 # multiples of v^2 / (2 a_max)


def main():
    cfg = config.load()
    arm = Arm(build_cage_model(cfg))
    el = np.deg2rad(NORMAL_ELEVATION_DEG)
    n = np.array([np.cos(el), 0.0, np.sin(el)])
    pos = STRIKE_POINT - (cfg["ball"]["radius"] + cfg["bat"]["thickness"] / 2) * n
    rng = np.random.default_rng(0)
    best, best_s = None, 0.0
    for _ in range(600):
        seed = rng.uniform(arm.q_min + 0.1, arm.q_max - 0.1)
        q, ep, er = arm.ik(pos, n, seed, seed, iters=200, null_gain=0.0)
        if ep > 1e-3 or er > 1e-2 or np.min(np.minimum(q - arm.q_min, arm.q_max - q)) < 0.4:
            continue
        if abs(q[0]) > 0.6:                      # face the pitcher
            continue
        s_max = 1.0 / np.max(np.abs(arm.strike_velocity(q, n, 1.0)) / QD_MAX)
        if s_max > best_s:
            best, best_s = q, s_max
    v = arm.strike_velocity(best, n, DESIGN_SPEED)
    ready = np.clip(best - RUNUP * v * np.abs(v) / (2 * 0.9 * QDD_MAX), arm.q_min + 0.1, arm.q_max - 0.1)
    p, R = arm.fk(ready)
    np.set_printoptions(precision=4, suppress=True)
    print(f"max bat speed at the nominal strike: {best_s:.2f} m/s")
    print("NOMINAL_STRIKE = np.array(%s)" % np.array2string(best, separator=", "))
    print("READY = np.array(%s)" % np.array2string(ready, separator=", "))
    print("ready sweet spot", p.round(3), "normal", R[:, 0].round(2), "blade axis", R[:, 2].round(2))


if __name__ == "__main__":
    main()
