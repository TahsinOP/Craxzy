"""Watch catches live in the MuJoCo viewer.

    env -u PYTHONPATH .venv/bin/python scripts/view.py --obj ball --throws 10 --slow 4
    env -u PYTHONPATH .venv/bin/python scripts/view.py --obj plate --effector suction
    env -u PYTHONPATH .venv/bin/python scripts/view.py --obj card --effector suction

Each throw runs the full stack (predictor -> trajectory QP -> computed torque)
with 50 Hz replanning, one world at a time, and is mirrored into a passive
viewer.  ``--no-window`` runs the same loop without a display (for checking).
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import mujoco
import mujoco.viewer
import numpy as np

from card_catcher import config
from card_catcher.envs.catch import DEFAULT_PARAMS, DEFAULT_PREDICTOR, CatchEnv, run_fixed_rate
from card_catcher.envs.flight import generate
from card_catcher.estim.predictor import make_predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="ball", choices=["ball", "card", "plate"])
    ap.add_argument("--predictor", default=None, choices=["physics", "hybrid", "oracle"],
                    help="default: physics for the ball, hybrid for card and plate")
    ap.add_argument("--effector", default="gripper", choices=["gripper", "suction"],
                    help="suction: a disc with MuJoCo's geom adhesion between the open fingers (good for plate and card)")
    ap.add_argument("--suction-force", type=float, default=None, help="N per contact; default 1 for the card, 20 otherwise")
    ap.add_argument("--throws", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=float, default=50.0, help="replanning rate in Hz")
    ap.add_argument("--slow", type=float, default=4.0, help="slow-motion factor (1 = real time)")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    cfg = config.load()
    kind = args.predictor or DEFAULT_PREDICTOR[args.obj]
    D = generate(args.obj, args.throws, cfg, seed=900 + args.seed)
    prm = DEFAULT_PARAMS[args.obj]
    if args.effector == "suction":
        force = args.suction_force if args.suction_force is not None else (1.0 if args.obj == "card" else 20.0)
        prm = dataclasses.replace(prm, effector="suction", suction_force=force)
    env = CatchEnv(args.obj, cfg, 1, make_predictor(kind, args.obj, 1, cfg), prm, seed=args.seed)
    data = mujoco.MjData(env.model)
    h = env.model.opt.timestep
    viewer = None
    if not args.no_window:
        viewer = mujoco.viewer.launch_passive(env.model, data)
        viewer.cam.lookat[:] = [0.85, 0.0, 0.85]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, 110.0, -12.0

    def cb(e):
        if viewer is None:
            return
        if not viewer.is_running():
            raise KeyboardInterrupt
        t0 = time.perf_counter()
        with viewer.lock():
            e.batch.get_data(0, data)
            mujoco.mj_forward(e.model, data)
        viewer.sync()
        time.sleep(max(0.0, h * args.slow - (time.perf_counter() - t0)))

    caught = 0
    try:
        for i in range(args.throws):
            out = run_fixed_rate(env, D, np.array([i]), rate_hz=args.rate, frame_cb=cb)
            ok = bool(out["retained" if args.effector == "suction" else "caught"][0])   # suction has no pinch: judge "retained"
            caught += int(ok)
            print(f"throw {i + 1:3d}: {'CAUGHT' if ok else 'missed'}   ({caught}/{i + 1})", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
            time.sleep(0.5)                      # let the render thread finish before the model is freed


if __name__ == "__main__":
    main()
