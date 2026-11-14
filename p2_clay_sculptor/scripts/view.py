"""Watch the arm sculpt, live in the MuJoCo viewer.

    env -u PYTHONPATH .venv/bin/python scripts/view.py --demo press          # scripted sequences: press, roll, stamp
    env -u PYTHONPATH .venv/bin/python scripts/view.py --target x --plan oracle
    env -u PYTHONPATH .venv/bin/python scripts/view.py --target slab --plan structured --slow 2

``--demo`` replays a scripted sequence.  ``--target`` plans towards a target with
the chosen rollout model (the planning itself happens between the motions) and
replays every executed action.  ``--no-window`` runs without a display.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from clay_sculptor import config
from clay_sculptor.control.mpc import CEMCfg, ModelEval, OracleEval, run_episode
from clay_sculptor.envs.targets import DEMOS, make_target

SCRIPTED = {
    "press": [("press", [-0.030, -0.030, 0.0, 0.012, 0.0]), ("press", [0.0, 0.0, 0.4, 0.012, 0.0]), ("press", [0.030, 0.030, 0.0, 0.012, 0.0])],
    "roll": [("roller", [0.0, y, np.pi / 2 - 1e-6, 0.010, 0.09]) for y in (-0.030, 0.0, 0.030)],
    "stamp": [("stamp", [0.0, 0.0, np.pi / 4 - 1e-6, 0.012, 0.0])],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", choices=list(SCRIPTED), default=None)
    ap.add_argument("--target", choices=list(DEMOS), default=None)
    ap.add_argument("--plan", default="oracle", choices=["oracle", "prior", "gnn", "structured", "unet", "demo"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slow", type=float, default=1.0, help="slow-motion factor (1 = real time)")
    ap.add_argument("--skip", type=int, default=2, help="simulation steps per drawn frame")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()
    cfg = config.load()
    env = cfg.env(nsim=32)

    if args.target:
        tgt = make_target(env, args.target)
        if args.plan == "demo":
            seq = tgt.demo
        else:
            if args.plan == "oracle":
                ev = OracleEval(env)
            else:
                from clay_sculptor.learn import train as T

                ev = ModelEval(T.load(args.plan, args.seed, env.geom), env)
            print(f"planning towards '{args.target}' with {args.plan} ...", flush=True)
            rec = run_episode(env, tgt, ev, np.random.default_rng(args.seed), CEMCfg())
            seq = list(zip(rec["tools"], rec["actions"]))
            print(f"  {len(seq)} actions, height-map RMSE {1e3 * rec['costs'][0]:.2f} -> {1e3 * rec['costs'][-1]:.2f} mm", flush=True)
    else:
        seq = SCRIPTED[args.demo or "press"]

    # group consecutive actions of one tool: each group is executed without resetting the arm
    groups = []
    for tool, a in seq:
        if groups and groups[-1][0] == tool:
            groups[-1][1].append(a)
        else:
            groups.append((tool, [a]))

    h = cfg.sim.timestep
    state = env.state0
    viewer, shown, vdata = None, None, None
    try:
        for tool, actions in groups:
            m, d = env.models[tool], env._data[tool]
            if viewer is not None and shown != tool:          # tool change: a different compiled model
                viewer.close()
                time.sleep(0.5)
                viewer = None
            if viewer is None and not args.no_window:
                vdata = mujoco.MjData(m)        # the viewer draws this copy; it is only written under the viewer lock
                env.set_state(vdata, state, tool)
                mujoco.mj_forward(m, vdata)
                viewer = mujoco.viewer.launch_passive(m, vdata)
                viewer.cam.lookat[:] = [cfg.clay.centre[0], 0.0, 0.08]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.75, 140.0, -28.0
                shown = tool

            def on_step(m_, d_, t0=[time.perf_counter()], n=[0], vdata=vdata):
                if viewer is None:
                    return
                if not viewer.is_running():
                    raise KeyboardInterrupt
                n[0] += 1
                if n[0] % args.skip:
                    return
                with viewer.lock():
                    env.set_state(vdata, env.get_state(d_, tool), tool)
                    mujoco.mj_forward(m_, vdata)
                viewer.sync()
                time.sleep(max(0.0, args.skip * h * args.slow - (time.perf_counter() - t0[0])))
                t0[0] = time.perf_counter()

            print(f"{tool}: {len(actions)} action(s)", flush=True)
            state = env.run_continuous(state, tool, actions, on_step=on_step)
        if viewer is not None:
            t_end = time.time() + 3.0
            while viewer.is_running() and time.time() < t_end:
                viewer.sync()
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
            time.sleep(0.5)                      # let the render thread finish before the model is freed


if __name__ == "__main__":
    main()
