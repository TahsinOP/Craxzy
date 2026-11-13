"""M0: the batting cage.  Pitch distribution, and tunnelling with speculative contacts off / on.

  1. pitch distribution: release speed, height, spin; where the ball crosses the plate;
     how much the (hidden) spin bends the flight;
  2. static bat: pitches aimed at the middle of a bat that is held still.  Does the ball
     bounce off or pass through?  speculative flag x timestep (8, 10, 16 ms);
  3. swinging bat: the M1 batter (100 Hz replanning) with the flag off / on;
  4. video: the same pitch with the flag off and on, side by side, slow motion.

Outputs: results/m0.json, m0_pitches.png, m0_tunnel.png, m0_tunnel.mp4.

    env -u PYTHONPATH .venv/bin/python scripts/m0_cage.py [--n 200] [--set key=value ...]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from batter import RESULTS, config
from batter.control.arm import NOMINAL_STRIKE, Arm
from batter.control.planner import crossing
from batter.control.triggers import Periodic
from batter.envs.cage import BatEnv, EnvParams, run_episode
from batter.estim.flight import FlightModel
from batter.eval import plotstyle as ps
from batter.eval import render
from batter.eval.runner import evaluate, mean_std
from batter.models.pitch import generate
from batter.models.scene import build_cage_model

TIMESTEPS = [0.008, 0.010, 0.016]


def pitch_distribution(cfg: dict, fm: FlightModel, n: int) -> dict:
    import matplotlib.pyplot as plt

    P = generate(fm, cfg, n, seed=0)
    steps = int(P.cross_t.max() / fm.h) + 6
    traj = fm.rollout(P.x0, steps)
    x0 = P.x0.copy()
    x0[:, 6:9] = 0.0
    traj0 = fm.rollout(x0, steps)                                      # the same release without spin
    sx = cfg["pitch"]["strike_x"]
    cr = [crossing(traj[:, i], sx) for i in range(n)]
    cr0 = [crossing(traj0[:, i], sx) for i in range(n)]
    cross = np.array([c[1] for c in cr])
    v_arr = np.array([c[2] for c in cr])
    dz = np.array([c[1][2] - c0[1][2] for c, c0 in zip(cr, cr0)])      # lift of the spin at the plate
    dy = np.array([c[1][1] - c0[1][1] for c, c0 in zip(cr, cr0)])
    spin_y = P.x0[:, 7]

    ps.apply()
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    a = ax[0, 0]
    for i in range(0, n, max(1, n // 70)):
        k = int(np.ceil(cr[i][0])) + 1
        a.plot(traj[:k, i, 0], traj[:k, i, 2], color=ps.ORANGE if spin_y[i] > 0 else ps.BLUE, lw=0.8, alpha=0.6)
    a.plot([], [], color=ps.ORANGE, label="backspin (floats)")
    a.plot([], [], color=ps.BLUE, label="topspin (dives)")
    a.axvline(sx, color=ps.INK2, lw=0.8, ls=":")
    a.set(xlabel="x [m]", ylabel="height [m]", title="Pitches, side view")
    a.legend()
    a = ax[0, 1]
    a.scatter(cross[:, 1], cross[:, 2], s=6, c=ps.BLUE, alpha=0.5)
    a.set(xlabel="y at the plate [m]", ylabel="z at the plate [m]", title="Where pitches cross the hitting plane")
    a.set_aspect("equal")
    a = ax[0, 2]
    a.scatter(spin_y, 100 * dz, s=6, c=ps.BLUE, alpha=0.5)
    a.axhspan(-0.6, 0.6, color=ps.ORANGE, alpha=0.35, label="bat thickness (1.2 cm)")
    a.set(xlabel="spin about y [rad/s]", ylabel="height change at the plate [cm]",
          title="What the spin does: same release, with and without spin")
    a.legend()
    for a, val, lab in [(ax[1, 0], P.speed, "release speed [m/s]"), (ax[1, 1], P.x0[:, 2], "release height [m]"),
                        (ax[1, 2], P.cross_t, "time of flight to the plate [s]")]:
        a.hist(val, bins=30, color=ps.BLUE, alpha=0.85)
        a.set(xlabel=lab, ylabel="pitches")
    fig.tight_layout()
    ps.save(fig, RESULTS / "m0_pitches.png")
    return {"n": n, "speed": [float(P.speed.min()), float(P.speed.max())],
            "release_z": [float(P.x0[:, 2].min()), float(P.x0[:, 2].max())],
            "spin": [float(np.linalg.norm(P.x0[:, 6:9], axis=1).min()), float(np.linalg.norm(P.x0[:, 6:9], axis=1).max())],
            "flight_time": [float(P.cross_t.min()), float(P.cross_t.max())],
            "arrival_speed": [float(np.linalg.norm(v_arr, axis=1).min()), float(np.linalg.norm(v_arr, axis=1).max())],
            "descent_deg": [float(np.rad2deg(np.arctan2(-v_arr[:, 2], -v_arr[:, 0])).min()),
                            float(np.rad2deg(np.arctan2(-v_arr[:, 2], -v_arr[:, 0])).max())],
            "cross_y": [float(cross[:, 1].min()), float(cross[:, 1].max())],
            "cross_z": [float(cross[:, 2].min()), float(cross[:, 2].max())],
            "spin_dz_cm": [float(100 * dz.min()), float(100 * dz.max()), float(100 * dz.std())],
            "spin_dy_cm": [float(100 * dy.min()), float(100 * dy.max()), float(100 * dy.std())]}


def static_bat(cfg: dict, n: int) -> list[dict]:
    """Pitches at a bat held still: fraction bounced / passed through, per flag and timestep."""
    arm = Arm(build_cage_model(cfg))
    sweet, R = arm.fk(NOMINAL_STRIKE)
    rows = []
    for h in TIMESTEPS:
        for spec in (False, True):
            per = []
            for seed in cfg["seeds"]:
                fm = FlightModel.identify(build_cage_model(cfg, {"timestep": h}))
                rng = np.random.default_rng(100 + seed)
                # aim at the blade: +-3 cm across it, +-12 cm along it; the ball centre stops one radius short
                off = rng.uniform(-0.03, 0.03, (n, 1)) * R[:, 1] + rng.uniform(-0.12, 0.12, (n, 1)) * R[:, 2]
                aim = sweet + off + (cfg["ball"]["radius"] + cfg["bat"]["thickness"] / 2) * R[:, 0]
                P = generate(fm, cfg, n, seed, aim=aim)
                out = evaluate(cfg, ("periodic", 0.0), P, seed,
                               env_kw={"sim": {"timestep": h, "speculative": spec},
                                       "env": {"swing": False, "q_start": tuple(NOMINAL_STRIKE)}})
                back = out["hit"] & (out["v_post"][:, 0] > 0.5)
                per.append({"bounced": float(back.mean()), "touched": float(out["hit"].mean()),
                            "passed": float((~back).mean())})
            row = {"timestep": h, "speculative": spec, **mean_std(per)}
            rows.append(row)
            print(f"  static bat  h {h * 1e3:4.0f} ms  speculative {'on ' if spec else 'off'}: bounced "
                  f"{100 * row['bounced'][0]:5.1f} +- {100 * row['bounced'][1]:.1f} %   passed through "
                  f"{100 * row['passed'][0]:5.1f} %   (contact registered {100 * row['touched'][0]:.1f} %)", flush=True)
    return rows


def swinging_bat(cfg: dict, n: int) -> list[dict]:
    rows = []
    for h in TIMESTEPS:
        for spec in (False, True):
            per = []
            for seed in cfg["seeds"]:
                fm = FlightModel.identify(build_cage_model(cfg, {"timestep": h}))
                P = generate(fm, cfg, n, 200 + seed)
                out = evaluate(cfg, ("periodic", 1000.0), P, seed, env_kw={"sim": {"timestep": h, "speculative": spec}})
                back = out["hit"] & (out["v_post"][:, 0] > 0.5)
                per.append({"hit": float(back.mean()), "touched": float(out["hit"].mean()),
                            "on_target": float(out["on_target"].mean()), "quality": float(out["quality"].mean())})
            row = {"timestep": h, "speculative": spec, **mean_std(per)}
            rows.append(row)
            print(f"  swinging bat h {h * 1e3:4.0f} ms  speculative {'on ' if spec else 'off'}: ball returned "
                  f"{100 * row['hit'][0]:5.1f} +- {100 * row['hit'][1]:.1f} %   on target {100 * row['on_target'][0]:5.1f} %",
                  flush=True)
    return rows


def tunnel_plot(static: list[dict], swing: list[dict]) -> None:
    import matplotlib.pyplot as plt

    ps.apply()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for a, rows, key, title in [(ax[0], static, "bounced", "Bat held still: ball bounces off"),
                                (ax[1], swing, "hit", "Swinging bat (100 Hz batter): ball returned")]:
        x = np.arange(len(TIMESTEPS))
        for spec, col, lab, dx in [(False, ps.ORANGE, "speculative contacts off", -0.19), (True, ps.BLUE, "speculative contacts on", 0.19)]:
            r = [q for q in rows if q["speculative"] == spec]
            y = [100 * q[key][0] for q in r]
            e = [100 * q[key][1] for q in r]
            a.bar(x + dx, y, 0.36, yerr=e, color=col, label=lab, capsize=3)
            for xi, yi in zip(x + dx, y):
                a.text(xi, yi + 2, f"{yi:.0f}", ha="center", fontsize=8, color=ps.INK2)
        a.set_xticks(x, [f"{1e3 * h:.0f} ms" for h in TIMESTEPS])
        a.set(xlabel="timestep", title=title, ylim=(0, 112))
    ax[0].set_ylabel("pitches [%]")
    ax[1].legend(loc="center right")
    fig.tight_layout()
    ps.save(fig, RESULTS / "m0_tunnel.png")


def video(cfg: dict, n_pitches: int = 4, renderer: str = "filament") -> None:
    """The same pitches with the flag off (left) and on (right), 10 ms steps, slow motion."""
    cam = render.free_camera([1.0, 0.0, 0.95], 2.6, 105.0, -8.0)
    frames = {}
    for spec in (False, True):
        env = BatEnv(cfg, 1, EnvParams(post_steps=40, max_time=1.3), seed=0, sim_override={"speculative": spec})
        rec = render.make_recorder(renderer, env, cam, 640, 480, text=f"speculative contacts {'ON' if spec else 'OFF'}", repeat=3)
        P = generate(env.fm, cfg, n_pitches, seed=3)
        for i in range(n_pitches):
            run_episode(env, P.take(np.array([i])), Periodic(1000.0), frame_cb=rec)
        frames[spec] = rec.pitches
        rec.close()
    out = render.side_by_side([frames[False], frames[True]])
    render.write_mp4(RESULTS / "m0_tunnel.mp4", out, fps=30)
    print(f"  wrote {RESULTS / 'm0_tunnel.mp4'} ({len(out)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="pitches per seed and setting")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--renderer", default="filament", choices=["filament", "classic"], help="renderer for the video")
    ap.add_argument("--video-only", action="store_true", help="only render the video")
    args = ap.parse_args()
    cfg = config.load(overrides=args.set)
    if args.video_only:
        video(cfg, renderer=args.renderer)
        return
    RESULTS.mkdir(exist_ok=True)
    fm = FlightModel.identify(build_cage_model(cfg))
    res = {"n": args.n, "pitches": pitch_distribution(cfg, fm, 1000)}
    print("pitch distribution:", json.dumps(res["pitches"]))
    res["static"] = static_bat(cfg, args.n)
    res["swing"] = swinging_bat(cfg, args.n)
    tunnel_plot(res["static"], res["swing"])
    (RESULTS / "m0.json").write_text(json.dumps(res, indent=2))
    if not args.no_video:
        video(cfg, renderer=args.renderer)


if __name__ == "__main__":
    main()
