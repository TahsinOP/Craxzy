"""M1: the model-based hitter.  EKF -> crossing forecast -> swing QP, replanned at 100 Hz.

  1. estimator : crossing-point error and its predicted standard deviation against the
                 time left to the plate (no arm involved);
  2. calibrate : effective restitution of the impact model, fitted on swings made with the
                 true ball state (so the fit sees the impact, not the estimator);
  3. hitter    : hit rate and target accuracy against pitch speed, 3 seeds, for
                   analytic planner at 100 Hz        (the M1 controller)
                   the same with the true ball state (what a perfect estimator would give)
                   the same planner called once      (what replanning buys);
  4. compute   : wall-clock per planner call;
  5. video.

Outputs: results/m1.json, m1_estimator.png, m1_speed.png, m1_hitter.mp4.

    env -u PYTHONPATH .venv/bin/python scripts/m1_hitter.py [--n 200]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from batter import RESULTS, config
from batter.control.triggers import Periodic
from batter.envs.cage import BatEnv, EnvParams, run_episode
from batter.estim.ekf import BallEKF, crossing_batch
from batter.estim.flight import FlightModel
from batter.eval import plotstyle as ps
from batter.eval import render
from batter.eval.runner import evaluate, mean_std, summarize, time_planner
from batter.models.pitch import generate
from batter.models.scene import build_cage_model

SPEEDS = [9.0, 10.0, 11.0, 12.0, 13.0]
TAU_BINS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65]


def estimator(cfg: dict, fm: FlightModel, n: int) -> dict:
    """Crossing-point error of the EKF forecast against the time to the plate."""
    rows = []
    for seed in cfg["seeds"]:
        P = generate(fm, cfg, n, 300 + seed)
        ekf = BallEKF(fm, n, cfg["measurement"])
        rng = np.random.default_rng(seed)
        x = P.x0.copy()
        truth = [x.copy()]
        per = []
        for k in range(int(P.cross_t.max() / fm.h) + 1):
            x = fm.step(x)
            truth.append(x.copy())
            j = k + 1 - ekf.nd
            if j < 0:
                continue
            z = truth[j][:, :3] + rng.normal(size=(n, 3)) * ekf.meas_sigma(truth[j][:, :3])[:, None]
            ekf.step(z, rng.random(n) >= cfg["measurement"]["dropout"])
            traj, Ppp = ekf.forecast(95)
            c = crossing_batch(traj, Ppp, cfg["pitch"]["strike_x"])
            ct = crossing_batch(fm.rollout(x, 95), np.zeros((96, n, 3, 3)), cfg["pitch"]["strike_x"])
            e = (c["p"] - ct["p"])[:, 1:]
            m2 = np.einsum("bi,bij,bj->b", e, np.linalg.inv(c["cov"] + 1e-12 * np.eye(2)), e)
            tau = (c["steps"] - ekf.nd) * fm.h
            ok = c["found"] & ct["found"] & (tau > 0) & (ekf.n_fix >= 5)
            per.append(np.stack([tau, np.linalg.norm(e, axis=1), np.sqrt(np.trace(c["cov"], axis1=1, axis2=2)), m2,
                                 np.abs(c["steps"] - ekf.nd - ct["steps"]) * fm.h], axis=1)[ok])
        rows.append(np.concatenate(per))
    out = {"tau": [], "err_med": [], "err_p90": [], "sd_pred": [], "cover90": [], "time_err_ms": []}
    for lo, hi in zip(TAU_BINS[:-1], TAU_BINS[1:]):
        sel = [r[(r[:, 0] >= lo) & (r[:, 0] < hi)] for r in rows]
        out["tau"].append(0.5 * (lo + hi))
        for key, f in [("err_med", lambda s: np.median(s[:, 1])), ("err_p90", lambda s: np.percentile(s[:, 1], 90)),
                       ("sd_pred", lambda s: np.median(s[:, 2])), ("cover90", lambda s: np.mean(s[:, 3] < 4.605)),
                       ("time_err_ms", lambda s: 1e3 * np.median(s[:, 4]))]:
            v = [f(s) for s in sel]
            out[key].append([float(np.mean(v)), float(np.std(v))])
    return out


E_GRID = [0.54, 0.56, 0.58, 0.60, 0.62]


def calibrate(cfg: dict, fm: FlightModel, n: int) -> dict:
    """Effective restitution of the planner's impact model, on swings made with the true ball state.

    Two estimates.  (a) From the velocities: (v_post - s n) . n = -e (v_pre - s n) . n with the
    commanded bat speed s and n the direction of the velocity change.  (b) From what the planner
    is judged on: the value of e for which balls land neither long nor short of the target on
    average (the mean radial landing error, linear in e, crosses zero).  (b) is the one used; it
    also absorbs the small difference between the commanded and the realised bat speed.
    """
    import copy

    P = generate(fm, cfg, n, seed=999)
    kw = {"env": {"oracle": True}}
    radial = []
    for e in E_GRID:
        c = copy.deepcopy(cfg)
        c["planner"]["restitution"] = e
        out = evaluate(c, ("periodic", 1000.0), P, 0, env_kw=kw)
        u = P.target / np.linalg.norm(P.target, axis=1, keepdims=True)
        ok = out["hit"] & np.isfinite(out["landing"][:, 0])
        radial.append(float(np.mean(np.einsum("bi,bi->b", (out["landing"] - P.target)[ok], u[ok]))))
        print(f"  calibrate: e = {e:.3f}: mean radial landing error {radial[-1]:+.3f} m", flush=True)
    slope, icpt = np.polyfit(E_GRID, radial, 1)
    e_land = float(-icpt / slope)

    out = evaluate(cfg, ("periodic", 1000.0), P, 0, env_kw=kw)
    hit = out["hit"] & np.isfinite(out["v_post"][:, 0]) & np.isfinite(out["bat_speed"])
    v_post = out["v_post"][hit] - fm.h * fm.g           # undo the one step of gravity since the impact
    dv = v_post - out["v_pre"][hit]
    nrm = dv / np.linalg.norm(dv, axis=1, keepdims=True)
    s = out["bat_speed"][hit]
    closing = s - np.einsum("bi,bi->b", out["v_pre"][hit], nrm)          # > 0
    leaving = np.einsum("bi,bi->b", v_post, nrm) - s
    e_vel = float(np.sum(leaving * closing) / np.sum(closing**2))
    return {"n": int(hit.sum()), "restitution_landing": e_land, "restitution_velocity": e_vel,
            "restitution_config": cfg["planner"]["restitution"], "e_grid": E_GRID, "radial_error_m": radial,
            "residual_std_mps": float((leaving - e_vel * closing).std()), "pair_restitution": cfg["bat"]["restitution"],
            "oracle": summarize(out)}


def speed_curves(cfg: dict, fm: FlightModel, n: int) -> dict:
    variants = {"analytic 100 Hz": (("periodic", 1000.0), {}),
                "analytic 100 Hz, true ball state": (("periodic", 1000.0), {"env": {"oracle": True}}),
                "analytic, planned once": (("periodic", 0.0), {})}
    res = {}
    for name, (spec, kw) in variants.items():
        rows = []
        for v in SPEEDS:
            per = [summarize(evaluate(cfg, spec, generate(fm, cfg, n, 400 + seed, speed=v), seed, env_kw=kw))
                   for seed in cfg["seeds"]]
            row = {"speed": v, **mean_std(per)}
            rows.append(row)
            print(f"  {name:34s} {v:5.1f} m/s: hit {100 * row['hit'][0]:5.1f} +- {100 * row['hit'][1]:.1f} %   on target "
                  f"{100 * row['on_target'][0]:5.1f} +- {100 * row['on_target'][1]:.1f} %   median miss {row['dist_med'][0]:.2f} m   "
                  f"calls {row['plans'][0]:.1f}   {row['ms_per_call'][0]:.1f} ms/call", flush=True)
        res[name] = rows
    # the whole pitch distribution
    for name, (spec, kw) in variants.items():
        per = [summarize(evaluate(cfg, spec, generate(fm, cfg, n, 500 + seed), seed, env_kw=kw)) for seed in cfg["seeds"]]
        res[name + " | all"] = mean_std(per)
        r = res[name + " | all"]
        print(f"  {name:34s} all speeds: hit {100 * r['hit'][0]:5.1f} +- {100 * r['hit'][1]:.1f} %   on target "
              f"{100 * r['on_target'][0]:5.1f} +- {100 * r['on_target'][1]:.1f} %   quality {r['quality'][0]:.3f}", flush=True)
    return res


def plots(res: dict) -> None:
    import matplotlib.pyplot as plt

    ps.apply()
    e = res["estimator"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    tau = np.array(e["tau"])
    med, sd = np.array(e["err_med"]), np.array(e["sd_pred"])
    ax[0].errorbar(tau * 1e3, 100 * med[:, 0], 100 * med[:, 1], color=ps.BLUE, marker="o", label="crossing-point error (median)")
    ax[0].plot(tau * 1e3, 100 * np.array(e["err_p90"])[:, 0], color=ps.BLUE, ls=":", label="90th percentile")
    ax[0].plot(tau * 1e3, 100 * sd[:, 0], color=ps.ORANGE, marker="s", label="the filter's own standard deviation")
    ax[0].axhline(4.5, color=ps.INK2, lw=0.8, ls="--")
    ax[0].text(640, 5.2, "half the bat's width", ha="right", fontsize=8, color=ps.INK2)
    ax[0].set(xlabel="time left to the plate [ms]", ylabel="[cm]", yscale="log",
              title="EKF forecast of where the ball crosses the plate")
    ax[0].legend()
    cov = np.array(e["cover90"])
    ax[1].errorbar(tau * 1e3, 100 * cov[:, 0], 100 * cov[:, 1], color=ps.BLUE, marker="o")
    ax[1].axhline(90, color=ps.INK2, lw=0.8, ls="--")
    ax[1].set(xlabel="time left to the plate [ms]", ylabel="truth inside the 90 % ellipse [%]", ylim=(60, 100),
              title="Calibration of the crossing-point covariance")
    fig.tight_layout()
    ps.save(fig, RESULTS / "m1_estimator.png")

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    style = {"analytic 100 Hz": (ps.BLUE, "o", "-"), "analytic 100 Hz, true ball state": (ps.MUTED, "s", "--"),
             "analytic, planned once": (ps.VIOLET, "^", ":")}
    for name, (col, mk, ls) in style.items():
        rows = res["speed"][name]
        v = [r["speed"] for r in rows]
        for a, key, scale in [(ax[0], "hit", 100), (ax[1], "on_target", 100), (ax[2], "dist_med", 1)]:
            y = np.array([r[key] for r in rows])
            a.errorbar(v, scale * y[:, 0], scale * y[:, 1], color=col, marker=mk, ls=ls, label=name, capsize=2)
    ax[0].set(xlabel="pitch speed at release [m/s]", ylabel="[%]", title="Hit rate", ylim=(-3, 103))
    ax[1].set(xlabel="pitch speed at release [m/s]", ylabel="[%]", title="Ball lands in the target zone (0.75 m)", ylim=(-3, 103))
    ax[2].set(xlabel="pitch speed at release [m/s]", ylabel="[m]", title="Median distance to the target centre (hits)")
    ax[0].legend(loc="center left")
    fig.tight_layout()
    ps.save(fig, RESULTS / "m1_speed.png")


def video(cfg: dict, n_pitches: int = 4, renderer: str = "filament") -> None:
    cam = render.free_camera([2.9, 0.0, 0.8], 7.2, 100.0, -14.0)
    env = BatEnv(cfg, 1, EnvParams(post_steps=70, max_time=1.6), seed=0)
    rec = render.make_recorder(renderer, env, cam, 960, 540, text="M1: EKF + analytic planner, replanned at 100 Hz", repeat=2)
    P = generate(env.fm, cfg, n_pitches, seed=11)
    for i in range(n_pitches):
        run_episode(env, P.take(np.array([i])), Periodic(1000.0), frame_cb=rec)
    render.write_mp4(RESULTS / "m1_hitter.mp4", rec.frames, fps=30)
    rec.close()
    print(f"  wrote {RESULTS / 'm1_hitter.mp4'}")


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
    res = {"estimator": estimator(cfg, fm, 300)}
    res["calibration"] = calibrate(cfg, fm, args.n)
    c = res["calibration"]
    print(f"  restitution: {c['restitution_landing']:.3f} from landings, {c['restitution_velocity']:.3f} from velocities, on "
          f"{c['n']} oracle swings (config {c['restitution_config']}, contact pair {c['pair_restitution']})")
    res["speed"] = speed_curves(cfg, fm, args.n)
    res["n"] = args.n
    res["timing"] = time_planner(cfg, ("periodic", 1000.0), generate(fm, cfg, 8, 4242))     # alone, on one core
    t = res["timing"]
    print(f"  planner call, measured alone on one core: median {t['ms_median']:.1f} ms, mean {t['ms_mean']:.1f}, p90 {t['ms_p90']:.1f}, "
          f"max {t['ms_max']:.1f} ({t['calls']} calls)")
    (RESULTS / "m1.json").write_text(json.dumps(res, indent=2))
    plots(res)
    if not args.no_video:
        video(cfg, renderer=args.renderer)


if __name__ == "__main__":
    main()
