"""Stretch: a two-arm ping-pong rally (thin paddles, speculative contacts).

  1. feasibility : the paddle speed a return needs against what the arm can give, on an ITTF table
                   and on the midi table used here;
  2. bounce      : what a table bounce does in MuJoCo with this contact (frictionless: an exact
                   restitution law; with friction: Coulomb holds but the ball can gain energy);
  3. calibrate   : the paddle's effective restitution, from rallies played with the true ball state;
  4. rallies     : rally length and how rallies end, 3 seeds x ``--n`` rallies, for
                     true ball state, replanning every step                (what perfect sensing gives)
                     EKF, replanning every step                            (always think)
                     EKF, confidence-gated replanning, every step          (think when the filter is sure)
                     EKF, gated, 25 Hz and 12.5 Hz
                     EKF, gated, speculative contacts off;
  5. video       : one rally, Filament.

Outputs: results/rally.json, rally.png, rally.mp4.

    env -u PYTHONPATH .venv/bin/python scripts/rally.py [--n 48] [--set section.key=value ...]
"""
from __future__ import annotations

import argparse
import json
import time

import mujoco
import numpy as np
from mujoco import batch as mjbatch

from batter import RESULTS, config, rally
from batter.control.arm import Arm
from batter.estim.flight import FlightModel
from batter.eval import plotstyle as ps
from batter.eval import render
from batter.rally import poses, serve
from batter.rally.env import RallyEnv, run_rally
from batter.rally.flight import BounceModel, TableFlight
from batter.rally.planner import RallyPlanner
from batter.rally.run import evaluate, summarize, with_overrides
from batter.rally.scene import build_arm_model, build_rally_model

STAGE_NONE, STAGE_VEL = int(mujoco.mjtStage.mjSTAGE_NONE), int(mujoco.mjtStage.mjSTAGE_VEL)

CONDITIONS = {
    "true ball state, every step": {"oracle": True, "max_vz_sd": np.inf},
    "EKF, every step (always think)": {"max_vz_sd": np.inf},
    "EKF, gated, every step": {},
    "EKF, gated, 25 Hz": {"rate": 25.0},
    "EKF, gated, 12.5 Hz": {"rate": 12.5},
    "EKF, gated, speculative contacts off": {"sim": {"speculative": False}},
}


def world(cfg):
    m = build_rally_model(cfg)
    fm = FlightModel.identify(m)
    return m, fm, TableFlight(fm, BounceModel.identify(m, fm, cfg["table"]))


def feasibility(cfg: dict) -> list[dict]:
    """Paddle speed needed for the first return of 40 serves, planned with the true ball state."""
    rows = []
    for name, ov in [("ITTF table, 2.74 m", {"table": {"length": 2.74, "width": 1.525}}), ("midi table, 1.80 m", {})]:
        c = cfg
        if ov:
            c = with_overrides(cfg, **ov)
            c["table"]["L"] = c["table"]["length"] + 2 * c["table"]["gap"]
        m, fm, tf = world(c)
        arm = Arm(build_arm_model(c))
        P = poses.design(arm, c)
        pl = RallyPlanner(arm, tf, c, np.array(P["nominal"]), c["paddle"]["restitution_model"])
        S = serve.generate(tf, c, 40, 0)
        need, vin, vout = [], [], []
        for x0 in S:
            plan = pl.plan(tf.rollout(x0, 160)[15:], 15, np.array(P["ready"]), np.zeros(7), None, None)
            if plan is not None:
                need.append(plan.info["speed"])
                vin.append(np.linalg.norm(plan.info["v_b"]))
                vout.append(np.linalg.norm(plan.info["v_out"]))
        rows.append({"table": name, "paddle_speed_needed": [float(v) for v in np.percentile(need, [5, 50, 95])],
                     "paddle_speed_available": P["max_speed"], "ball_in": float(np.median(vin)), "ball_out": float(np.median(vout))})
        print(f"  {name}: paddle speed needed {rows[-1]['paddle_speed_needed'][1]:.2f} m/s (5-95 %: "
              f"{rows[-1]['paddle_speed_needed'][0]:.2f}-{rows[-1]['paddle_speed_needed'][2]:.2f}), available {P['max_speed']:.2f}; "
              f"ball arrives at {rows[-1]['ball_in']:.1f} m/s and must leave at {rows[-1]['ball_out']:.1f}", flush=True)
    return rows


def bounce_study(cfg: dict, n: int = 2000) -> dict:
    """Simulated single bounces, frictionless (as used) and with the table's friction."""
    out = {}
    for name, friction in [("frictionless", None), ("friction 0.25", 0.25)]:
        m, fm, tf = world(cfg)
        pid = m.pair("table_ball").id
        if friction is not None:
            m.pair_dim[pid] = 3
            m.pair_friction[pid, :2] = friction
        h = m.opt.timestep
        zt = cfg["table"]["height"] + fm.radius
        rng = np.random.default_rng(1)
        a, dd = int(m.jnt_qposadr[m.joint("ball").id]), int(m.jnt_dofadr[m.joint("ball").id])
        v = np.stack([rng.uniform(1.5, 6, n) * rng.choice([-1, 1], n), rng.uniform(-1, 1, n), -rng.uniform(1.0, 4.0, n)], 1)
        w = rng.normal(0, 60, (n, 3))
        z = zt + rng.uniform(0.05, 0.95, n) * h * np.abs(v[:, 2]) + 1e-4
        x_mid = cfg["table"]["gap"] + cfg["table"]["length"] / 4
        b = mjbatch.Batch(m, nsim=n, nthread=0, persistent=True)
        qpos, qvel = b.bind("qpos"), b.bind("qvel")
        b.reset(stage=STAGE_NONE)
        qpos[:, a:a + 3] = np.stack([np.full(n, x_mid), rng.uniform(-0.3, 0.3, n), z], 1)
        qpos[:, a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]
        qvel[:] = 0.0
        qvel[:, dd:dd + 3], qvel[:, dd + 3:dd + 6] = v, w
        b.forward(stage=STAGE_VEL)
        b.step(stage=STAGE_VEL)
        ratio = qvel[:, dd + 2] / -v[:, 2]
        out[name] = {"ratio_median": float(np.median(ratio)), "ratio_std": float(ratio.std()), "ratio_max": float(ratio.max()),
                     "gains_energy": float(np.mean(ratio > 1.0))}
        print(f"  bounce, {name}: v_z'/|v_z| median {np.median(ratio):.3f}, std {ratio.std():.3f}, max {ratio.max():.2f}; "
              f"faster after than before in {100 * np.mean(ratio > 1.0):.0f} % of bounces", flush=True)
    m, fm, tf = world(cfg)
    out["model"] = {"e": tf.bm.e, "b": tf.bm.b}
    return out


def calibrate(cfg: dict, n: int = 12) -> dict:
    """Effective paddle restitution from true-state rallies: (v' - s n).n = -e (v - s n).n."""
    env = RallyEnv(cfg, n, oracle=True, nthread=0)
    env.reset(serve.generate(env.tf, cfg, n, 3))
    d = mujoco.MjData(env.model)
    prev, rows, done, pre = env.n_hits.copy(), [], False, {}
    while not done:
        xs0 = env.ball_state().copy()
        for i in range(n):
            env.batch.get_data(i, d)
            pre[i] = (xs0[i].copy(), [d.site_xpos[env.ix.sweet[a]].copy() for a in (0, 1)])
        done = env.step()
        xs = env.ball_state()
        for i in np.flatnonzero(env.n_hits > prev):
            arm = env.hit_log[i][-1]["arm"]
            env.batch.get_data(i, d)
            nrm = d.site_xmat[env.ix.sweet[arm]].reshape(3, 3)[:, 0]
            s = (d.site_xpos[env.ix.sweet[arm]] - pre[i][1][arm]) @ nrm / env.h
            rows.append([(pre[i][0][3:6] - s * nrm) @ nrm, (xs[i, 3:6] - s * nrm) @ nrm])
        prev = env.n_hits.copy()
    R = np.array(rows)
    e = float(-np.sum(R[:, 0] * R[:, 1]) / np.sum(R[:, 0] ** 2))
    print(f"  effective paddle restitution {e:.3f} on {len(R)} hits (pair {cfg['paddle']['restitution']}; config "
          f"{cfg['paddle']['restitution_model']})", flush=True)
    return {"e": e, "n": int(len(R)), "pair": cfg["paddle"]["restitution"], "config": cfg["paddle"]["restitution_model"]}


def rallies(cfg: dict, tf, n: int, done: dict | None = None, save=None) -> dict:
    res = dict(done or {})
    for name, opts in CONDITIONS.items():
        if name in res:
            print(f"  {name:38s}: (from results/rally.json)", flush=True)
            continue
        per, raw = [], []
        t0 = time.time()
        for seed in cfg["seeds"]:
            S = serve.generate(tf, cfg, n, 100 + seed)
            out = evaluate(cfg, S, seed, opts)
            per.append(summarize(out, cfg["rally"]["max_hits"]))
            raw.append(out["n_hits"])
        hits = np.concatenate(raw)
        row = {"returns": [float(np.mean([p["returns_mean"] for p in per])), float(np.std([p["returns_mean"] for p in per]))],
               "full": [float(np.mean([p["full"] for p in per])), float(np.std([p["full"] for p in per]))],
               "median": float(np.median(hits)), "hist": np.bincount(hits, minlength=cfg["rally"]["max_hits"] + 1).tolist(),
               "plans_per_return": float(np.mean([p["plans_per_return"] for p in per])),
               "ms_per_call": float(np.mean([p["ms_per_call"] for p in per])),
               "reasons": {r: float(np.mean([p["reasons"].get(r, 0.0) for p in per])) for r in
                           sorted({r for p in per for r in p["reasons"]})}}
        res[name] = row
        if save is not None:
            save(res)
        print(f"  {name:38s}: {row['returns'][0]:5.2f} +- {row['returns'][1]:.2f} returns (median {row['median']:.0f}), "
              f"{100 * row['full'][0]:3.0f} % reach {cfg['rally']['max_hits']}; {row['plans_per_return']:.0f} planner calls per return; "
              f"ends: { {k: round(v, 2) for k, v in row['reasons'].items()} }  ({time.time() - t0:.0f}s)", flush=True)
    return res


def plot(res: dict, cfg: dict) -> None:
    import matplotlib.pyplot as plt

    ps.apply()
    names = list(CONDITIONS)
    colors = [ps.MUTED, ps.BLUE, ps.AQUA, ps.AQUA, ps.AQUA, ps.ORANGE]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [1.1, 1]})
    a = ax[0]
    y = np.array([res["rallies"][k]["returns"] for k in names])
    a.barh(range(len(names)), y[:, 0], xerr=y[:, 1], color=colors, capsize=3, alpha=0.9)
    for i, k in enumerate(names):
        a.text(y[i, 0] + 0.3, i, f"{y[i, 0]:.1f}", va="center", fontsize=8.5, color=ps.INK2)
    a.set_yticks(range(len(names)), names, fontsize=8.5)
    a.invert_yaxis()
    a.set(xlabel=f"returns per rally (mean of 3 seeds; capped at {cfg['rally']['max_hits']})", title="Rally length")
    a = ax[1]
    for k, col, ls in [(names[0], ps.MUTED, "--"), (names[1], ps.BLUE, "-"), (names[2], ps.AQUA, "-")]:
        h = np.array(res["rallies"][k]["hist"], float)
        surv = 1.0 - np.cumsum(h) / h.sum()
        a.step(np.arange(len(h)), np.r_[1.0, surv[:-1]] * 100, where="post", color=col, ls=ls, label=k)
    a.set(xlabel="returns", ylabel="rallies still going [%]", title="How long rallies last", ylim=(0, 103))
    a.legend(fontsize=8)
    fig.tight_layout()
    ps.save(fig, RESULTS / "rally.png")


def video(cfg: dict, n_rallies: int = 2) -> None:
    from batter.rally.video import RallyRecorder

    env = RallyEnv(cfg, 1, nthread=1, seed=1)
    cam = render.free_camera([1.65, 0.0, 0.85], 3.9, 118.0, -18.0)
    rec = RallyRecorder(env, cam, 1280, 720, text="Rally: two Pandas, 6 mm paddles, speculative contacts; EKF + analytic planner, confidence-gated", repeat=1)
    S = serve.generate(env.tf, cfg, 6, 7)
    for i in range(n_rallies):
        env.rng = np.random.default_rng(i)
        run_rally(env, S[i:i + 1], frame_cb=rec)
    render.write_mp4(RESULTS / "rally.mp4", rec.frames, fps=50)
    rec.close()
    print(f"  wrote {RESULTS / 'rally.mp4'} ({len(rec.frames)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="rallies per seed and condition")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--video-only", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore a partial results/rally.json")
    args = ap.parse_args()
    cfg = rally.load(config.load(), args.set)
    if args.video_only:
        video(cfg)
        return
    poses.load(Arm(build_arm_model(cfg)), cfg)          # design once, before the pool starts
    path = RESULTS / "rally.json"
    res: dict = {"n": args.n, "config": {k: cfg[k] for k in ("table", "paddle", "ball", "rally")}}
    old = json.loads(path.read_text()) if path.exists() and not args.fresh else {}
    if old.get("n") != args.n or old.get("config") != json.loads(json.dumps(res["config"])):
        old = {}                                       # resume only an identical run
    save = lambda: path.write_text(json.dumps(res, indent=2))
    for key, fn in [("feasibility", feasibility), ("bounce", bounce_study), ("calibration", calibrate)]:
        res[key] = old[key] if key in old else fn(cfg)
        save()
    _, _, tf = world(cfg)
    res["rallies"] = {}

    def save_rallies(rows):
        res["rallies"] = rows
        save()

    res["rallies"] = rallies(cfg, tf, args.n, old.get("rallies"), save_rallies)
    save()
    plot(res, cfg)
    if not args.no_video:
        video(cfg)


if __name__ == "__main__":
    main()
