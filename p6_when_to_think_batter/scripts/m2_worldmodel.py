"""M2: a learned world-model planner, against the analytic planner.

  1. collect  : batting data.  A noisy version of the analytic batter plays; every step after
                the last planner call of a pitch is a sample (camera fixes, arm state, target,
                the swing that was held) -> (hit?, where the ball landed);
  2. train    : latent world model (encoder, latent dynamics, outcome model, policy prior),
                3 seeds; and a data-size sweep with one seed;
  3. evaluate : world-model planner (MPPI in the latent space, 100 Hz) against the analytic
                planner on hit rate, target accuracy against pitch speed, and planning time;
                ablation: the policy prior alone, without MPPI;
  4. video.

Outputs: data/wm_data_*.npz, data/models/wm_*.pt, results/m2.json, m2_speed.png, m2_data.png, m2_worldmodel.mp4.

    env -u PYTHONPATH .venv/bin/python scripts/m2_worldmodel.py [--n-train 48000] [--steps 20000] [--n 128]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from batter import DATA, RESULTS, config
from batter.control.triggers import Periodic
from batter.envs.cage import BatEnv, EnvParams, run_episode
from batter.estim.flight import FlightModel
from batter.eval import plotstyle as ps
from batter.eval import render
from batter.eval.runner import evaluate, mean_std, summarize, time_planner, wm_collect_parallel, wm_train_parallel
from batter.learn import wm
from batter.learn.wm_planner import WorldModelPlanner
from batter.models.pitch import generate
from batter.models.scene import build_cage_model

SPEEDS = [9.0, 10.0, 11.0, 12.0, 13.0]


def model_path(tag: str):
    return DATA / "models" / f"wm_{tag}.pt"


def data_path(n: int, seed: int):
    return DATA / f"wm_data_n{n}_s{seed}.npz"


def dataset(cfg: dict, fm, n: int, seed: int) -> dict:
    path = data_path(n, seed)
    if path.exists():
        return dict(np.load(path))
    t0 = time.time()
    data = wm_collect_parallel(cfg, generate(fm, cfg, n, 20_000 + seed), seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    print(f"  collected {n} pitches -> {len(data['u'])} samples in {time.time() - t0:.0f} s "
          f"(hit rate of the noisy batter {100 * data['hit'].mean():.0f} %)", flush=True)
    return data


def eval_planner(cfg: dict, fm, env_kw: dict, n: int, speed, seeds, model_tags=None) -> dict:
    per = []
    for i, seed in enumerate(seeds):
        kw = dict(env_kw)
        if model_tags is not None:
            kw["planner"] = str(model_path(model_tags[i]))
        P = generate(fm, cfg, n, (400 if speed is not None else 500) + seed, speed=speed)
        per.append(summarize(evaluate(cfg, ("periodic", 1000.0), P, seed, env_kw=kw)))
    return mean_std(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=48000, help="pitches of batting data")
    ap.add_argument("--steps", type=int, default=20000, help="gradient steps per model")
    ap.add_argument("--n", type=int, default=128, help="evaluation pitches per seed and setting")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--renderer", default="filament", choices=["filament", "classic"], help="renderer for the video")
    ap.add_argument("--video-only", action="store_true", help="only render the video")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = config.load(overrides=args.set)
    if args.video_only:
        video(cfg, renderer=args.renderer)
        return
    RESULTS.mkdir(exist_ok=True)
    (DATA / "models").mkdir(parents=True, exist_ok=True)
    fm = FlightModel.identify(build_cage_model(cfg))
    seeds = cfg["seeds"]
    sigma = cfg["target"]["sigma"]
    res: dict = {"n": args.n, "n_train_pitches": args.n_train, "steps": args.steps}

    # 1. data
    train = dataset(cfg, fm, args.n_train, 0)
    val = dataset(cfg, fm, 2000, 1)
    res["n_samples"] = int(len(train["u"]))
    res["data_hit_rate"] = float(train["hit"].mean())

    # 2. models: 3 seeds on all the data, 1 seed on fractions of it
    fracs = [1 / 16, 1 / 4]
    jobs = []
    for s in seeds:
        if args.retrain or not model_path(f"s{s}").exists():
            jobs.append((str(data_path(args.n_train, 0)), 1.0, wm.TrainConfig(steps=args.steps, seed=s), sigma, model_path(f"s{s}")))
    for f in fracs:
        tag = f"frac{int(1 / f)}"
        if args.retrain or not model_path(tag).exists():
            jobs.append((str(data_path(args.n_train, 0)), f, wm.TrainConfig(steps=args.steps, seed=0), sigma, model_path(tag)))
    if jobs:
        t0 = time.time()
        wm_train_parallel(jobs)
        print(f"  trained {len(jobs)} world models in {time.time() - t0:.0f} s", flush=True)
    res["validation"] = {}
    for tag in [f"s{s}" for s in seeds] + [f"frac{int(1 / f)}" for f in fracs]:
        ck = torch.load(model_path(tag), map_location="cpu")
        model = wm.WorldModel(sigma_target=sigma)
        model.load_state_dict(ck["state"])
        res["validation"][tag] = wm.validate(model.eval(), val)
        v = res["validation"][tag]
        print(f"  model {tag:7s}: hit/miss accuracy {100 * v['hit_accuracy']:.1f} %   landing error {v['landing_err_med_m']:.2f} m   "
              f"quality MAE {v['quality_mae']:.3f}", flush=True)

    # 3. closed loop against pitch speed
    variants = {
        "analytic planner (classical)": ({}, None),
        "world-model planner (pure learning)": ({}, [f"s{s}" for s in seeds]),
        "world-model prior only, no MPPI": ({"planner_kw": {"iters": 0}}, [f"s{s}" for s in seeds]),
    }
    res["speed"], res["all"] = {}, {}
    for name, (kw, tags) in variants.items():
        rows = []
        for v in SPEEDS:
            row = {"speed": v, **eval_planner(cfg, fm, kw, args.n, v, seeds, tags)}
            rows.append(row)
            print(f"  {name:38s} {v:5.1f} m/s: hit {100 * row['hit'][0]:5.1f} +- {100 * row['hit'][1]:.1f} %   on target "
                  f"{100 * row['on_target'][0]:5.1f} +- {100 * row['on_target'][1]:.1f} %   median miss {row['dist_med'][0]:.2f} m", flush=True)
        res["speed"][name] = rows
        res["all"][name] = eval_planner(cfg, fm, kw, args.n, None, seeds, tags)
        r = res["all"][name]
        print(f"  {name:38s} all speeds: hit {100 * r['hit'][0]:5.1f} %   on target {100 * r['on_target'][0]:5.1f} %   "
              f"quality {r['quality'][0]:.3f} +- {r['quality'][1]:.3f}", flush=True)
        (RESULTS / "m2.json").write_text(json.dumps(res, indent=2))
    res["data_sweep"] = []
    for tag, frac in [("frac16", 1 / 16), ("frac4", 1 / 4), ("s0", 1.0)]:
        r = eval_planner(cfg, fm, {}, args.n, None, seeds, [tag] * len(seeds))
        res["data_sweep"].append({"pitches": int(frac * args.n_train), **r})
        print(f"  data sweep {int(frac * args.n_train):6d} pitches: hit {100 * r['hit'][0]:5.1f} %   on target "
              f"{100 * r['on_target'][0]:5.1f} %   quality {r['quality'][0]:.3f}", flush=True)

    # 4. planning time, measured alone on one core
    Pt = generate(fm, cfg, 8, 4242)
    res["timing"] = {"analytic": time_planner(cfg, ("periodic", 1000.0), Pt),
                     "world model": time_planner(cfg, ("periodic", 1000.0), Pt, {"planner": str(model_path("s0"))}),
                     "world model prior only": time_planner(cfg, ("periodic", 1000.0), Pt, {"planner": str(model_path("s0")),
                                                                                           "planner_kw": {"iters": 0}})}
    for k, t in res["timing"].items():
        print(f"  planning time, {k:24s}: median {t['ms_median']:.1f} ms   mean {t['ms_mean']:.1f}   p90 {t['ms_p90']:.1f}   "
              f"max {t['ms_max']:.1f}  ({t['calls']} calls)", flush=True)
    (RESULTS / "m2.json").write_text(json.dumps(res, indent=2))
    plots(res)
    if not args.no_video:
        video(cfg, renderer=args.renderer)


def plots(res: dict) -> None:
    import matplotlib.pyplot as plt

    ps.apply()
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    style = {"analytic planner (classical)": (ps.BLUE, "o", "-"), "world-model planner (pure learning)": (ps.ORANGE, "s", "-"),
             "world-model prior only, no MPPI": (ps.ORANGE, "x", ":")}
    for name, (col, mk, ls) in style.items():
        rows = res["speed"][name]
        v = [r["speed"] for r in rows]
        for a, key, sc in [(ax[0], "hit", 100), (ax[1], "on_target", 100), (ax[2], "dist_med", 1)]:
            y = np.array([r[key] for r in rows])
            a.errorbar(v, sc * y[:, 0], sc * y[:, 1], color=col, marker=mk, ls=ls, label=name, capsize=2)
    ax[0].set(xlabel="pitch speed at release [m/s]", ylabel="[%]", title="Hit rate", ylim=(-3, 103))
    ax[1].set(xlabel="pitch speed at release [m/s]", ylabel="[%]", title="Ball lands in the target zone", ylim=(-3, 103))
    ax[2].set(xlabel="pitch speed at release [m/s]", ylabel="[m]", title="Median distance to the target (hits)")
    ax[0].legend(loc="lower left", fontsize=7.5)
    t = res["timing"]
    names = ["analytic", "world model", "world model prior only"]
    ax[3].bar(range(3), [t[k]["ms_median"] for k in names], color=[ps.BLUE, ps.ORANGE, ps.ORANGE],
              yerr=[[0, 0, 0], [t[k]["ms_p90"] - t[k]["ms_median"] for k in names]], capsize=3)
    ax[3].axhline(10.0, color=ps.INK2, lw=0.8, ls="--")
    ax[3].text(2.45, 10.3, "control period", ha="right", fontsize=8, color=ps.INK2)
    ax[3].set_xticks(range(3), ["analytic", "world model\n(MPPI)", "prior only"])
    ax[3].set(ylabel="ms per planner call (median, bar to p90)", title="Planning time, one core")
    fig.tight_layout()
    ps.save(fig, RESULTS / "m2_speed.png")

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    d = res["data_sweep"]
    x = [r["pitches"] for r in d]
    for key, lab, mk in [("hit", "hit rate", "o"), ("on_target", "on target", "s")]:
        y = np.array([r[key] for r in d])
        ax.errorbar(x, 100 * y[:, 0], 100 * y[:, 1], color=ps.ORANGE, marker=mk, ls="-" if key == "hit" else "--", label=f"world model: {lab}", capsize=2)
    a = res["all"]["analytic planner (classical)"]
    ax.axhline(100 * a["on_target"][0], color=ps.BLUE, ls="--", label="analytic: on target")
    ax.set(xscale="log", xlabel="pitches of batting data", ylabel="[%]", title="World-model planner against data", ylim=(0, 103))
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    ps.save(fig, RESULTS / "m2_data.png")


def video(cfg: dict, n_pitches: int = 4, renderer: str = "filament") -> None:
    cam = render.free_camera([2.9, 0.0, 0.8], 7.2, 100.0, -14.0)
    env = BatEnv(cfg, 1, EnvParams(post_steps=70, max_time=1.6), seed=0)
    env.planner = WorldModelPlanner.load(str(model_path("s0")), env)
    rec = render.make_recorder(renderer, env, cam, 960, 540, text="M2: learned world-model planner (MPPI in latent space), 100 Hz", repeat=2,
                          show_forecast=False)
    P = generate(env.fm, cfg, n_pitches, seed=11)
    for i in range(n_pitches):
        run_episode(env, P.take(np.array([i])), Periodic(1000.0), frame_cb=rec)
    render.write_mp4(RESULTS / "m2_worldmodel.mp4", rec.frames, fps=30)
    rec.close()


if __name__ == "__main__":
    main()
