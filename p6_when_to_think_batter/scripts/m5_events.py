"""Stretch: trigger on events from an emulated event camera instead of on state innovation.

  1. an event camera (eval/eventcam.py) watches the flight corridor; log-brightness changes
     above a contrast threshold become events;
  2. event trigger: the centroid of the ON events against the image position the cached
     plan's forecast predicts; chi-square gate on the pixel innovation;
  3. compared with the M3 innovation trigger (delayed camera fixes) and a fixed rate, on
     ordinary pitches and on pitches with a gust or a spin change: hit quality against
     planner calls, and latency from the onset of a surprise to the next planner call.

The planner itself still runs on the EKF, which is fed by the delayed frame camera; only
the decision *when* to plan comes from events.

Outputs: results/m5.json, m5_events.png.

    env -u PYTHONPATH .venv/bin/python scripts/m5_events.py [--n 64]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from batter import RESULTS, config  # first: the package selects MuJoCo's EGL backend before mujoco is imported

import mujoco  # noqa: E402
from batter.envs.cage import BatEnv
from batter.estim.flight import FlightModel
from batter.eval import plotstyle as ps
from batter.eval.eventcam import EventRig, evaluate_events, use_software_rendering
from batter.eval.runner import evaluate, mean_std, summarize
from batter.eval.sweeps import at_budget
from batter.models.pitch import generate
from batter.models.scene import build_cage_model

GATES_EV = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.5, 7.0, 12.0]
GATES_ST = [1.0, 2.0, 3.0, 4.5, 6.5, 10.0, 16.0]
RATES = [25.0, 12.5, 8.0, 5.0, 3.3]
KINDS = ["none", "gust", "spin"]
BUDGET = 4.0
WINDOW = 0.15


def latency(out: dict, s_step: np.ndarray, h: float) -> np.ndarray:
    lat = np.full(len(out["hit"]), np.inf)
    for i, steps in enumerate(out["plan_steps"]):
        later = np.asarray(steps)[np.asarray(steps) > s_step[i]]
        if later.size:
            lat[i] = (later[0] - s_step[i]) * h
    return lat


def run(cfg, fm, kind: str, setting: float, n: int, surprise: str):
    """One trigger setting over the seeds; returns ([mean, std] metrics, latencies to the next call)."""
    per, lats = [], []
    for seed in cfg["seeds"]:
        P = generate(fm, cfg, n, 900 + seed, surprise=surprise)
        out = evaluate_events(cfg, setting, P, seed) if kind == "events" else evaluate(cfg, (kind, setting), P, seed)
        per.append(summarize(out))
        s_step = generate(fm, cfg, n, 900 + seed, surprise="gust").s_step       # the onset the pitch has (or would have had)
        lats.append(latency(out, s_step, fm.h))
    return mean_std(per), np.concatenate(lats)


def event_picture(cfg) -> np.ndarray:
    """Events of one pitch accumulated into an image: ON red, OFF blue, over the camera's first frame."""
    env = BatEnv(cfg, 1, nthread=1)
    rig = EventRig(env)
    cam = rig.cams[0]
    env.reset(generate(env.fm, cfg, 1, 5))
    env.batch.get_data(0, cam.data)
    mujoco.mj_forward(env.model, cam.data)
    grey = np.exp(cam.log_image(cam.data)) - 8.0
    img = np.repeat(np.clip(grey, 0, 255)[..., None], 3, axis=2).astype(np.uint8)
    cam.reset()
    for k in range(60):
        env.batch.get_data(0, cam.data)
        mujoco.mj_forward(env.model, cam.data)
        rows, cols, pol = cam.events(cam.data)
        # ON events only: the ball leaves every pixel it entered one step later, so each step's OFF
        # events fall exactly on the previous step's ON events and would hide them
        img[rows[pol > 0], cols[pol > 0]] = (235, 60, 50)
        env.step()
    img[:, cam.col_min - 1] = (255, 255, 255)
    rig.close()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="pitches per seed and setting")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from results/m5.json")
    args = ap.parse_args()
    cfg = config.load(overrides=args.set)
    if args.plot_only:
        plot(json.loads((RESULTS / "m5.json").read_text()), event_picture(cfg))
        return
    dev = use_software_rendering()                    # before the pool starts: every worker renders
    print(f"  event cameras render on EGL device {dev} (llvmpipe)" if dev is not None else
          "  no software EGL device found: rendering on the default device", flush=True)
    fm = FlightModel.identify(build_cage_model(cfg))
    res: dict = {"n": args.n, "budget": BUDGET, "window_s": WINDOW, "pareto": {}, "reaction": {}}
    lat: dict = {}
    for kind, settings in [("events", GATES_EV), ("innovation", GATES_ST), ("periodic", RATES)]:
        for sur in KINDS:
            rows = []
            for s in settings:
                row, l = run(cfg, fm, kind, s, args.n, sur)
                rows.append({"setting": s, **row})
                lat[(kind, sur, s)] = l
                print(f"  [{sur:4s}] {kind:10s} {s:6}: quality {row['quality'][0]:.3f} +- {row['quality'][1]:.3f}   hit {100 * row['hit'][0]:5.1f} %   "
                      f"calls/pitch {row['plans'][0]:5.2f}", flush=True)
            res["pareto"].setdefault(sur, {})[kind] = rows
            (RESULTS / "m5.json").write_text(json.dumps(res, indent=2))
    # reaction at the setting that costs about BUDGET calls on ordinary pitches
    for kind in ("events", "innovation", "periodic"):
        s = at_budget(res["pareto"]["none"][kind], BUDGET)["setting"]
        plain = next(r for r in res["pareto"]["none"][kind] if r["setting"] == s)
        res["reaction"][kind] = {"setting": s, "calls_plain": plain["plans"], "quality_plain": plain["quality"]}
        for sur in ("gust", "spin"):
            row = next(r for r in res["pareto"][sur][kind] if r["setting"] == s)
            ls, lp = lat[(kind, sur, s)], lat[(kind, "none", s)]
            res["reaction"][kind][sur] = {
                "calls": row["plans"], "quality": row["quality"], "hit": row["hit"],
                "react": float((ls <= WINDOW).mean()), "react_placebo": float((lp <= WINDOW).mean()),
                "latency_med_ms": float(1e3 * np.median(ls)), "latency_placebo_med_ms": float(1e3 * np.median(lp)),
                "latency": np.sort(ls).tolist(), "latency_placebo": np.sort(lp).tolist()}
            r = res["reaction"][kind][sur]
            print(f"  reaction [{sur}] {kind:10s} (setting {s}): calls {plain['plans'][0]:.1f} -> {row['plans'][0]:.1f}   quality "
                  f"{plain['quality'][0]:.2f} -> {row['quality'][0]:.2f}   called within {1e3 * WINDOW:.0f} ms {100 * r['react']:.0f} % "
                  f"(placebo {100 * r['react_placebo']:.0f} %)   median latency {r['latency_med_ms']:.0f} ms", flush=True)
    (RESULTS / "m5.json").write_text(json.dumps(res, indent=2))
    plot(res, event_picture(cfg))


def plot(res: dict, picture: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    ps.apply()
    col = {"events": ps.AQUA, "innovation": ps.YELLOW, "periodic": ps.BLUE}
    lab = {"events": "event trigger (event camera, no latency)", "innovation": "innovation trigger (frame camera, 30 ms late)",
           "periodic": "fixed rate"}
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.0), gridspec_kw={"width_ratios": [1.5, 1, 1, 1]})
    ax[0].imshow(picture)
    ax[0].set(title="ON events of one pitch; robot masked left of the line")
    ax[0].axis("off")
    for a, sur, title in [(ax[1], "none", "Ordinary pitches"), (ax[2], "gust", "Pitches with a gust")]:
        for kind in ("periodic", "innovation", "events"):
            rows = sorted(res["pareto"][sur][kind], key=lambda r: r["plans"][0])
            y = np.array([r["quality"] for r in rows])
            a.errorbar([r["plans"][0] for r in rows], y[:, 0], y[:, 1], color=col[kind], marker="o", label=lab[kind], capsize=2)
        a.set(xscale="log", xlabel="planner calls per pitch", ylabel="hit quality", title=title, ylim=(0, 1.02))
    ax[1].legend(fontsize=7, loc="lower right")
    grid = np.linspace(0.0, 0.4, 81)
    for kind in ("periodic", "innovation", "events"):
        r = res["reaction"][kind]["gust"]
        ax[3].plot(1e3 * grid, [100 * np.mean(np.array(r["latency"]) <= g) for g in grid], color=col[kind], label=lab[kind].split(" (")[0])
        ax[3].plot(1e3 * grid, [100 * np.mean(np.array(r["latency_placebo"]) <= g) for g in grid], color=col[kind], ls=":", lw=1.0)
    ax[3].plot([], [], color=ps.INK2, ls=":", lw=1.0, label="same pitches, no gust")
    ax[3].set(xlabel="time since the onset of the gust [ms]", ylabel="pitches with a planner call by then [%]",
              title="Latency to the next planner call")
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    ps.save(fig, RESULTS / "m5_events.png")


if __name__ == "__main__":
    main()
