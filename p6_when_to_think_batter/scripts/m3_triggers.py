"""M3: compute triggers.  Hit quality against planner calls for five triggers.

  periodic    : fixed rate (100 Hz is the M1 baseline; the rate is swept down to "plan once")
  innovation  : chi-square gate on the camera's innovation against the plan's forecast
  variance    : tr C(t_plan) - tr C(t) > delta^2          (covariances only; Trimpe & D'Andrea)
  drift       : the crossing-point estimate has moved by more than delta (send-on-delta)
  learned     : PPO policy, reward = hit quality - lam * planner calls (one policy, lam is an input)
  (wm)        : the M2 world-model planner at fixed rates, as the pure-learning baseline

Outputs: data/models/trigger_nominal.pt, results/m3.json, m3_pareto.png, m3_when.png, m3_ppo.png.

    env -u PYTHONPATH .venv/bin/python scripts/m3_triggers.py [--n 200] [--ppo-iters 150] [--ppo-batch 192]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from batter import DATA, RESULTS, config
from batter.estim.flight import FlightModel
from batter.eval import plotstyle as ps
from batter.eval.runner import evaluate
from batter.eval.sweeps import KINDS, LABEL, LAMS, at_budget, policy_path, sweep, train_ppo
from batter.models.pitch import generate
from batter.models.scene import build_cage_model

BUDGET = 4.0        # planner calls per pitch at which the triggers' timing is compared
WM_RATES = [1000.0, 25.0, 10.0, 5.0, 2.5]


def pareto_plot(res: dict, path, title: str) -> None:
    import matplotlib.pyplot as plt

    ps.apply()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    markers = {"periodic": "o", "innovation": "s", "variance": "^", "drift": "v", "learned": "D", "wm": "P"}
    for kind in ("periodic", "innovation", "variance", "drift", "learned", "wm"):
        if kind not in res:
            continue
        rows = sorted(res[kind], key=lambda r: r["plans"][0])
        x = np.array([r["plans"][0] for r in rows])
        for a, key, xs in [(ax[0], "quality", x), (ax[1], "hit", x),
                           (ax[2], "quality", np.array([r["ms_per_pitch"][0] for r in rows]))]:
            y = np.array([r[key] for r in rows])
            sc = 100 if key == "hit" else 1
            a.errorbar(xs, sc * y[:, 0], sc * y[:, 1], color=ps.TRIGGER[kind], marker=markers[kind], label=LABEL[kind],
                       capsize=2, lw=2.4 if kind in ("learned", "periodic") else 1.5)
    ax[0].set(xscale="log", xlabel="planner calls per pitch", ylabel="hit quality", title=title)
    ax[1].set(xscale="log", xlabel="planner calls per pitch", ylabel="hit rate [%]", title="Hit rate")
    ax[2].set(xscale="log", xlabel="planning time per pitch [ms, wall-clock on one core]", ylabel="hit quality",
              title="The same against measured compute")
    ax[0].legend(loc="lower right")
    fig.tight_layout()
    ps.save(fig, path)


def when_plot(cfg: dict, fm, res: dict, n: int) -> dict:
    """When in the flight does each trigger call the planner, at the same budget?"""
    import matplotlib.pyplot as plt

    P = generate(fm, cfg, n, 777)
    ps.apply()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    bins = np.linspace(0.0, 0.7, 29)
    info = {}
    for kind in ("periodic", "innovation", "variance", "drift", "learned"):
        row = at_budget(res[kind], BUDGET)
        spec = ("learned", (str(policy_path("nominal")), row["setting"])) if kind == "learned" else (kind, row["setting"])
        out = evaluate(cfg, spec, P, 0)
        tau = np.concatenate([P.cross_t[i] - np.asarray(s, float)[1:] * fm.h for i, s in enumerate(out["plan_steps"])])
        hist, _ = np.histogram(tau, bins)
        ax.plot(1e3 * 0.5 * (bins[:-1] + bins[1:]), hist / n, color=ps.TRIGGER[kind], marker=".",
                label=f"{LABEL[kind]}: {row['plans'][0]:.1f} calls, quality {row['quality'][0]:.2f}")
        info[kind] = {"setting": row["setting"], "plans": row["plans"], "quality": row["quality"],
                      "median_tau_ms": float(1e3 * np.median(tau)) if tau.size else float("nan")}
    ax.invert_xaxis()
    ax.set(xlabel="time left to the plate [ms]  (the pitch runs left to right)", ylabel="replans per pitch per 25 ms",
           title=f"When each trigger thinks, at about {BUDGET:.0f} planner calls per pitch")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    ps.save(fig, RESULTS / "m3_when.png")
    return info


def ppo_plot(name: str, path) -> None:
    import matplotlib.pyplot as plt

    log = torch.load(policy_path(name), map_location="cpu")["log"]
    ps.apply()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    it = [r["iter"] for r in log]
    smooth = lambda y: np.convolve(np.asarray(y, float), np.ones(8) / 8, mode="valid")
    for a, key, lab in [(ax[0], "return", "return = quality - lam * replans"), (ax[1], "quality", "hit quality"),
                        (ax[2], "plans", "planner calls per pitch")]:
        a.plot(it, [r[key] for r in log], color=ps.AQUA, alpha=0.35)
        a.plot(it[7:], smooth([r[key] for r in log]), color=ps.AQUA)
        a.set(xlabel="PPO iteration", title=lab)
    fig.tight_layout()
    ps.save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="pitches per seed and setting")
    ap.add_argument("--ppo-iters", type=int, default=150)
    ap.add_argument("--ppo-batch", type=int, default=192)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = config.load(overrides=args.set)
    RESULTS.mkdir(exist_ok=True)
    fm = FlightModel.identify(build_cage_model(cfg))
    path = RESULTS / "m3.json"
    res: dict = {"n": args.n}

    train_ppo("nominal", cfg, fm, args.ppo_iters, args.ppo_batch, surprise_frac=0.0, force=args.retrain)
    ppo_plot("nominal", RESULTS / "m3_ppo.png")
    for kind, settings in KINDS.items():
        sweep(cfg, fm, kind, settings, args.n, "none", res, path)
    sweep(cfg, fm, "learned", LAMS, args.n, "none", res, path, make_spec=lambda lam: ("learned", (str(policy_path("nominal")), lam)))
    wm_path = DATA / "models" / "wm_s0.pt"                       # pure-learning baseline: the M2 planner at fixed rates
    if wm_path.exists():
        sweep(cfg, fm, "wm", WM_RATES, args.n, "none", res, path, make_spec=lambda r: ("periodic", r),
              env_kw={"planner": str(wm_path)})
    pareto_plot(res, RESULTS / "m3_pareto.png", "Hit quality against planner calls")
    res["when"] = when_plot(cfg, fm, res, 300)
    path.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
