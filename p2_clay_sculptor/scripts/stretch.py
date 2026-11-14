"""Stretch goals: sculpt a letter, and choose the tool.

    env -u PYTHONPATH .venv/bin/python scripts/stretch.py

* ``t``      the letter T, five presses in the demonstration; the planner gets seven.
* ``combo``  a rolled band, a stamped cross and a pressed pocket.  The tool is not
             given: at every step the planner searches with each of the three tools
             and executes the best prediction (three times the search budget).

Planners (``--planners``): the structured model alone, the structured ensemble with
oracle re-ranking when it disagrees (the M4 hybrid at its median threshold), and
``oracle``, MuJoCo itself (slow: about 100 simulated primitives per decision and
tool; one seed).  Results accumulate in results/stretch.json across runs.
"""
import argparse
import json
import time

import numpy as np
import torch

from clay_sculptor import RESULTS, config
from clay_sculptor.control.mpc import CEMCfg, HybridEval, ModelEval, OracleEval
from clay_sculptor.envs.targets import make_target
from clay_sculptor.eval import plots, render
from clay_sculptor.eval.episodes import episode, mean_std
from clay_sculptor.learn import train as T

ap = argparse.ArgumentParser()
ap.add_argument("--targets", nargs="*", default=["t", "combo"])
ap.add_argument("--seeds", nargs="*", type=int, default=None)
ap.add_argument("--planners", nargs="*", default=["structured", "fallback"], choices=["structured", "fallback", "oracle"],
                help="results of planners not listed are kept from an earlier run")
ap.add_argument("--nthread", type=int, default=0)
ap.add_argument("--members", nargs="*", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--threads", type=int, default=8)
args = ap.parse_args()
torch.set_num_threads(args.threads)
cfg = config.load()
seeds = args.seeds if args.seeds is not None else cfg.raw["seeds"]
env = cfg.env(nsim=32, nthread=args.nthread)
cem = CEMCfg()
m4 = RESULTS / "m4_fallback.json"
tau = float(np.median(json.loads(m4.read_text())["spread_at_decisions_mm"])) * 1e-3 if m4.exists() else 3e-4
planners = {}
if "structured" in args.planners:
    planners["structured"] = lambda s: ModelEval(T.load("structured", s, env.geom), env)
if "fallback" in args.planners:
    ens = T.load_ensemble("structured", args.members, env.geom)
    planners["fallback"] = lambda s: HybridEval(ens, env, trigger="disagreement", tau=tau)
if "oracle" in args.planners:
    planners["oracle"] = lambda s: OracleEval(env)
prev_json, prev_npz = RESULTS / "stretch.json", RESULTS / "stretch_shapes.npz"
out = json.loads(prev_json.read_text()) if prev_json.exists() else dict(seeds=seeds, targets={})
shapes = dict(np.load(prev_npz)) if prev_npz.exists() else {}
out["tau_mm"], out["seeds"] = 1e3 * tau, seeds
for name in args.targets:
    tgt = make_target(env, name)
    shapes[f"{name}_target"] = tgt.nodes.astype(np.float32)
    out["targets"].setdefault(name, dict(planners={}))
    out["targets"][name].update(demo=[[t, list(map(float, a))] for t, a in tgt.demo], initial_mm=1e3 * tgt.initial_cost)
    from clay_sculptor.envs.targets import shape_errors  # noqa: E402
    out["targets"][name]["untouched"] = shape_errors(env, env.state0, tgt)
    for pname, make in planners.items():
        eps = []
        for seed in (seeds if pname != "oracle" else seeds[:1]):
            t0 = time.time()
            summ, nodes = episode(env, tgt, make(seed), seed, cem)
            summ["seed"], summ["wall_s"] = seed, time.time() - t0
            eps.append(summ)
            shapes[f"{pname}_{name}_seed{seed}"] = nodes
        agg = {k: mean_std([e[k] for e in eps]) for k in ("map_rmse_mm", "chamfer_mm", "emd_mm", "remaining", "n_actions", "plan_calls", "plan_time_s")}
        out["targets"][name]["planners"][pname] = dict(episodes=eps, summary=agg)
        print(f"{name:6s} {pname:10s}: RMSE {1e3 * tgt.initial_cost:.2f} -> {agg['map_rmse_mm'][0]:.2f} +- {agg['map_rmse_mm'][1]:.2f} mm, "
              f"{agg['n_actions'][0]:.1f} actions, {agg['plan_calls'][0]:.0f} oracle rollouts, tools {eps[0]['tools']}", flush=True)
        (RESULTS / "stretch.json").write_text(json.dumps(out, indent=1))
np.savez_compressed(RESULTS / "stretch_shapes.npz", **shapes)
done = [p for p in ("structured", "fallback", "oracle") if all(p in out["targets"][t]["planners"] for t in out["targets"])]
plots.stretch_figure(env.geom, shapes, list(out["targets"]), done, seeds[0], RESULTS / "stretch.png")
print("wrote results/stretch.json", flush=True)
