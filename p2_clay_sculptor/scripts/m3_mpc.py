"""M3: MPC with each learned model in place of the oracle.

    env -u PYTHONPATH .venv/bin/python scripts/m3_mpc.py

Same planner, same search budget and same targets as M1; only the rollout model
changes: the geometric prior alone (classical), the GNN and the height-map U-Net
(pure learning) and the structured model (hybrid).  Model seed i plans with
planner seed i.  The oracle rows are read from results/m1_oracle.json.
"""
import argparse
import json
import time

import numpy as np
import torch

from clay_sculptor import RESULTS, config
from clay_sculptor.control.mpc import CEMCfg, ModelEval
from clay_sculptor.envs.targets import MAIN, make_target
from clay_sculptor.eval import plots
from clay_sculptor.eval.episodes import episode, mean_std
from clay_sculptor.learn import train as T

ap = argparse.ArgumentParser()
ap.add_argument("--kinds", nargs="*", default=["prior", "gnn", "structured", "unet"])
ap.add_argument("--targets", nargs="*", default=list(MAIN))
ap.add_argument("--seeds", nargs="*", type=int, default=None)
ap.add_argument("--pop", type=int, default=32)
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out", default="m3_mpc")
args = ap.parse_args()
torch.set_num_threads(args.threads)
cfg = config.load()
seeds = args.seeds if args.seeds is not None else cfg.raw["seeds"]
cem = CEMCfg(pop=args.pop, iters=args.iters)
env = cfg.env(nsim=16)
keys = ("map_rmse_mm", "chamfer_mm", "emd_mm", "remaining", "n_actions", "plan_calls", "plan_time_s", "plan_time_per_decision_s")
out = dict(cem=dict(pop=cem.pop, iters=cem.iters), seeds=seeds, methods={})
shapes = {}
targets = {n: make_target(env, n) for n in args.targets}
for n, t in targets.items():
    shapes[f"{n}_target"] = t.nodes.astype(np.float32)
for kind in args.kinds:
    out["methods"][kind] = {}
    for name, tgt in targets.items():
        eps = []
        for seed in seeds:
            model = T.load(kind, seed % 3, env.geom)          # three trained seeds per model, cycled over planner seeds
            t0 = time.time()
            summ, nodes = episode(env, tgt, ModelEval(model, env), seed, cem)
            summ["seed"], summ["wall_s"] = seed, time.time() - t0
            eps.append(summ)
            shapes[f"{kind}_{name}_seed{seed}"] = nodes
        agg = {k: mean_std([e[k] for e in eps]) for k in keys}
        out["methods"][kind][name] = dict(episodes=eps, summary=agg)
        print(f"{kind:11s} {name:5s}: RMSE {agg['map_rmse_mm'][0]:.3f} +- {agg['map_rmse_mm'][1]:.3f} mm "
              f"({100 * agg['remaining'][0]:.1f} % left), {agg['n_actions'][0]:.1f} actions, "
              f"{agg['plan_time_per_decision_s'][0]:.2f} s per decision", flush=True)
        (RESULTS / f"{args.out}.json").write_text(json.dumps(out, indent=1))

m1 = RESULTS / "m1_oracle.json"
if m1.exists():
    o = json.loads(m1.read_text())
    out["methods"]["oracle"] = {}
    for n in args.targets:
        eps = o["targets"][n]["episodes"]
        out["methods"]["oracle"][n] = dict(episodes=eps, summary={k: mean_std([e[k] for e in eps]) for k in keys})
    out["untouched"] = {n: o["targets"][n]["untouched"] for n in args.targets}
# across targets: fraction of the initial error that is left, and planning time per decision
for kind, res in out["methods"].items():
    rem = np.array([[e["remaining"] for e in res[n]["episodes"]] for n in args.targets])        # (targets, seeds)
    tpd = np.array([[e["plan_time_per_decision_s"] for e in res[n]["episodes"]] for n in args.targets])
    res["overall"] = dict(remaining=mean_std(rem.mean(axis=0)), plan_time_per_decision_s=mean_std(tpd.mean(axis=0)),
                          plan_calls=mean_std(np.array([[e["plan_calls"] for e in res[n]["episodes"]] for n in args.targets]).sum(axis=0)))
    print(f"== {kind:11s} overall: {100 * res['overall']['remaining'][0]:.1f} +- {100 * res['overall']['remaining'][1]:.1f} % of the initial error left, "
          f"{res['overall']['plan_time_per_decision_s'][0]:.2f} s per decision", flush=True)
(RESULTS / f"{args.out}.json").write_text(json.dumps(out, indent=1))
np.savez_compressed(RESULTS / f"{args.out}_shapes.npz", **shapes)
plots.m3_tradeoff(out, args.targets, RESULTS / f"{args.out}.png")
plots.m3_shapes(env.geom, shapes, args.targets, [k for k in ("prior", "gnn", "structured", "unet") if k in args.kinds], seeds[0],
                RESULTS / f"{args.out}_shapes.png", RESULTS / "m1_oracle_shapes.npz")
print(f"wrote results/{args.out}.json", flush=True)
