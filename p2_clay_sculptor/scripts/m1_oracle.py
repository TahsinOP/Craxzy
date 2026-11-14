"""M1: oracle MPC.  CEM over tool primitives with MuJoCo itself as the rollout model.

    env -u PYTHONPATH .venv/bin/python scripts/m1_oracle.py

Three targets (flat slab, single dent, X stamp) x seeds.  Reports the height-map
RMSE, Chamfer and EMD to the target, the number of actions, the number of
simulated primitives used for planning and the wall-clock planning time.  This is
the upper bound for the learned-model planners of M3.
"""
import argparse
import json
import time

import numpy as np

from clay_sculptor import RESULTS, config
from clay_sculptor.control.mpc import CEMCfg, OracleEval
from clay_sculptor.envs.targets import MAIN, make_target, shape_errors
from clay_sculptor.eval import plots
from clay_sculptor.eval.episodes import episode, mean_std

ap = argparse.ArgumentParser()
ap.add_argument("--targets", nargs="*", default=list(MAIN))
ap.add_argument("--seeds", nargs="*", type=int, default=None)
ap.add_argument("--pop", type=int, default=32)
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--out", default="m1_oracle")
ap.add_argument("--set", nargs="*", default=[])
args = ap.parse_args()
cfg = config.load(overrides=args.set)
seeds = args.seeds if args.seeds is not None else cfg.raw["seeds"]
cem = CEMCfg(pop=args.pop, iters=args.iters)
env = cfg.env(nsim=max(args.pop, 16))
oracle = OracleEval(env)
out = dict(cem=dict(pop=cem.pop, iters=cem.iters, elite=cem.elite, guided=cem.guided), seeds=seeds, targets={})
shapes = {}
h0 = env.geom.heightmap(env.nodes(env.state0))
for name in args.targets:
    tgt = make_target(env, name)
    shapes[f"{name}_target"] = tgt.nodes.astype(np.float32)
    eps = []
    for seed in seeds:
        t0 = time.time()
        summ, nodes = episode(env, tgt, oracle, seed, cem)
        summ["seed"], summ["wall_s"] = seed, time.time() - t0
        eps.append(summ)
        shapes[f"{name}_seed{seed}"] = nodes
        print(f"{name} seed {seed}: RMSE {summ['initial_mm']:.2f} -> {summ['map_rmse_mm']:.3f} mm ({100 * summ['remaining']:.1f} % left), "
              f"chamfer {summ['chamfer_mm']:.3f} mm, EMD {summ['emd_mm']:.3f} mm, {summ['n_actions']} actions, "
              f"{summ['plan_calls']} oracle calls, {summ['plan_time_s']:.0f} s planning", flush=True)
    agg = {k: mean_std([e[k] for e in eps]) for k in ("map_rmse_mm", "chamfer_mm", "emd_mm", "remaining", "n_actions", "plan_calls", "plan_time_s")}
    untouched = shape_errors(env, env.state0, tgt)
    out["targets"][name] = dict(tool=tgt.tool, demo=[[t, list(map(float, a))] for t, a in tgt.demo], max_steps=tgt.max_steps,
                                untouched=untouched, episodes=eps, summary=agg)
    print(f"== {name}: RMSE {agg['map_rmse_mm'][0]:.3f} +- {agg['map_rmse_mm'][1]:.3f} mm (untouched {untouched['map_rmse_mm']:.2f}), "
          f"calls {agg['plan_calls'][0]:.0f}, planning {agg['plan_time_s'][0]:.0f} s", flush=True)
    (RESULTS / f"{args.out}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(RESULTS / f"{args.out}_shapes.npz", **shapes)

plots.targets_figure(env.geom, h0, shapes, args.targets, seeds, RESULTS / f"{args.out}.png", "oracle MPC")
plots.cost_curves({n: [e["costs_mm"] for e in out["targets"][n]["episodes"]] for n in args.targets}, RESULTS / f"{args.out}_curves.png")
print(f"wrote results/{args.out}.json", flush=True)
