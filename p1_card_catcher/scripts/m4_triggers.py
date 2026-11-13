"""M4: event-triggered replanning.  Catch rate against planner calls for four triggers.

  periodic  : replan at a fixed rate (50 Hz is the baseline; the rate is swept)
  variance  : replan when tr P(t_plan) - tr P(t) > delta^2          (covariances only)
  drift     : replan when the catch-point estimate has moved by more than delta
  learned   : PPO policy, reward = catch - lam * replans (one policy, lam is an input)

Outputs: data/models/trigger_<obj>.pt, results/m4_<obj>.json, results/m4_pareto.png.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from card_catcher import DATA, RESULTS, config
from card_catcher.control.triggers import Drift, Learned, Periodic, Variance
from card_catcher.envs.catch import DEFAULT_PARAMS, DEFAULT_PREDICTOR, CatchEnv, run_episode
from card_catcher.envs.flight import generate
from card_catcher.estim.predictor import make_predictor
from card_catcher.eval import plotstyle as ps
from card_catcher.learn import ppo

RATES = [125.0, 50.0, 25.0, 12.5, 6.25, 3.125, 0.0]          # Hz; 0 = plan once
DELTAS_VAR = [0.002, 0.004, 0.008, 0.015, 0.03, 0.06]         # m
DELTAS_DRIFT = [0.002, 0.004, 0.008, 0.015, 0.03, 0.06]
LAMS = [0.001, 0.003, 0.01, 0.03, 0.1]


def train_ppo(obj: str, cfg: dict, iters: int, B: int, seed: int = 0) -> ppo.TriggerPolicy:
    path = DATA / "models" / f"trigger_{obj}.pt"
    policy = ppo.TriggerPolicy()
    if path.exists():
        policy.load_state_dict(torch.load(path, map_location="cpu")["state"])
        return policy
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = CatchEnv(obj, cfg, B, make_predictor(DEFAULT_PREDICTOR[obj], obj, B, cfg), DEFAULT_PARAMS[obj], seed=seed)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    log, t0 = [], time.time()
    for it in range(iters):
        D = generate(obj, B, cfg, seed=5000 + it)
        ro = ppo.collect(env, D, np.arange(B), policy, rng)
        info = ppo.update(policy, opt, ro, rng)
        log.append({"iter": it, **ro.stats, **info, "decisions": int(len(ro.actions)), "seconds": time.time() - t0})
        if it % 5 == 0 or it == iters - 1:
            print(f"  ppo[{obj}] it {it:3d}: return {ro.stats['return']:+.3f}  catch {ro.stats['catch']:.2f}  plans/throw "
                  f"{ro.stats['plans']:.1f}  p(replan) {info['p_replan']:.2f}  entropy {info['entropy']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": policy.state_dict(), "log": log}, path)
    (RESULTS / f"m4_ppo_log_{obj}.json").write_text(json.dumps(log))
    return policy


def evaluate(obj: str, cfg: dict, trigger_fn, n: int, seed: int, B: int) -> dict:
    D = generate(obj, n, cfg, seed=600 + seed)
    caught, plans, wall = [], [], 0.0
    for i in range(0, n, B):
        idx = np.arange(i, min(i + B, n))
        env = CatchEnv(obj, cfg, len(idx), make_predictor(DEFAULT_PREDICTOR[obj], obj, len(idx), cfg), DEFAULT_PARAMS[obj], seed=seed)
        t0 = time.time()
        out = run_episode(env, D, idx, trigger_fn(seed))
        wall += time.time() - t0
        caught.append(out["caught"]), plans.append(out["n_plans"])
    return {"catch": float(np.concatenate(caught).mean()), "plans": float(np.concatenate(plans).mean()), "wall_s_per_throw": wall / n}


def sweep(obj, cfg, name, settings, make, n, B, out, path):
    rows = []
    for s in settings:
        per = [evaluate(obj, cfg, lambda seed, s=s: make(s, seed), n, seed, B) for seed in cfg["seeds"]]
        row = {"setting": s, **{k: [float(np.mean([p[k] for p in per])), float(np.std([p[k] for p in per]))] for k in per[0]}}
        rows.append(row)
        print(f"  {obj} {name:9s} {s:>8}: catch {row['catch'][0]*100:5.1f} +- {row['catch'][1]*100:.1f} %   planner calls/throw "
              f"{row['plans'][0]:6.2f}   {row['wall_s_per_throw'][0]*1e3:.0f} ms/throw", flush=True)
        out[name] = rows
        path.write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="*", default=["card", "ball"])
    ap.add_argument("--n", type=int, default=128, help="throws per seed and setting")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--ppo-iters", type=int, default=60)
    ap.add_argument("--ppo-batch", type=int, default=96)
    args = ap.parse_args()
    torch.set_num_threads(4)
    cfg = config.load()
    if args.plot_only:
        plot([o for o in ("card", "ball") if (RESULTS / f"m4_{o}.json").exists()])
        return
    for obj in args.objects:
        path = RESULTS / f"m4_{obj}.json"
        out = {"n_per_seed": args.n, "seeds": cfg["seeds"], "predictor": DEFAULT_PREDICTOR[obj], "ppo_batch": args.ppo_batch}
        sweep(obj, cfg, "periodic", RATES, lambda r, seed: Periodic(r), args.n, args.batch, out, path)
        sweep(obj, cfg, "variance", DELTAS_VAR, lambda d, seed: Variance(d), args.n, args.batch, out, path)
        sweep(obj, cfg, "drift", DELTAS_DRIFT, lambda d, seed: Drift(d), args.n, args.batch, out, path)
        policy = train_ppo(obj, cfg, args.ppo_iters, args.ppo_batch)
        sweep(obj, cfg, "learned", LAMS, lambda lam, seed: Learned(policy, lam, np.random.default_rng(900 + seed)), args.n, args.batch, out, path)
    if len(args.objects) > 1:
        plot(args.objects)


def plot(objects) -> None:
    ps.apply()
    import matplotlib.pyplot as plt

    names = {"periodic": "periodic (fixed rate)", "variance": "variance-triggered", "drift": "drift-triggered", "learned": "learned (PPO)"}
    cols = {"periodic": ps.BLUE, "variance": ps.YELLOW, "drift": ps.VIOLET, "learned": ps.AQUA}
    marks = {"periodic": "o", "variance": "s", "drift": "D", "learned": "^"}
    fig, ax = plt.subplots(1, len(objects), figsize=(5.6 * len(objects), 4.2), squeeze=False)
    for a, obj in zip(ax[0], objects):
        res = json.loads((RESULTS / f"m4_{obj}.json").read_text())
        for name in ("periodic", "variance", "drift", "learned"):
            if name not in res:
                continue
            rows = sorted(res[name], key=lambda r: r["plans"][0])
            x = [r["plans"][0] for r in rows]
            y = np.array([r["catch"][0] * 100 for r in rows])
            e = np.array([r["catch"][1] * 100 for r in rows])
            a.errorbar(x, y, yerr=e, color=cols[name], marker=marks[name], label=names[name], capsize=2, elinewidth=1, markersize=6)
            if name == "periodic":
                for r in rows:
                    if r["setting"] == 50.0:
                        a.annotate("50 Hz", (r["plans"][0], r["catch"][0] * 100), textcoords="offset points", xytext=(4, -13),
                                   fontsize=8.5, color=ps.INK2)
        a.set(xscale="log", xlabel="planner calls per throw", ylabel="catch rate [%]",
              title=f"{obj}: catch rate vs planner calls ({res['predictor']} predictor)")
        a.legend(loc="lower right")
    fig.tight_layout()
    ps.save(fig, RESULTS / "m4_pareto.png")
    print("figure:", RESULTS / "m4_pareto.png")


if __name__ == "__main__":
    main()
