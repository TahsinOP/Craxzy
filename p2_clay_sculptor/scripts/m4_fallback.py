"""M4: know when the model is wrong.

    env -u PYTHONPATH .venv/bin/python scripts/m4_fallback.py

The structured model is trained five times (deep ensemble).  The planner searches
with the ensemble mean; when the members disagree about the chosen action by more
than a threshold, the top candidates are re-ranked with oracle rollouts and the
truly best one is executed ("think harder only when unsure").

Compared at equal search budget:
  never      ensemble only, no oracle calls            (pure learned)
  always     re-rank at every decision
  disagreement(tau)   re-rank when the ensemble spread of the chosen action's map exceeds tau
  misrank(tau) re-rank when the members make it likely that another top candidate is better
  random(p)  re-rank with probability p                (is the *signal* worth anything?)
  oracle     MuJoCo for every candidate (M1)           (pure oracle)
Result: shape error against oracle calls per episode (Pareto plot).
Also: is the ensemble spread calibrated against the true one-step error, and does
either signal predict the gain of a re-ranking (measured at every decision of "always")?
"""
import argparse
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from clay_sculptor import DATA, RESULTS, config
from clay_sculptor.control.mpc import CEMCfg, HybridEval
from clay_sculptor.envs.targets import MAIN, make_target
from clay_sculptor.eval import plots
from clay_sculptor.eval.episodes import episode, mean_std
from clay_sculptor.learn import data as D
from clay_sculptor.learn import train as T

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="structured")
ap.add_argument("--members", nargs="*", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--targets", nargs="*", default=list(MAIN))
ap.add_argument("--seeds", nargs="*", type=int, default=None)
ap.add_argument("--quantiles", nargs="*", type=float, default=[0.25, 0.5, 0.75])
ap.add_argument("--triggers", nargs="*", default=["disagreement", "misrank", "random"])
ap.add_argument("--top-m", type=int, default=8)
ap.add_argument("--pop", type=int, default=32)
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out", default="m4_fallback")
ap.add_argument("--resume", action="store_true", help="keep the configurations already in results/<out>.json and run only the missing ones")
args = ap.parse_args()
torch.set_num_threads(args.threads)
cfg = config.load()
seeds = args.seeds if args.seeds is not None else cfg.raw["seeds"]
cem = CEMCfg(pop=args.pop, iters=args.iters)
env = cfg.env(nsim=16)
ens = T.load_ensemble(args.kind, args.members, env.geom)
targets = {n: make_target(env, n) for n in args.targets}
out = dict(kind=args.kind, members=args.members, top_m=args.top_m, seeds=seeds, cem=dict(pop=cem.pop, iters=cem.iters), configs={})

# ---- is the ensemble spread informative?  (held-out transitions) -----------------------------
data = D.load(DATA / "transitions_main.npz")
te = T.Transitions(data, T.split(data)[2], env.geom)
hb, ha = te.maps
with torch.no_grad():
    preds = torch.stack([torch.cat([m.heightmap(m(te.before[i:i + 64], te.tool[i:i + 64], te.action[i:i + 64]))
                                    for i in range(0, len(te), 64)]) for m in ens.members])       # (M, n, G, G)
spread = preds.std(0).pow(2).mean(dim=(-1, -2)).sqrt().numpy() * 1e3
err = T.map_rmse(preds.mean(0), ha).numpy() * 1e3
err_single = T.map_rmse(preds[0], ha).numpy() * 1e3
rho = spearmanr(spread, err)
order = np.argsort(spread)
bins = np.array_split(order, 8)
out["calibration"] = dict(n=len(te), spearman=float(rho.statistic), p_value=float(rho.pvalue),
                          ensemble_mean_rmse_mm=float(err.mean()), single_member_rmse_mm=float(err_single.mean()),
                          spread_mm=[float(spread[b].mean()) for b in bins], error_mm=[float(err[b].mean()) for b in bins],
                          top_quarter_share_of_error=float(err[order[-len(order) // 4:]].sum() / err.sum()))
print(f"calibration: Spearman {rho.statistic:.3f} (n = {len(te)}); ensemble mean {err.mean():.3f} mm, single member {err_single.mean():.3f} mm", flush=True)
np.savez_compressed(RESULTS / f"{args.out}_calibration.npz", spread=spread, err=err)


prev = {}
if args.resume and (RESULTS / f"{args.out}.json").exists():
    old = json.loads((RESULTS / f"{args.out}.json").read_text())
    if old.get("seeds") == seeds and old.get("members") == args.members and old.get("top_m") == args.top_m:
        prev = old["configs"]
        print(f"resuming: kept {list(prev)}", flush=True)


def run(label: str, **kw) -> dict:
    if label in prev:                                   # finished in an earlier run with the same settings
        out["configs"][label] = prev[label]
        return prev[label]["episodes"]
    res = {}
    for name, tgt in targets.items():
        eps = []
        for seed in seeds:
            t0 = time.time()
            summ, _ = episode(env, tgt, HybridEval(ens, env, top_m=args.top_m, **kw), seed, cem)
            summ["seed"], summ["wall_s"] = seed, time.time() - t0
            summ["fired"] = int(sum(i.get("fired", False) for i in summ["info"]))
            summ["decisions"] = len(summ["info"])
            eps.append(summ)
        res[name] = eps
    rem = np.array([[e["remaining"] for e in res[n]] for n in targets])            # (targets, seeds)
    calls = np.array([[e["plan_calls"] for e in res[n]] for n in targets])
    fired = sum(e["fired"] for n in targets for e in res[n])
    decisions = sum(e["decisions"] for n in targets for e in res[n])
    summary = dict(remaining=mean_std(rem.mean(axis=0)), calls_per_episode=mean_std(calls.mean(axis=0)),
                   fire_rate=fired / max(decisions, 1), by_target={n: dict(map_rmse_mm=mean_std([e["map_rmse_mm"] for e in res[n]]),
                                                                           remaining=mean_std([e["remaining"] for e in res[n]]),
                                                                           calls=mean_std([e["plan_calls"] for e in res[n]])) for n in targets},
                   plan_time_per_decision_s=mean_std([e["plan_time_per_decision_s"] for n in targets for e in res[n]]))
    out["configs"][label] = dict(params={k: (float(v) if isinstance(v, (int, float)) else v) for k, v in kw.items()}, episodes=res, summary=summary)
    print(f"{label:22s}: {100 * summary['remaining'][0]:.1f} +- {100 * summary['remaining'][1]:.1f} % left, "
          f"{summary['calls_per_episode'][0]:.1f} oracle calls per episode, fire rate {summary['fire_rate']:.2f}", flush=True)
    (RESULTS / f"{args.out}.json").write_text(json.dumps(out, indent=1))
    return res


never = run("never", trigger="never")
signal = {k: np.array([i[k] for n in targets for e in never[n] for i in e["info"]]) for k in ("spread", "regret", "misrank")}
out["at_decisions"] = dict(spread_mm=(1e3 * signal["spread"]).tolist(), regret_mm=(1e3 * signal["regret"]).tolist(),
                           misrank=signal["misrank"].tolist(), share_regret_zero=float((signal["regret"] == 0).mean()))
print(f"at decisions of 'never': members all rank the chosen action first at {100 * out['at_decisions']['share_regret_zero']:.0f} %", flush=True)
always = run("always", trigger="always")
# which signal predicts what a re-ranking gains?  gain = true cost of the model's choice - true cost of the re-ranked choice
dec = [(i, e["predicted_mm"][k]) for n in targets for e in always[n] for k, i in enumerate(e["info"])]
gain = np.array([1e3 * i["true_cost_of_model_best"] - p for i, p in dec])
out["signal"] = dict(n=len(gain), gain_mm=gain.tolist(), mean_gain_mm=float(gain.mean()),
                     share_of_gain_in_top_quarter={}, spearman={})
for k in ("spread", "regret", "misrank"):
    x = np.array([i[k] for i, _ in dec])
    out["signal"][k] = x.tolist()
    out["signal"]["spearman"][k] = float(spearmanr(x, gain).statistic)
    top = np.argsort(x)[-max(len(x) // 4, 1):]
    out["signal"]["share_of_gain_in_top_quarter"][k] = float(gain[top].sum() / max(gain.sum(), 1e-12))
    print(f"signal {k:8s}: Spearman with the re-ranking gain {out['signal']['spearman'][k]:+.3f} (n = {len(gain)}), "
          f"top quarter holds {100 * out['signal']['share_of_gain_in_top_quarter'][k]:.0f} % of the gain", flush=True)
for trig in args.triggers:
    for q in args.quantiles:
        if trig == "random":
            run(f"random p{int(100 * (1 - q))}", trigger="random", p=1 - q)
        else:
            run(f"{trig} q{int(100 * q)}", trigger=trig, tau=float(np.quantile(signal["spread" if trig == "disagreement" else trig], q)))

for k, v in prev.items():                              # keep earlier configurations that were not rerun
    out["configs"].setdefault(k, v)

m1 = RESULTS / "m1_oracle.json"
if m1.exists():
    o = json.loads(m1.read_text())
    rem = np.array([[e["remaining"] for e in o["targets"][n]["episodes"]] for n in targets])
    calls = np.array([[e["plan_calls"] for e in o["targets"][n]["episodes"]] for n in targets])
    out["oracle"] = dict(remaining=mean_std(rem.mean(axis=0)), calls_per_episode=mean_std(calls.mean(axis=0)))
(RESULTS / f"{args.out}.json").write_text(json.dumps(out, indent=1))
plots.m4_pareto(out, RESULTS / f"{args.out}.png")
plots.m4_calibration(spread, err, out["calibration"], RESULTS / f"{args.out}_calibration.png")
print(f"wrote results/{args.out}.json", flush=True)
