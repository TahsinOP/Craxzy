"""Shared by M3 and M4: the trigger settings that are swept, PPO training, evaluation rows."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from batter import DATA
from batter.eval.runner import collect_parallel, evaluate, mean_std, summarize
from batter.learn import ppo
from batter.models.pitch import Pitches, generate

RATES = [1000.0, 50.0, 25.0, 12.5, 8.0, 5.0, 3.3, 2.5, 0.0]        # Hz; 1000 = every step (100 Hz), 0 = plan once
GATES = [1.0, 2.0, 3.0, 4.5, 6.5, 10.0, 16.0, 30.0]                # chi-square gate on the mean innovation (3 dof: mean 3)
DELTAS_VAR = [0.004, 0.008, 0.015, 0.03, 0.06, 0.12, 0.25]         # m
DELTAS_DRIFT = [0.004, 0.008, 0.015, 0.03, 0.06, 0.12, 0.25]       # m
LAMS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

KINDS = {"periodic": RATES, "innovation": GATES, "variance": DELTAS_VAR, "drift": DELTAS_DRIFT}
LABEL = {"periodic": "fixed rate", "innovation": "innovation (chi-square gate)", "variance": "variance (Trimpe)",
         "drift": "drift (send-on-delta)", "learned": "learned (PPO)", "wm": "world-model planner"}


def policy_path(name: str) -> Path:
    return DATA / "models" / f"trigger_{name}.pt"


def mixed_pitches(fm, cfg: dict, n: int, seed: int, surprise_frac: float) -> Pitches:
    """n pitches of which a fraction carries a mid-flight surprise (spin change or gust)."""
    P = generate(fm, cfg, n, seed, surprise="mixed")
    keep = np.random.default_rng(seed + 31).random(n) < surprise_frac
    P.s_kind[~keep] = 0
    return P


def train_ppo(name: str, cfg: dict, fm, iters: int, batch: int, surprise_frac: float, seed: int = 0, lr: float = 3e-4,
              force: bool = False) -> ppo.TriggerPolicy:
    """Train (or load) the trigger policy ``name``; ``surprise_frac`` of the training pitches carry a surprise."""
    path = policy_path(name)
    policy = ppo.TriggerPolicy()
    if path.exists() and not force:
        policy.load_state_dict(torch.load(path, map_location="cpu")["state"])
        return policy
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    log, t0 = [], time.time()
    for it in range(iters):
        P = mixed_pitches(fm, cfg, batch, 5000 + 97 * seed + it, surprise_frac)
        ro = collect_parallel(cfg, policy, P, seed=10_000 * (seed + 1) + it)
        info = ppo.update(policy, opt, ro, rng)
        log.append({"iter": it, **ro.stats, **info, "decisions": int(len(ro.actions)), "seconds": time.time() - t0})
        if it % 10 == 0 or it == iters - 1:
            print(f"  ppo[{name}] it {it:3d}: return {ro.stats['return']:+.3f}  quality {ro.stats['quality']:.3f}  "
                  f"calls/pitch {ro.stats['plans']:.1f}  p(replan) {info['p_replan']:.2f}  entropy {info['entropy']:.2f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": policy.state_dict(), "log": log, "surprise_frac": surprise_frac}, path)
    return policy


def eval_setting(cfg: dict, fm, spec, n: int, surprise: str = "none", seed_base: int = 600, env_kw: dict | None = None,
                 keep: bool = False):
    """One trigger setting over the config's seeds: [mean, std] of every metric (and the raw outcomes)."""
    per, raw = [], []
    for seed in cfg["seeds"]:
        out = evaluate(cfg, spec, generate(fm, cfg, n, seed_base + seed, surprise=surprise), seed, env_kw=env_kw)
        per.append(summarize(out))
        if keep:
            raw.append(out)
    return (mean_std(per), raw) if keep else mean_std(per)


def sweep(cfg: dict, fm, kind: str, settings, n: int, surprise: str, out: dict, path: Path, make_spec=None, tag: str = "",
          env_kw: dict | None = None) -> list[dict]:
    rows = []
    for s in settings:
        spec = make_spec(s) if make_spec else (kind, s)
        row = {"setting": s, **eval_setting(cfg, fm, spec, n, surprise, env_kw=env_kw)}
        rows.append(row)
        print(f"  {tag}{kind:10s} {s:>8}: quality {row['quality'][0]:.3f} +- {row['quality'][1]:.3f}   hit {100 * row['hit'][0]:5.1f} %   "
              f"on target {100 * row['on_target'][0]:5.1f} %   calls/pitch {row['plans'][0]:6.2f}   "
              f"{row['ms_per_pitch'][0]:6.1f} ms planning/pitch", flush=True)
        out[kind] = rows
        path.write_text(json.dumps(out, indent=2))
    return rows


def at_budget(rows: list[dict], budget: float) -> dict:
    """The swept setting whose planner calls per pitch are closest to ``budget`` (log scale)."""
    return min(rows, key=lambda r: abs(np.log(r["plans"][0]) - np.log(budget)))
