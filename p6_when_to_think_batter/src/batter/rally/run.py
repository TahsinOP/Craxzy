"""Rally evaluation over the process pool (one small batch of worlds per process, as for the batter)."""
from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np

from batter.eval.runner import pool

_ENVS: dict = {}


def _env(cfg: dict, B: int, opts: dict):
    from batter.rally.env import RallyEnv

    key = json.dumps([cfg, B, {k: v for k, v in opts.items() if k != "seed"}], sort_keys=True, default=str)
    env = _ENVS.get(key)
    if env is None:
        # one environment per worker: each holds two arm models with meshes, a batch and solver caches,
        # and caching one per condition in 24 workers ran a 31 GB machine out of memory
        _ENVS.clear()
        import gc

        gc.collect()
        env = RallyEnv(cfg, B, rate_hz=opts.get("rate", 1000.0), oracle=opts.get("oracle", False), nthread=1,
                       sim_override=opts.get("sim"), commit_time=opts.get("commit_time"),
                       late_corrections=opts.get("late_corrections", True), max_vz_sd=opts.get("max_vz_sd"))
        if opts.get("prm"):
            env.planner.prm = dataclasses.replace(env.planner.prm, **opts["prm"])
        _ENVS[key] = env
    return env


def _work(args):
    from batter.rally.env import run_rally

    cfg, serves, seed, opts = args
    env = _env(cfg, len(serves), opts)
    env.rng = np.random.default_rng(seed)
    out = run_rally(env, serves)
    out["hit_k"] = np.empty(len(serves), dtype=object)
    for i, log in enumerate(env.hit_log):
        out["hit_k"][i] = np.array([h["k"] for h in log], int)
    return out


def evaluate(cfg: dict, serves: np.ndarray, seed: int = 0, opts: dict | None = None, B: int = 2) -> dict:
    """Play one rally per serve; returns the concatenated per-rally outcomes."""
    opts = opts or {}
    n = len(serves)
    jobs = [(cfg, serves[i:i + B], seed * 1000 + i, opts) for i in range(0, n, B)]
    outs = list(pool().map(_work, jobs))
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def summarize(out: dict, max_hits: int) -> dict:
    n = out["n_hits"]
    reasons = {r: float(np.mean(out["reason"] == r)) for r in sorted(set(out["reason"]))}
    return {"returns_mean": float(n.mean()), "returns_median": float(np.median(n)), "full": float(np.mean(n >= max_hits)),
            "reasons": reasons, "plans_per_return": float(out["n_plans"].sum() / max(n.sum(), 1)),
            "ms_per_call": float(1e3 * out["plan_wall"].sum() / max(out["n_plans"].sum(), 1))}


def with_overrides(cfg: dict, **sections) -> dict:
    """A copy of cfg with nested keys replaced: with_overrides(cfg, rally={"land_x": [...]})."""
    c = copy.deepcopy(cfg)
    for sec, vals in sections.items():
        c[sec] = {**c[sec], **vals}
    return c
