"""Evaluation across processes.

The planner is Python and single-threaded, so a batch of worlds is not enough to use the
machine: pitches are split into chunks, each chunk runs in its own process with its own
small ``mujoco.batch`` (one simulation thread).  Wall-clock per planner call is measured
inside the worker, around the planner only.

A trigger is described by a picklable spec ``(kind, value)``:
    ("periodic", rate_hz) | ("innovation", gate) | ("variance", delta) | ("drift", delta)
    | ("learned", (state_dict_path, lam))
"""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_POOL: ProcessPoolExecutor | None = None


def pool(workers: int | None = None) -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        workers = workers or max(1, min(24, (os.cpu_count() or 4) - 2))
        _POOL = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    return _POOL


def make_trigger(spec, seed: int = 0):
    from batter.control import triggers as T

    kind, value = spec
    if kind == "periodic":
        return T.Periodic(value)
    if kind == "innovation":
        return T.Innovation(value)
    if kind == "variance":
        return T.Variance(value)
    if kind == "drift":
        return T.Drift(value)
    if kind == "learned":
        import torch

        from batter.learn.ppo import TriggerPolicy

        path, lam = value
        policy = TriggerPolicy()
        policy.load_state_dict(torch.load(path, map_location="cpu")["state"])
        return T.Learned(policy, lam, np.random.default_rng(seed), greedy=False)
    raise ValueError(kind)


_ENVS: dict = {}


def cached_env(cfg: dict, B: int, seed: int, env_kw: dict | None = None):
    """One environment per (config, size, options) and process, re-seeded on every use."""
    import json

    key = json.dumps([cfg, B, env_kw], sort_keys=True, default=str)
    env = _ENVS.get(key)
    if env is None:
        if len(_ENVS) > 6:
            _ENVS.clear()
        env = _ENVS[key] = make_env(cfg, B, seed, env_kw)
    env.rng = np.random.default_rng(seed)
    return env


def make_env(cfg: dict, B: int, seed: int, env_kw: dict | None = None):
    """BatEnv from picklable keyword arguments: env (EnvParams fields), sim (overrides), planner
    ("analytic" or a world-model checkpoint path), ekf (EKFParams fields)."""
    from batter.envs.cage import BatEnv, EnvParams
    from batter.estim.ekf import EKFParams

    kw = env_kw or {}
    env = BatEnv(cfg, B, EnvParams(**kw.get("env", {})), ekf_params=EKFParams(**kw.get("ekf", {})), seed=seed, nthread=1,
                 sim_override=kw.get("sim"))
    if kw.get("planner", "analytic") != "analytic":
        from batter.learn.wm_planner import WorldModelPlanner

        env.planner = WorldModelPlanner.load(kw["planner"], env, **kw.get("planner_kw", {}))
    return env


def _work(args):
    import torch

    torch.set_num_threads(1)
    from batter.envs.cage import run_episode

    cfg, spec, P, seed, env_kw = args
    env = cached_env(cfg, len(P), seed, env_kw)
    out = run_episode(env, P, make_trigger(spec, seed))
    steps = np.empty(len(P), dtype=object)             # ragged: one array of planner-call steps per pitch
    for i, st in enumerate(env.plan_steps):
        steps[i] = np.array(st, int)
    out["plan_steps"] = steps
    out["s_step"], out["s_kind"] = P.s_step.copy(), P.s_kind.copy()
    return out


def evaluate(cfg: dict, spec, P, seed: int = 0, B: int | None = None, env_kw: dict | None = None,
             workers: int | None = None) -> dict:
    """Run pitches P under trigger ``spec``; returns the concatenated per-pitch outcome arrays."""
    n = len(P)
    if B is None:                                   # spread over the pool, at most 16 worlds per process
        B = int(np.clip(np.ceil(n / pool(workers)._max_workers), 4, 16))
    jobs = [(cfg, spec, P.take(np.arange(i, min(i + B, n))), seed * 1000 + i, env_kw) for i in range(0, n, B)]
    outs = list(pool(workers).map(_work, jobs))
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def _collect(args):
    import torch

    torch.set_num_threads(1)
    from batter.learn import ppo

    cfg, state, P, seed, env_kw = args
    env = cached_env(cfg, len(P), seed, env_kw)
    policy = ppo.TriggerPolicy()
    policy.load_state_dict(state)
    return ppo.collect(env, P, policy, np.random.default_rng(seed))


def collect_parallel(cfg: dict, policy, P, seed: int, B: int = 8, env_kw: dict | None = None):
    """PPO rollouts of the trigger policy over the process pool; returns the merged rollout."""
    from batter.learn import ppo

    state = {k: v.clone() for k, v in policy.state_dict().items()}
    n = len(P)
    jobs = [(cfg, state, P.take(np.arange(i, min(i + B, n))), seed * 1000 + i, env_kw) for i in range(0, n, B)]
    return ppo.merge(list(pool().map(_collect, jobs)))


def _wm_collect(args):
    import torch

    torch.set_num_threads(1)
    from batter.learn import wm_planner

    cfg, P, seed = args
    env = cached_env(cfg, len(P), seed, {"tag": "wm_data"})
    return wm_planner.collect(env, P, seed)


def wm_collect_parallel(cfg: dict, P, seed: int, B: int = 16) -> dict:
    """Batting data for the world model over the process pool."""
    n = len(P)
    jobs = [(cfg, P.take(np.arange(i, min(i + B, n))), seed * 100_000 + i) for i in range(0, n, B)]
    outs = [o for o in pool().map(_wm_collect, jobs) if len(o["u"])]
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def _wm_train(args):
    import torch

    torch.set_num_threads(4)
    from batter.learn import wm

    data_path, frac, tc, sigma, path = args
    data = dict(np.load(data_path))
    if frac < 1.0:                                   # the first ``frac`` of the pitches (samples are stored in pitch order)
        n = int(frac * len(data["u"]))
        data = {k: v[:n] for k, v in data.items()}
    model, log = wm.train(data, tc, sigma_target=sigma, verbose=False)
    torch.save({"state": model.state_dict(), "log": log, "sigma_target": sigma, "n_samples": int(len(data["u"]))}, path)
    return log


def wm_train_parallel(jobs: list) -> list:
    """Train several world models at once; jobs = [(data path, fraction, TrainConfig, sigma_target, out path), ...]."""
    return list(pool().map(_wm_train, jobs))


def time_planner(cfg: dict, spec, P, env_kw: dict | None = None, seed: int = 0) -> dict:
    """Wall-clock per planner call measured in this process, with nothing else running."""
    from batter.envs.cage import run_episode

    env = make_env(cfg, len(P), seed, env_kw)
    run_episode(env, P, make_trigger(spec, seed))
    env.planner.times.clear()
    out = run_episode(env, P, make_trigger(spec, seed))          # second pass: caches are warm
    t = 1e3 * np.array(env.planner.times)
    return {"calls": int(t.size), "ms_median": float(np.median(t)), "ms_mean": float(t.mean()), "ms_p90": float(np.percentile(t, 90)),
            "ms_max": float(t.max()), "hit": float(out["hit"].mean())}


def summarize(out: dict) -> dict:
    """Scalar metrics of one evaluation."""
    hit = out["hit"]
    calls = out["n_plans"].sum()
    return {"hit": float(hit.mean()), "on_target": float(out["on_target"].mean()), "quality": float(out["quality"].mean()),
            "dist_med": float(np.median(out["dist"][hit])) if hit.any() else float("nan"),
            "plans": float(out["n_plans"].mean()), "ms_per_call": float(1e3 * out["plan_wall"].sum() / max(calls, 1)),
            "ms_per_pitch": float(1e3 * out["plan_wall"].mean())}


def mean_std(rows: list[dict]) -> dict:
    """[mean, std] over seeds for every scalar metric."""
    return {k: [float(np.nanmean([r[k] for r in rows])), float(np.nanstd([r[k] for r in rows]))] for k in rows[0]}
