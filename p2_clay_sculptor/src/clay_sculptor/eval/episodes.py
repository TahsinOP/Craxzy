"""Bookkeeping shared by the MPC scripts (M1, M3, M4)."""
from __future__ import annotations

import numpy as np

from clay_sculptor.control.mpc import CEMCfg, run_episode
from clay_sculptor.envs.clay_env import ClayEnv
from clay_sculptor.envs.targets import Target, shape_errors


def summarize(env: ClayEnv, target: Target, rec: dict) -> dict:
    """JSON-ready summary of one closed-loop episode (lengths in mm, times in s)."""
    err = shape_errors(env, rec["final_state"], target)
    return dict(
        target=target.name,
        costs_mm=[1e3 * c for c in rec["costs"]],
        predicted_mm=[1e3 * c for c in rec["predicted"]],
        actions=rec["actions"], tools=rec.get("tools", []),
        n_actions=len(rec["actions"]),
        plan_calls=int(np.sum(rec["plan_calls"])), plan_calls_per_decision=[int(c) for c in rec["plan_calls"]],
        plan_time_s=float(np.sum(rec["plan_time"])), plan_time_per_decision_s=float(np.mean(rec["plan_time"])),
        info=rec["info"],
        initial_mm=1e3 * target.initial_cost,
        remaining=err["map_rmse_mm"] / (1e3 * target.initial_cost),
        **err,
    )


def episode(env: ClayEnv, target: Target, evaluator, seed: int, cfg: CEMCfg = CEMCfg()) -> tuple[dict, np.ndarray]:
    """Run one episode; returns (summary, final cage nodes as float32)."""
    rec = run_episode(env, target, evaluator, np.random.default_rng(seed), cfg)
    return summarize(env, target, rec), env.nodes(rec["final_state"]).astype(np.float32)


def mean_std(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std())
