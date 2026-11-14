"""Transition datasets: random primitives applied to the clay.

An episode starts from the undeformed block and applies ``steps`` random
primitives.  Half of the episodes use one tool throughout, the other half draw a
tool per step; parameters are uniform inside ``primitives.bounds``.  Stored per
episode: the cage nodes before the first primitive and after each one
(``nodes[e, k]``, float32, clay coordinates), the tool id and the 5 parameters.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from clay_sculptor.control.primitives import TOOL_ID, bounds
from clay_sculptor.envs.clay_env import ClayEnv
from clay_sculptor.models.scene import TOOLS


def sample_actions(rng: np.random.Generator, tools: np.ndarray, prim) -> np.ndarray:
    out = np.zeros((len(tools), 5))
    for t in TOOLS:
        sel = tools == TOOL_ID[t]
        lo, hi = bounds(t, prim)
        out[sel] = rng.uniform(lo, hi, size=(int(sel.sum()), 5))
    return out


def generate(env: ClayEnv, episodes: int, steps: int, seed: int, log=print) -> dict:
    rng = np.random.default_rng(seed)
    nn = env.geom.nn
    nodes = np.zeros((episodes, steps + 1, *nn, 3), dtype=np.float32)
    tool = np.zeros((episodes, steps), dtype=np.int8)
    action = np.zeros((episodes, steps, 5), dtype=np.float32)
    ok = np.ones((episodes, steps), dtype=bool)
    t0 = time.time()
    for lo in range(0, episodes, env.nsim):
        hi = min(lo + env.nsim, episodes)
        b = hi - lo
        states = np.repeat(env.state0[None], b, axis=0)
        nodes[lo:hi, 0] = env.nodes(states)
        single = rng.random(b) < 0.5
        fixed = rng.integers(0, len(TOOLS), b)
        for k in range(steps):
            tk = np.where(single, fixed, rng.integers(0, len(TOOLS), b))
            ak = sample_actions(rng, tk, env.prim)
            for t in TOOLS:
                ids = np.flatnonzero(tk == TOOL_ID[t])
                if len(ids):
                    nxt, good = env.rollout(states[ids], t, ak[ids])
                    states[ids] = nxt
                    ok[lo + ids, k] = good
            tool[lo:hi, k], action[lo:hi, k] = tk, ak
            nodes[lo:hi, k + 1] = env.nodes(states)
        ok[lo:hi] = np.logical_and.accumulate(ok[lo:hi], axis=1)      # a failed step invalidates the rest
        done = hi * steps
        log(f"  {hi}/{episodes} episodes, {done} transitions, {time.time() - t0:.0f} s "
            f"({(time.time() - t0) / done:.3f} s per transition), failed steps {int((~ok[:hi]).sum())}")
    return dict(nodes=nodes, tool=tool, action=action, ok=ok)


def save(path: Path, data: dict, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data, **{f"meta_{k}": v for k, v in meta.items()})


def load(path: Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
