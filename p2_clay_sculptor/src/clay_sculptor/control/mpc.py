"""Sampling-based MPC over tool primitives (cross-entropy method).

One planner, several rollout models.  At each decision the planner searches the
five primitive parameters of the given tool for the action whose *predicted* next
height map is closest to the target (root-mean-square height difference), executes
it in the simulator, observes the true shape and repeats (greedy receding horizon).

Rollout models ("evaluators"), all scored in the same currency, the height map:

* ``OracleEval``   MuJoCo itself: one simulated primitive per candidate;
* ``ModelEval``    a learned model or the geometric prior;
* ``HybridEval``   an ensemble for the search, plus oracle rollouts for the top
                   candidates when the ensemble disagrees (M4).

The search budget (population x iterations) is the same for every evaluator, so
differences in the outcome come from the model, not from the search.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from clay_sculptor.control.primitives import PrimCfg, bounds, yaw_period
from clay_sculptor.envs.clay_env import ClayEnv
from clay_sculptor.envs.shape import map_rmse
from clay_sculptor.models.scene import TOOLS


@dataclass(frozen=True)
class CEMCfg:
    pop: int = 32
    iters: int = 3
    elite: int = 6
    guided: float = 0.5        # share of the first population drawn from the guided proposal
    min_std: float = 0.03      # floor on the sampling std, as a fraction of each parameter's range
    stop_tol: float = 2e-5     # stop when the best predicted improvement is below this [m RMSE]


def wrap_action(a: np.ndarray, tool: str, prim: PrimCfg) -> np.ndarray:
    """Clip to the bounds; the yaw wraps around its symmetry period instead of clipping."""
    lo, hi = bounds(tool, prim)
    a = np.array(a, dtype=float)
    per = yaw_period(tool)
    a[..., 2] = lo[2] + (a[..., 2] - lo[2]) % per
    return np.clip(a, lo, hi)


def guided_proposal(rng: np.random.Generator, h_cur: np.ndarray, h_tgt: np.ndarray, tool: str, n: int, env: ClayEnv) -> np.ndarray:
    """Actions centred where the clay is too high, pressed to the target height there.

    Position: a pixel drawn with probability proportional to the squared excess
    height.  Depth: the one that puts the tool bottom at the target height of that
    pixel, plus 1 mm noise.  Yaw and travel: uniform."""
    lo, hi = bounds(tool, env.prim)
    a = rng.uniform(lo, hi, size=(n, 5))
    excess = np.clip(h_cur - h_tgt, 0.0, None) ** 2
    if excess.sum() <= 0:
        return a
    p = (excess / excess.sum()).ravel()
    pix = rng.choice(p.size, size=n, p=p)
    i, j = np.unravel_index(pix, h_cur.shape)
    ax = env.geom.map.axis
    a[:, 0] = ax[i] + rng.normal(0, 0.004, n)
    a[:, 1] = ax[j] + rng.normal(0, 0.004, n)
    a[:, 3] = env.clay.top - h_tgt[i, j] + rng.normal(0, 0.001, n)
    return wrap_action(a, tool, env.prim)


def _circular_stats(x: np.ndarray, period: float) -> tuple[float, float]:
    ang = 2 * np.pi * x / period
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    r = min(np.hypot(c, s), 1 - 1e-9)
    return np.arctan2(s, c) * period / (2 * np.pi), np.sqrt(-2 * np.log(r)) * period / (2 * np.pi)


def cem(evaluate, tool: str, rng: np.random.Generator, cfg: CEMCfg, h_cur: np.ndarray, h_tgt: np.ndarray, env: ClayEnv) -> dict:
    """Minimise ``evaluate(actions (P, 5)) -> costs (P,)``.  Returns every evaluated
    action and its cost, sorted by cost."""
    lo, hi = bounds(tool, env.prim)
    span = np.where(hi > lo, hi - lo, 1.0)
    n_g = int(round(cfg.guided * cfg.pop))
    pop = np.vstack([guided_proposal(rng, h_cur, h_tgt, tool, n_g, env), rng.uniform(lo, hi, size=(cfg.pop - n_g, 5))])
    acts, costs = [], []
    for it in range(cfg.iters):
        c = np.asarray(evaluate(pop), dtype=float)
        acts.append(pop)
        costs.append(c)
        if it == cfg.iters - 1:
            break
        all_a, all_c = np.vstack(acts), np.concatenate(costs)
        elite = all_a[np.argsort(all_c)[:cfg.elite]]
        mean, std = elite.mean(axis=0), elite.std(axis=0)
        mean[2], std[2] = _circular_stats(elite[:, 2], yaw_period(tool))
        std = np.maximum(std, cfg.min_std * span) * (hi > lo)
        pop = wrap_action(mean + std * rng.standard_normal((cfg.pop, 5)), tool, env.prim)
    all_a, all_c = np.vstack(acts), np.concatenate(costs)
    order = np.argsort(all_c)
    return dict(actions=all_a[order], costs=all_c[order])


# ---- evaluators -------------------------------------------------------------------
class OracleEval:
    """MuJoCo as the rollout model."""

    name = "oracle"

    def __init__(self, env: ClayEnv):
        self.env = env

    def reset(self, state: np.ndarray, tool: str, h_tgt: np.ndarray) -> None:
        self.state, self.tool, self.h_tgt = state, tool, h_tgt

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        nxt, ok = self.env.rollout(self.state, self.tool, actions)
        c = map_rmse(self.env.geom.heightmap(self.env.nodes(nxt)), self.h_tgt)
        return np.where(ok, c, np.inf)

    def choose(self, result: dict, rng) -> tuple[np.ndarray, float, dict]:
        return result["actions"][0], float(result["costs"][0]), {}


class ModelEval:
    """A learned model, an ensemble (mean of the members' maps) or the prior."""

    def __init__(self, model, env: ClayEnv, name: str | None = None):
        self.model, self.env = model, env
        self.members = getattr(model, "members", [model])
        self.name = name or model.kind

    def reset(self, state: np.ndarray, tool: str, h_tgt: np.ndarray) -> None:
        nodes = torch.from_numpy(self.env.nodes(state).astype(np.float32))[None]
        self.x = self.model.encode(nodes)
        self.tool = tool
        self.h_tgt = torch.from_numpy(h_tgt.astype(np.float32))

    @torch.no_grad()
    def maps(self, actions: np.ndarray) -> torch.Tensor:
        """Predicted height maps of every member: ``(M, P, G, G)``."""
        a = torch.from_numpy(np.asarray(actions, dtype=np.float32))
        t = torch.full((len(a),), TOOLS.index(self.tool), dtype=torch.long)
        x = self.x.expand(len(a), *self.x.shape[1:])
        return torch.stack([m.heightmap(m(x, t, a)) for m in self.members])

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        return map_rmse(self.maps(actions).mean(0), self.h_tgt).numpy()

    @torch.no_grad()
    def disagreement(self, actions: np.ndarray) -> np.ndarray:
        """Ensemble spread per candidate [m]: RMS over pixels of the std across members."""
        return self.maps(actions).std(0).pow(2).mean(dim=(-1, -2)).sqrt().numpy()

    def choose(self, result: dict, rng) -> tuple[np.ndarray, float, dict]:
        return result["actions"][0], float(result["costs"][0]), {}


class HybridEval(ModelEval):
    """Search with the ensemble; when triggered, re-rank the ``top_m`` candidates with
    oracle rollouts and take the truly best one.

    ``trigger``: ``"disagreement"`` (fire when the ensemble spread of the chosen
    action's height map exceeds ``tau``), ``"misrank"`` (fire when the probability that
    another top candidate is truly better exceeds ``tau``, see :meth:`signals`), ``"random"`` (fire with probability ``p``), ``"always"`` or
    ``"never"``.  Each firing costs ``top_m`` simulated primitives."""

    def __init__(self, model, env: ClayEnv, trigger: str = "disagreement", tau: float = 0.0, p: float = 0.0,
                 top_m: int = 8, name: str | None = None):
        super().__init__(model, env, name or f"hybrid-{trigger}")
        self.trigger, self.tau, self.p, self.top_m = trigger, tau, p, top_m
        self.oracle = OracleEval(env)

    def reset(self, state: np.ndarray, tool: str, h_tgt: np.ndarray) -> None:
        super().reset(state, tool, h_tgt)
        self.oracle.reset(state, tool, h_tgt)

    @torch.no_grad()
    def signals(self, actions: np.ndarray) -> dict:
        """Three measures of doubt about executing ``actions[0]`` (the candidate the
        ensemble mean ranks first) instead of one of ``actions[1:]``.

        Treat the M members as samples of the true dynamics, with costs c_m(a_j).
        ``spread``: RMS over pixels of the members' std of the map of a_1 [m]
        (doubt about the shape).  ``regret``: mean over members of
        c_m(a_1) - min_j c_m(a_j) [m] (>= 0, zero iff every member ranks a_1 first).
        ``misrank``: max over j of Phi(-g_j / s_j), with g_j the mean and s_j the
        std over members of c_m(a_j) - c_m(a_1): the probability, under a normal fit
        to the members, that a_j is truly better than a_1 (in [0, 1], 1/2 for a tie)."""
        maps = self.maps(actions)                                      # (M, K, G, G)
        c = map_rmse(maps, self.h_tgt).double()                        # (M, K)
        d = c[:, 1:] - c[:, :1]                                        # (M, K-1)
        g, sd = d.mean(0), d.std(0)
        z = torch.where(sd > 0, -g / sd.clamp_min(1e-12), torch.where(g > 0, -torch.inf, torch.where(g < 0, torch.inf, 0.0)))
        mis = float(torch.special.ndtr(z).max()) if d.shape[1] else 0.0
        return dict(spread=float(maps[:, 0].std(0).pow(2).mean().sqrt()), regret=float((c[:, 0] - c.min(dim=1).values).mean()),
                    misrank=mis)

    def choose(self, result: dict, rng) -> tuple[np.ndarray, float, dict]:
        best = result["actions"][0]
        sig = self.signals(result["actions"][:self.top_m])
        u = rng.random()                                               # drawn for every trigger: same random stream
        fire = dict(disagreement=sig["spread"] > self.tau, misrank=sig["misrank"] > self.tau, random=u < self.p, always=True,
                    never=False)[self.trigger]
        info = dict(**sig, fired=bool(fire), model_cost=float(result["costs"][0]))
        if not fire:
            return best, float(result["costs"][0]), info
        cand = result["actions"][:self.top_m]
        true = self.oracle(cand)
        k = int(np.argmin(true))
        info["true_cost_of_model_best"] = float(true[0])
        return cand[k], float(true[k]), info


# ---- closed loop ------------------------------------------------------------------
def run_episode(env: ClayEnv, target, evaluator, rng: np.random.Generator, cfg: CEMCfg = CEMCfg(),
                max_steps: int | None = None, state0: np.ndarray | None = None) -> dict:
    """Plan and execute until no improvement is predicted or ``max_steps`` is reached.

    ``target`` has ``tool`` (``None``: the planner picks the tool each step by planning
    with every tool and keeping the best prediction), ``hmap`` and ``max_steps``.  Returns the trajectory of true
    costs, the actions, the planning time and the number of oracle rollouts used for
    planning (execution steps are counted separately)."""
    state = env.state0 if state0 is None else state0
    h = env.geom.heightmap(env.nodes(state))
    cost = float(map_rmse(h, target.hmap))
    rec = dict(costs=[cost], actions=[], predicted=[], plan_time=[], plan_calls=[], info=[], states=[state])
    tools = [target.tool] if target.tool else list(TOOLS)      # tool not given: plan with each, keep the best
    rec["tools"] = []
    for _ in range(max_steps or target.max_steps):
        calls0, t0 = env.oracle_calls, time.time()
        best = None
        for tool in tools:
            evaluator.reset(state, tool, target.hmap)
            result = cem(evaluator, tool, rng, cfg, h, target.hmap, env)
            action, pred, info = evaluator.choose(result, rng)
            if best is None or pred < best[1]:
                best = (action, pred, info, tool)
        action, pred, info, tool = best
        rec["plan_time"].append(time.time() - t0)
        rec["plan_calls"].append(env.oracle_calls - calls0)
        rec["predicted"].append(pred)
        rec["info"].append(info)
        if not pred < cost - cfg.stop_tol:
            break
        state = env.step(state, tool, action)
        env.oracle_calls -= 1                      # executing an action is not a planning call
        h = env.geom.heightmap(env.nodes(state))
        cost = float(map_rmse(h, target.hmap))
        rec["actions"].append(action.tolist())
        rec["tools"].append(tool)
        rec["costs"].append(cost)
        rec["states"].append(state)
    rec["final_state"] = state
    return rec
