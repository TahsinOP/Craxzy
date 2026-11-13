"""Planners built on the swing parameterisation u (see learn/wm.py).

* ``SwingDecoder``        u -> joint-space swing (IK + strike velocity + swing QP).  Shared.
* ``DataPlanner``         the analytic planner's strike on the plane x = 0.62, perturbed, executed
                          through the decoder.  It logs (observation, u) and is how batting data
                          is collected.
* ``WorldModelPlanner``   MPPI over u in the latent space of the learned world model.

All three have the environment's planner interface (``plan_ctx``) and time themselves.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from batter.control.arm import NOMINAL_STRIKE
from batter.control.planner import SwingPlanner, planner_params
from batter.control.traj import Plan, SwingQP
from batter.learn import wm


class SwingDecoder:
    def __init__(self, arm, offset: float):
        self.arm, self.offset = arm, offset
        self.qp = SwingQP(arm.h, arm.q_min, arm.q_max)

    def decode(self, u: np.ndarray, k_now: int, q0: np.ndarray, v0: np.ndarray, prev: Plan | None) -> Plan | None:
        """u = (time to impact, y, z, azimuth, elevation, bat speed), physical units."""
        steps = float(u[0]) / self.arm.h
        if steps < 2.0:
            return None
        n = wm.normal_from_angles(u[3], u[4])
        pos = np.array([wm.STRIKE_X, u[1], u[2]]) - self.offset * n
        warm = prev.info.get("q_imp", NOMINAL_STRIKE) if prev is not None else NOMINAL_STRIKE
        q_imp, ep, er = self.arm.ik(pos, n, warm, NOMINAL_STRIKE)
        v_full = self.arm.strike_velocity(q_imp, n, float(u[5]))
        scale = min(1.0, 1.0 / max(float(np.max(np.abs(v_full) / self.qp.v_max)), 1e-9))
        plan, miss_q, miss_v, _ = self.qp.swing(q0, v0, q_imp, scale * v_full, steps, k0=k_now)
        plan.info = {"u": np.array(u, float), "n": n, "speed": float(u[5]), "scale": scale, "q_imp": q_imp,
                     "k_impact": k_now + steps, "miss_q": float(np.abs(miss_q).max()), "ik_err": ep}
        return plan


def obs_of(ctx: dict) -> np.ndarray:
    env, i = ctx["env"], ctx["world"]
    return wm.observation(env.fix_p[i], env.fix_k[i], env.k, env.h, ctx["q0"], ctx["v0"], ctx["target"])


class DataPlanner:
    """Analytic strike + exploration noise, through the decoder; logs what it did."""

    SIGMA = np.array([0.015, 0.04, 0.04, 0.10, 0.10, 0.30])          # s, m, m, rad, rad, m/s
    SCALES = np.array([0.0, 0.5, 1.0, 2.0])
    SCALE_P = np.array([0.3, 0.3, 0.25, 0.15])

    def __init__(self, env, seed: int = 0):
        prm = planner_params(env.cfg, strike_x=(wm.STRIKE_X,))
        self.analytic = SwingPlanner(env.arm, env.fm, env.cfg["bat"]["thickness"], prm)
        self.decoder = SwingDecoder(env.arm, self.analytic.offset)
        self.rng = np.random.default_rng(seed)
        self.wall, self.calls, self.times = 0.0, 0, []
        self.log: dict[int, tuple[int, np.ndarray]] = {}             # world -> (step, u) of its latest call

    def plan_ctx(self, ctx: dict) -> Plan | None:
        t0 = time.perf_counter()
        strikes = self.analytic.strikes(ctx["traj"], ctx["target"])
        plan = None
        if strikes:
            prev_u = ctx["prev"].info.get("cand") if ctx["prev"] is not None else None
            st = next((s for s in strikes if s["cand"] == prev_u), strikes[int(self.rng.integers(len(strikes)))])
            az, el = wm.angles_from_normal(st["n"])
            u = np.array([st["steps"] * ctx["env"].h, st["p_b"][1], st["p_b"][2], az, el, st["speed"]])
            u = u + self.rng.choice(self.SCALES, p=self.SCALE_P) * self.SIGMA * self.rng.normal(size=6)
            u = np.clip(u, wm.U_LO, wm.U_HI)
            plan = self.decoder.decode(u, ctx["k"], ctx["q0"], ctx["v0"], ctx["prev"])
            if plan is not None:
                plan.info["cand"] = st["cand"]
                self.log[ctx["world"]] = (ctx["env"].k, u)
        dt = time.perf_counter() - t0
        self.wall += dt
        self.calls += 1
        self.times.append(dt)
        return plan


class WorldModelPlanner:
    """MPPI in the latent space: sample swings around the prior, score them with the outcome model
    along a short latent rollout, re-fit the sampling distribution to the best ones."""

    def __init__(self, model: wm.WorldModel, env, samples: int = 192, elites: int = 24, iters: int = 3, horizon: int = 2,
                 sigma0: float = 0.15, temperature: float = 0.05, seed: int = 0):
        self.model = model.eval()
        self.decoder = SwingDecoder(env.arm, env.fm.radius + env.cfg["bat"]["thickness"] / 2.0)
        self.samples, self.elites, self.iters, self.horizon = samples, elites, iters, horizon
        self.sigma0, self.temperature = sigma0, temperature
        self.gen = torch.Generator().manual_seed(seed)
        self.wall, self.calls, self.times = 0.0, 0, []

    @staticmethod
    def load(path: str, env, **kw) -> "WorldModelPlanner":
        ck = torch.load(path, map_location="cpu")
        model = wm.WorldModel(sigma_target=ck.get("sigma_target", 1.5))
        model.load_state_dict(ck["state"])
        return WorldModelPlanner(model, env, **kw)

    @torch.no_grad()
    def optimise(self, o: np.ndarray) -> np.ndarray:
        """Normalised swing u maximising the predicted hit quality for observation o."""
        model = self.model
        z = model.encode(torch.from_numpy(o)[None])
        mean = model.policy(z)[0]
        std = torch.full((wm.N_ACT,), self.sigma0)
        for _ in range(self.iters):
            u = (mean + std * torch.randn(self.samples, wm.N_ACT, generator=self.gen)).clamp(-1.0, 1.0)
            u[0] = mean
            zk, uk = z.expand(self.samples, -1), u
            score = model.quality(zk, uk)
            for _ in range(self.horizon):                     # the swing is held: every latent step predicts the same outcome
                zk, uk = model.step(zk, uk)
                score = score + model.quality(zk, uk)
            score = score / (self.horizon + 1)
            top = torch.topk(score, self.elites)
            w = torch.softmax((top.values - top.values.max()) / self.temperature, 0)
            mean = (w[:, None] * u[top.indices]).sum(0)
            std = torch.sqrt((w[:, None] * (u[top.indices] - mean) ** 2).sum(0)).clamp(min=0.01)
        return mean.numpy()

    def plan_ctx(self, ctx: dict) -> Plan | None:
        t0 = time.perf_counter()
        u = wm.denormalize_u(self.optimise(obs_of(ctx)))
        plan = self.decoder.decode(u, ctx["k"], ctx["q0"], ctx["v0"], ctx["prev"])
        dt = time.perf_counter() - t0
        self.wall += dt
        self.calls += 1
        self.times.append(dt)
        return plan


class RandomTrigger:
    """Replan with a per-world probability per step, drawn from ``ps`` at reset (data collection:
    the last call of a pitch, from which the swing is held, lands anywhere in the flight)."""

    def __init__(self, ps, rng: np.random.Generator):
        self.ps, self.rng = np.atleast_1d(ps), rng

    def reset(self, B: int) -> None:
        self.p = self.rng.choice(self.ps, B)

    def __call__(self, obs: dict) -> np.ndarray:
        return obs["active"] & obs["has_plan"] & (self.rng.random(len(obs["active"])) < self.p)


def collect(env, P, seed: int, p_replan=(0.0, 0.02, 0.05, 0.1, 0.2), horizon: int = 2, keep: float = 0.5) -> dict:
    """One batch of batting data: every step of every pitch after its last planner call is a sample.

    Consecutive steps of a pitch are nearly the same sample, so only a fraction ``keep`` of them is
    stored.  Returns arrays obs (N, horizon + 1, N_OBS), mask (N, horizon + 1), u (N, 6) normalised,
    hit, off, quality.
    """
    rng = np.random.default_rng(seed)
    planner = DataPlanner(env, seed)
    env.planner = planner
    trig = RandomTrigger(p_replan, rng)
    obs = env.reset(P)
    planner.log.clear()
    trig.reset(env.B)
    O, K, done = [], [], False
    while not done:
        rp = trig(obs)
        obs, done = env.step(rp)
        q = np.empty((env.B, 7))
        qd = np.empty((env.B, 7))
        for i in range(env.B):
            q[i], qd[i], _ = env._reference(i, env.k)
        O.append(wm.observation(env.fix_p, env.fix_k, env.k, env.h, q, qd, P.target))
        K.append(env.k)
    out = env.outcome()
    O = np.stack(O)                                               # (steps, B, N_OBS); O[j] is the observation at step K[j]
    K = np.array(K)
    off = np.nan_to_num(out["landing"] - P.target)
    rows = {k: [] for k in ("obs", "mask", "u", "hit", "off", "quality")}
    D = wm.DT_LATENT
    for i, (k_call, u) in planner.log.items():
        k_imp = k_call + u[0] / env.h                              # the swing's impact step
        k_end = int(min(out["hit_k"][i] if out["hit"][i] else k_imp, k_imp)) - 3
        for k in range(k_call + 1, k_end):
            j = k - K[0]
            if j < 0 or j >= len(O) or rng.random() > keep:
                continue
            seq, m = np.zeros((horizon + 1, wm.N_OBS), np.float32), np.zeros(horizon + 1, bool)
            for hstep in range(horizon + 1):
                jj = j + hstep * D
                if jj < len(O) and K[jj] < k_end:
                    seq[hstep], m[hstep] = O[jj, i], True
            uk = u.copy()
            uk[0] = (k_imp - k) * env.h                            # same swing, less time to impact
            rows["obs"].append(seq), rows["mask"].append(m), rows["u"].append(wm.normalize_u(uk))
            rows["hit"].append(out["hit"][i]), rows["off"].append(off[i]), rows["quality"].append(out["quality"][i])
    return {k: np.array(v) for k, v in rows.items()}
