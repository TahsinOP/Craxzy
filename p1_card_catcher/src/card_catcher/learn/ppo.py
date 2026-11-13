"""PPO for the learned replanning trigger (CPU; worlds stepped with ``mujoco.batch``).

Decision process.  A decision is taken at every simulator step at which a plan
exists and replanning is still allowed (the object is in flight and more than
``min_tau`` from the catch).  Action a in {0: keep the plan, 1: replan}.

Reward (TASK.md):  R = catch - lam * (number of replans),
i.e. r_t = -lam * a_t at each decision and +1 at the last decision of a throw
that ends in a catch.  gamma = 1 (finite episodes).

One policy serves every price lam: log10(lam) is an input and lam is drawn
log-uniformly per episode during training, so a single training run yields the
whole catch-rate / planner-calls trade-off curve.

Observation (all computed by the predictor, none of it privileged):
  tau                    predicted time to the catch [s]
  since                  time since the last plan [s]
  drift                  |c_hat(t) - c_hat(t_plan)|, plus the timing change as a distance [cm]
  sd_now, sd_plan        sqrt(tr P(t)), sqrt(tr P(t_plan))   [cm]
  sd_gain                sqrt(max(tr P(t_plan) - tr P(t), 0)) [cm]  (the variance trigger's statistic)
  log10(lam)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

N_OBS = 7
LAM_RANGE = (1e-3, 1e-1)


def features(obs: dict, lam: np.ndarray) -> np.ndarray:
    d = np.sqrt(obs["drift"] ** 2 + (obs["speed"] * obs["time_drift"]) ** 2)
    f = np.stack([
        obs["tau"] / 0.5,
        obs["since"] / 0.2,
        np.log1p(100.0 * d),
        np.log1p(100.0 * np.sqrt(np.maximum(obs["tr_now"], 0.0))),
        np.log1p(100.0 * np.sqrt(np.maximum(obs["tr_plan"], 0.0))),
        np.log1p(100.0 * np.sqrt(np.maximum(obs["tr_plan"] - obs["tr_now"], 0.0))),
        (np.log10(lam) + 2.0),
    ], axis=1)
    return f.astype(np.float32)


class TriggerPolicy(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(N_OBS, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.vf = nn.Sequential(nn.Linear(N_OBS, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.constant_(self.pi[-1].bias, -1.5)     # start by replanning rarely (p ~ 0.18)
        nn.init.normal_(self.pi[-1].weight, std=0.01)

    def forward(self, x: torch.Tensor):
        return self.pi(x).squeeze(-1), self.vf(x).squeeze(-1)


def act(policy: TriggerPolicy, obs: dict, lam: np.ndarray, rng: np.random.Generator, greedy: bool = False):
    """Sample actions for every world.  Returns (action bool (B,), features (B, N_OBS), logit (B,))."""
    f = features(obs, lam)
    with torch.no_grad():
        logit, _ = policy(torch.from_numpy(f))
    logit = logit.numpy()
    p = 1.0 / (1.0 + np.exp(-logit))
    a = (p > 0.5) if greedy else (rng.random(len(p)) < p)
    return a, f, logit


@dataclass
class Rollout:
    feats: np.ndarray     # (T, N_OBS) decision features, all worlds concatenated
    actions: np.ndarray   # (T,)
    logp: np.ndarray      # (T,) log-probability of the action under the behaviour policy
    adv: np.ndarray       # (T,) GAE advantages
    ret: np.ndarray       # (T,) returns (value targets)
    stats: dict


def collect(env, D, idx: np.ndarray, policy: TriggerPolicy, rng: np.random.Generator, lam: np.ndarray | None = None,
            gae_lambda: float = 0.95) -> Rollout:
    """Run one batch of throws under the stochastic policy and build PPO targets."""
    B = env.B
    lam = np.exp(rng.uniform(np.log(LAM_RANGE[0]), np.log(LAM_RANGE[1]), B)) if lam is None else lam
    obs = env.reset(D, idx)
    F, A, LP, V, W = [], [], [], [], []       # per step: features, actions, logp, values, decision mask
    done = False
    while not done:
        dec = obs["active"] & obs["has_plan"]
        a, f, logit = act(policy, obs, lam, rng)
        with torch.no_grad():
            v = policy(torch.from_numpy(f))[1].numpy()
        logp = np.where(a, -np.log1p(np.exp(-logit)), -np.log1p(np.exp(logit)))
        F.append(f), A.append(a), LP.append(logp), V.append(v), W.append(dec)
        obs, done = env.step(a & dec)
    out = env.outcome()
    F, A, LP, V, W = np.stack(F), np.stack(A), np.stack(LP), np.stack(V), np.stack(W)     # (K, B, ...)
    feats, acts, logps, advs, rets = [], [], [], [], []
    for i in range(B):
        t = np.flatnonzero(W[:, i])
        if t.size == 0:
            continue
        r = -lam[i] * A[t, i].astype(float)
        r[-1] += float(out["caught"][i])
        v = np.append(V[t, i], 0.0)
        delta = r + v[1:] - v[:-1]
        adv = np.zeros(t.size)
        g = 0.0
        for k in range(t.size - 1, -1, -1):
            g = delta[k] + gae_lambda * g
            adv[k] = g
        feats.append(F[t, i]), acts.append(A[t, i]), logps.append(LP[t, i]), advs.append(adv), rets.append(adv + v[:-1])
    stats = {"catch": float(out["caught"].mean()), "plans": float(out["n_plans"].mean()),
             "return": float(np.mean(out["caught"] - lam * np.maximum(out["n_plans"] - 1, 0)))}
    return Rollout(np.concatenate(feats), np.concatenate(acts), np.concatenate(logps), np.concatenate(advs),
                   np.concatenate(rets), stats)


def update(policy: TriggerPolicy, opt: torch.optim.Optimizer, ro: Rollout, rng: np.random.Generator, epochs: int = 4,
           minibatch: int = 1024, clip: float = 0.2, vf_coef: float = 0.5, ent_coef: float = 0.01) -> dict:
    x = torch.from_numpy(ro.feats)
    a = torch.from_numpy(ro.actions.astype(np.float32))
    logp_old = torch.from_numpy(ro.logp.astype(np.float32))
    adv = torch.from_numpy(((ro.adv - ro.adv.mean()) / (ro.adv.std() + 1e-8)).astype(np.float32))
    ret = torch.from_numpy(ro.ret.astype(np.float32))
    n = len(x)
    info = {}
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, minibatch):
            b = torch.from_numpy(perm[i:i + minibatch])
            logit, v = policy(x[b])
            dist = torch.distributions.Bernoulli(logits=logit)
            logp = dist.log_prob(a[b])
            ratio = torch.exp(logp - logp_old[b])
            pg = -torch.min(ratio * adv[b], torch.clamp(ratio, 1 - clip, 1 + clip) * adv[b]).mean()
            vl = ((v - ret[b]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pg + vf_coef * vl - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            info = {"pg": float(pg.detach()), "vf": float(vl.detach()), "entropy": float(ent.detach()),
                    "p_replan": float(torch.sigmoid(logit).mean().detach())}
    return info
