"""A small latent world model of batting (TD-MPC2-style components, sized for a CPU).

What it replaces.  The analytic planner knows the ball's flight (EKF + flight model), the
impact (restitution model) and the flight to the target.  The learned planner knows none
of that.  It sees raw camera fixes and has to learn, from batting data, what a swing does.

Observation o (N_OBS numbers; nothing from the EKF):
    the last 16 camera fixes (x, y, z, age) with a validity flag     80
    the arm's reference joint position and velocity                  14
    the target on the ground (x, y)                                   2

Action u (6 numbers, the *swing*; turned into joint motion by IK + the swing QP):
    time to impact, strike point (y, z) on the plane x = 0.62, azimuth and elevation of the
    bat's face normal, bat speed along the normal.

Model (Hansen et al., TD-MPC2, reduced):
    encoder     z = h(o)                      latent state (64)
    dynamics    z' = d(z, u)                  one decision interval (DT_LATENT) ahead, with the swing u held
    outcome     hit logit, landing offset (x, y) from the target      <- the "reward" model
    prior       u = pi(z)                     policy prior that seeds the planner

Training on batting data (learn/wm_planner.py collects it):
    * outcome loss on (o_k, u_k) -> what happened, for every step k of a pitch after the last
      planner call (the swing u was held from there on; its time-to-impact component is
      shifted by the elapsed time);
    * latent consistency: d(h(o_k), u_k) should equal h(o_{k + DT}) (stop-gradient target), and
      the outcome predicted from the rolled-out latent should match too;
    * prior: regression of u on z, weighted by exp(quality / temperature) (advantage-weighted).

Planning: MPPI over u in the latent space (learn/wm_planner.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

NFIX = 16
N_OBS = NFIX * 5 + 14 + 2
N_ACT = 6
DT_LATENT = 4                       # simulator steps per latent step
STRIKE_X = 0.62

# action ranges: time to impact [s], y [m], z [m], azimuth [rad], elevation [rad], bat speed [m/s]
U_LO = np.array([0.0, -0.40, 0.45, -0.60, -0.10, -1.5])
U_HI = np.array([0.85, 0.40, 1.45, 0.60, 1.20, 2.4])


def normalize_u(u: np.ndarray) -> np.ndarray:
    return 2.0 * (u - U_LO) / (U_HI - U_LO) - 1.0


def denormalize_u(un: np.ndarray) -> np.ndarray:
    return U_LO + 0.5 * (un + 1.0) * (U_HI - U_LO)


def normal_from_angles(az, el) -> np.ndarray:
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def angles_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.arctan2(n[..., 1], n[..., 0]), np.arcsin(np.clip(n[..., 2], -1.0, 1.0))


def observation(fix_p: np.ndarray, fix_k: np.ndarray, k: int, h: float, q: np.ndarray, qd: np.ndarray,
                target: np.ndarray) -> np.ndarray:
    """Features (..., N_OBS) from the fix buffer (..., NFIX, 3), its step stamps (..., NFIX), the arm
    reference and the target."""
    valid = (fix_k >= 0).astype(float)
    age = (k - fix_k) * h * valid
    f = np.concatenate([fix_p[..., 0:1] / 6.0, fix_p[..., 1:2] / 0.3, (fix_p[..., 2:3] - 1.0) / 0.5], axis=-1) * valid[..., None]
    fx = np.concatenate([f, 5.0 * age[..., None], valid[..., None]], axis=-1).reshape(fix_p.shape[:-2] + (NFIX * 5,))
    return np.concatenate([fx, q, qd / 2.0, target / 5.0], axis=-1).astype(np.float32)


def mlp(n_in: int, n_out: int, hidden: int = 256, layers: int = 2) -> nn.Sequential:
    mods, n = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(n, hidden), nn.LayerNorm(hidden), nn.Mish()]
        n = hidden
    mods.append(nn.Linear(n, n_out))
    return nn.Sequential(*mods)


class WorldModel(nn.Module):
    def __init__(self, latent: int = 64, hidden: int = 256, sigma_target: float = 1.5):
        super().__init__()
        self.latent, self.sigma_target = latent, sigma_target
        self.encoder = mlp(N_OBS, latent, hidden)
        self.dynamics = mlp(latent + N_ACT, latent, hidden)
        self.outcome = mlp(latent + N_ACT, 3, hidden)          # hit logit, landing offset (x, y) [m / 2]
        self.prior = mlp(latent, N_ACT, hidden)
        self.register_buffer("dt_u", torch.zeros(N_ACT))        # shift of the normalised action per latent step
        self.dt_u[0] = 2.0 * DT_LATENT * 0.01 / float(U_HI[0] - U_LO[0])

    def encode(self, o: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.encoder(o))                      # bounded latent (TD-MPC2 uses SimNorm)

    def step(self, z: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One latent step with the swing held: (z', u') with u' = u with DT less time to impact."""
        return torch.tanh(self.dynamics(torch.cat([z, u], -1))), u - self.dt_u

    def predict(self, z: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.outcome(torch.cat([z, u], -1))
        return out[..., 0], 2.0 * out[..., 1:3]                 # hit logit, landing offset [m]

    def quality(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        logit, off = self.predict(z, u)
        return torch.sigmoid(logit) * torch.exp(-0.5 * (off**2).sum(-1) / self.sigma_target**2)

    def policy(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.prior(z))


@dataclass
class TrainConfig:
    steps: int = 20000
    batch: int = 512
    lr: float = 1e-3
    horizon: int = 2                 # latent steps unrolled in the consistency loss
    w_consistency: float = 2.0
    w_outcome: float = 1.0
    w_prior: float = 1.0
    temperature: float = 0.25        # advantage weighting of the prior
    seed: int = 0


def train(data: dict, cfg: TrainConfig, sigma_target: float = 1.5, log_every: int = 2000, verbose: bool = True) -> tuple[WorldModel, list]:
    """Fit the world model.  ``data``: arrays over samples

        obs (N, H + 1, N_OBS)   observations at steps k, k + DT, ..., k + H * DT (zeros where the pitch ended)
        mask (N, H + 1)         1 where that observation exists
        u (N, N_ACT)            normalised swing at step k
        hit (N,), off (N, 2), quality (N,)
    """
    torch.manual_seed(cfg.seed)
    model = WorldModel(sigma_target=sigma_target)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    obs = torch.from_numpy(data["obs"])
    mask = torch.from_numpy(data["mask"].astype(np.float32))
    u = torch.from_numpy(data["u"].astype(np.float32))
    hit = torch.from_numpy(data["hit"].astype(np.float32))
    off = torch.from_numpy(data["off"].astype(np.float32))
    wq = torch.exp(torch.from_numpy(data["quality"].astype(np.float32)) / cfg.temperature)
    wq = wq / wq.mean()
    N = len(u)
    rng = np.random.default_rng(cfg.seed)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    log = []

    def outcome_loss(z, uu, b, m):
        logit, pred = model.predict(z, uu)
        l_hit = bce(logit, hit[b])
        l_off = (nn.functional.huber_loss(pred, off[b], reduction="none", delta=1.0).sum(-1) * hit[b])
        return ((l_hit + l_off) * m).sum() / m.sum().clamp(min=1.0), l_hit, l_off

    for it in range(cfg.steps):
        b = torch.from_numpy(rng.integers(0, N, cfg.batch))
        o, m, uu = obs[b], mask[b], u[b]
        z = model.encode(o[:, 0])
        loss_o, l_hit, l_off = outcome_loss(z, uu, b, m[:, 0])
        loss_p = (((model.policy(z.detach()) - uu) ** 2).sum(-1) * wq[b] * hit[b]).mean()
        loss_c = torch.zeros(())
        zk, uk = z, uu
        for j in range(1, cfg.horizon + 1):
            zk, uk = model.step(zk, uk)
            with torch.no_grad():
                zt = model.encode(o[:, j])
            mj = m[:, j]
            loss_c = loss_c + (((zk - zt) ** 2).mean(-1) * mj).sum() / mj.sum().clamp(min=1.0)
            lo, _, _ = outcome_loss(zk, uk, b, mj)
            loss_o = loss_o + 0.5 * lo
        loss = cfg.w_outcome * loss_o + cfg.w_consistency * loss_c + cfg.w_prior * loss_p
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        if it % log_every == 0 or it == cfg.steps - 1:
            row = {"step": it, "loss": float(loss.detach()), "outcome": float(loss_o.detach()), "consistency": float(loss_c.detach()),
                   "prior": float(loss_p.detach())}
            log.append(row)
            if verbose:
                print(f"    wm step {it:6d}: loss {row['loss']:.4f}  outcome {row['outcome']:.4f}  consistency "
                      f"{row['consistency']:.5f}  prior {row['prior']:.4f}", flush=True)
    return model, log


@torch.no_grad()
def validate(model: WorldModel, data: dict) -> dict:
    """Held-out accuracy of the outcome model and of the prior."""
    o = torch.from_numpy(data["obs"][:, 0])
    u = torch.from_numpy(data["u"].astype(np.float32))
    z = model.encode(o)
    logit, off = model.predict(z, u)
    hit = data["hit"].astype(bool)
    pred_hit = (logit.numpy() > 0)
    err = np.linalg.norm(off.numpy() - data["off"], axis=1)[hit]
    q = model.quality(z, u).numpy()
    up = denormalize_u(model.policy(z).numpy())
    ut = denormalize_u(data["u"])
    good = data["quality"] > 0.8
    return {"n": int(len(u)), "hit_accuracy": float((pred_hit == hit).mean()), "hit_rate_data": float(hit.mean()),
            "landing_err_med_m": float(np.median(err)) if hit.any() else float("nan"),
            "quality_mae": float(np.abs(q - data["quality"]).mean()),
            "prior_err": [float(v) for v in np.median(np.abs(up - ut)[good], axis=0)] if good.any() else []}
