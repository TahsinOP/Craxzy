"""Online catch-point predictors, batched over worlds.

All predictors share one interface:

    step(z, valid)   absorb one frame of position fixes (after each simulator step)
    forecast()       catch-plane crossing (time, point, velocity) and its covariance
    reset(ids)

* ``PhysicsPredictor``: the M1 EKF with the drag-ballistic model.
* ``HybridPredictor``: M2.  A physics EKF produces features, a GRU turns them
  into the residual acceleration (mean, std) for the next ``horizon`` steps, and
  a second EKF -- physics + residual mean, with kappa times the residual variance
  as its process noise and the drag fixed at the fitted value -- provides the
  state the forecast starts from.
* ``OraclePredictor``: returns the true crossing (to separate control from
  estimation in M3).
"""
from __future__ import annotations

import numpy as np
import torch

from card_catcher.config import Plane
from card_catcher.estim.ekf import BatchEKF, EKFParams, forecast
from card_catcher.learn import data as ld
from card_catcher.learn.models import SeqPredictor

NMAX = 200


class PhysicsPredictor:
    def __init__(self, B: int, h: float, prm: EKFParams, plane: Plane):
        self.B, self.h, self.prm, self.plane = B, h, prm, plane
        self.ekf = BatchEKF(B, h, prm)
        self.nupd = np.zeros(B, int)

    def reset(self, ids=None) -> None:
        self.ekf.reset(ids)
        self.nupd[slice(None) if ids is None else ids] = 0

    def step(self, z: np.ndarray, valid: np.ndarray) -> None:
        self.ekf.step(z, valid)
        self.nupd += valid

    def ready(self) -> np.ndarray:
        return self.nupd >= ld.MIN_UPDATES

    def forecast(self) -> dict:
        fc = forecast(self.ekf.x, self.ekf.P, self.h, self.plane, NMAX, self.prm.sigma_a)
        fc["found"] &= self.ready()
        return fc


class HybridPredictor:
    def __init__(self, B: int, h: float, prm: EKFParams, plane: Plane, model: SeqPredictor, phi: float,
                 kappa: float = 1.0, sigma_floor: float = 0.05):
        self.B, self.h, self.prm, self.plane, self.model, self.phi, self.kappa = B, h, prm, plane, model, phi, kappa
        self.feat_ekf = BatchEKF(B, h, prm)                                     # physics EKF: feature extractor
        prm_h = EKFParams(sigma_a=prm.sigma_a, kd0=prm.kd0, kd0_std=0.0, sigma_meas=prm.sigma_meas, q_kd=0.0)
        self.ekf = BatchEKF(B, h, prm_h)                                        # hybrid EKF: state for the forecast
        self.hid = torch.zeros(1, B, model.hidden)
        self.mu = np.zeros((B, model.horizon, 3))
        self.sig = np.full((B, model.horizon, 3), prm.sigma_a)
        self.have_net = np.zeros(B, bool)
        self.nupd = np.zeros(B, int)
        self.sigma_floor = sigma_floor

    def reset(self, ids=None) -> None:
        ids = slice(None) if ids is None else ids
        self.feat_ekf.reset(ids)
        self.ekf.reset(ids)
        self.hid[:, ids] = 0.0
        self.mu[ids] = 0.0
        self.sig[ids] = self.prm.sigma_a
        self.have_net[ids] = False
        self.nupd[ids] = 0

    def step(self, z: np.ndarray, valid: np.ndarray) -> None:
        # 1. hybrid EKF: predict with the nowcast issued at the previous frame, then update
        hn = self.have_net
        r = np.where(hn[:, None], self.mu[:, 0], 0.0)
        rv = np.where(hn[:, None], self.kappa * self.sig[:, 0] ** 2, self.prm.sigma_a**2)
        self.ekf.step(z, valid, r, rv)
        # 2. physics EKF -> features -> GRU -> residual forecast for the coming steps
        self.feat_ekf.step(z, valid)
        ini = self.feat_ekf.initialized
        feats = ld.hybrid_features(self.feat_ekf.x, self.feat_ekf.nu, valid, self.prm)
        feats[~ini] = 0.0
        with torch.no_grad():
            H, hid = self.model.encode(torch.from_numpy(feats)[:, None, :], self.hid)
            mu, sig = self.model.decode_physical(H[:, 0])
        # tracks that have not started keep a zero hidden state (the offline pipeline starts at the first fix)
        keep = torch.from_numpy(ini)
        self.hid = torch.where(keep[None, :, None], hid, self.hid)
        self.mu[ini] = mu.numpy()[ini]
        self.sig[ini] = np.maximum(sig.numpy()[ini], self.sigma_floor)
        self.have_net |= ini
        self.nupd += valid

    def ready(self) -> np.ndarray:
        return self.nupd >= ld.MIN_UPDATES

    def forecast(self) -> dict:
        pad = NMAX - self.model.horizon
        mu = np.concatenate([self.mu, np.repeat(self.mu[:, -1:], pad, axis=1)], axis=1)
        sig = np.concatenate([self.sig, np.repeat(self.sig[:, -1:], pad, axis=1)], axis=1)
        fc = forecast(self.ekf.x, self.ekf.P, self.h, self.plane, NMAX, sig, mu, self.phi)
        fc["found"] &= self.ready()
        return fc


class OraclePredictor:
    """Knows the true crossing; ``set_truth`` must be called at reset with (t, p, v) per world."""

    def __init__(self, B: int, h: float, plane: Plane):
        self.B, self.h, self.plane = B, h, plane
        self.t = np.zeros(B)
        self.p = np.zeros((B, 3))
        self.v = np.zeros((B, 3))
        self.k = np.zeros(B, int)

    def reset(self, ids=None) -> None:
        self.k[slice(None) if ids is None else ids] = 0

    def set_truth(self, ids, t, p, v) -> None:
        self.t[ids], self.p[ids], self.v[ids] = t, p, v

    def step(self, z, valid) -> None:
        self.k += 1

    def ready(self) -> np.ndarray:
        return np.ones(self.B, bool)

    def forecast(self) -> dict:
        tau = self.t - (self.k - 1) * self.h      # the first step() call corresponds to time 0
        B = self.B
        return {"found": tau > 0, "t": tau, "p": self.p.copy(), "v": self.v.copy(), "k": np.floor(tau / self.h).astype(int),
                "cov": np.zeros((B, 3, 3)), "cov_uw": np.zeros((B, 2, 2)), "var_t": np.zeros(B)}


def load_ekf_params(obj: str) -> EKFParams:
    """EKF parameters tuned in M1 (results/m1_params.json)."""
    import json

    from card_catcher import RESULTS

    p = json.loads((RESULTS / "m1_params.json").read_text())[obj]
    return EKFParams(sigma_a=p["sigma_a"], kd0=p["kd0"], kd0_std=p["kd0_std"], sigma_meas=p["sigma_meas"], q_kd=p["q_kd"])


def make_predictor(kind: str, obj: str, B: int, cfg: dict, n_train: int = 4000, model_seed: int = 0):
    """kind: "oracle" | "physics" | "hybrid" (hybrid needs the M2 model and its fitted phi)."""
    import json

    from card_catcher import DATA, RESULTS
    from card_catcher.learn.train import load

    plane, h = Plane.from_cfg(cfg), cfg["sim"]["timestep"]
    if kind == "oracle":
        return OraclePredictor(B, h, plane)
    prm = load_ekf_params(obj)
    if kind == "physics":
        return PhysicsPredictor(B, h, prm, plane)
    if kind == "hybrid":
        model, _ = load(DATA / "models" / f"{obj}_residual_n{n_train}_s{model_seed}.pt")
        r = json.loads((RESULTS / "m2_residual.json").read_text())["objects"][obj]
        i = r["seeds"].index(model_seed)
        return HybridPredictor(B, h, prm, plane, model, r["phi_per_seed"][i], r["kappa_per_seed"][i])
    raise ValueError(kind)
