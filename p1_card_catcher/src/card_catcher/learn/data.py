"""Feature and target construction for the sequence predictors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from card_catcher.envs.flight import FlightData
from card_catcher.estim.ekf import EKFParams
from card_catcher.estim.physics import residual_accel
from card_catcher.estim.runner import run_filter, simulate_measurements

V_SCALE = 5.0          # m/s, feature normalisation
NU_CLIP = 8.0          # innovations are fed in units of the measurement noise and clipped
MIN_UPDATES = 4        # forecasts start once a track has this many fixes
N_FEAT_HYBRID, N_FEAT_PURE = 8, 5


def hybrid_features(X: np.ndarray, nu: np.ndarray, valid: np.ndarray, prm: EKFParams) -> np.ndarray:
    """Features from the physics EKF: filtered velocity, innovation, validity, drag estimate.

    X (..., 7) posterior means, nu (..., 3) innovations, valid (...,) -> (..., 8).
    """
    f = np.zeros(X.shape[:-1] + (N_FEAT_HYBRID,), np.float32)
    f[..., 0:3] = X[..., 3:6] / V_SCALE
    f[..., 3:6] = np.clip(nu / prm.sigma_meas, -NU_CLIP, NU_CLIP)
    f[..., 6] = valid
    f[..., 7] = (X[..., 6] - prm.kd0) / max(prm.kd0, 1e-6)
    return f


def pure_features_step(z: np.ndarray, valid: np.ndarray, anchor: np.ndarray, gap: np.ndarray, h: float):
    """One frame of raw features.  anchor (B,3): last valid fix; gap (B,): steps since it.

    Returns features (B,5) and the updated (anchor, gap).  Feature = finite-difference
    velocity between consecutive valid fixes (zero when the frame is missing), the
    validity flag and the gap length.
    """
    B = z.shape[0]
    f = np.zeros((B, N_FEAT_PURE), np.float32)
    gap = gap + 1
    has = valid & np.isfinite(anchor[:, 0])
    f[has, 0:3] = (z[has] - anchor[has]) / (gap[has, None] * h) / V_SCALE
    f[:, 3] = valid
    f[:, 4] = np.where(valid, gap, 0) / 5.0
    anchor = np.where(valid[:, None], z, anchor)
    gap = np.where(valid, 0, gap)
    return f, anchor, gap


@dataclass
class SeqData:
    """Tensors for training/evaluating one predictor on N tracks of K+1 frames."""

    feats: np.ndarray        # (N, K+1, n_in)
    k_mask: np.ndarray       # (N, K+1) bool: frames from which a forecast is made
    r: np.ndarray            # (N, K, 3) true residual accelerations (hybrid target source)
    p: np.ndarray            # (N, K+1, 3) true positions (pure target source)
    anchor: np.ndarray       # (N, K+1, 3) last valid fix at each frame (pure)
    X: np.ndarray            # (N, K+1, 7) physics-EKF means
    z: np.ndarray
    valid: np.ndarray


def build(D: FlightData, prm: EKFParams, meas: dict, rng: np.random.Generator, kind: str) -> SeqData:
    z, valid = simulate_measurements(D.p, meas, rng)
    res = run_filter(z, valid, D.dt, prm)
    N, K1, _ = z.shape
    nupd = np.cumsum(valid, axis=1)
    k_mask = (nupd >= MIN_UPDATES) & (np.arange(K1)[None] <= D.cross_k[:, None])
    anchor_hist = np.zeros((N, K1, 3))
    if kind == "residual":
        feats = hybrid_features(res["X"], res["nu"], valid, prm)
    else:
        feats = np.zeros((N, K1, N_FEAT_PURE), np.float32)
        anchor, gap = np.full((N, 3), np.nan), np.zeros(N)
        for k in range(K1):
            feats[:, k], anchor, gap = pure_features_step(z[:, k], valid[:, k], anchor, gap, D.dt)
            anchor_hist[:, k] = anchor
    return SeqData(feats=feats, k_mask=k_mask, r=residual_accel(D.v, prm.kd0, D.dt), p=D.p, anchor=anchor_hist,
                   X=res["X"], z=z, valid=valid)


def future_index(K1: int, horizon: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Index matrix idx[k, j] = k + j + offset (clipped) and its in-range mask, for gathering targets."""
    idx = np.arange(K1)[:, None] + np.arange(horizon)[None, :] + offset
    return np.minimum(idx, 10**9), idx
