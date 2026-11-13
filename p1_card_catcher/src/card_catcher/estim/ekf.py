"""Extended Kalman filter for a thrown object, and catch-point forecasting.

State  x = [p (3), v (3), k_d]   (position, velocity, quadratic drag coefficient)
Model  the discrete physics map of ``card_catcher.estim.physics`` with step h:

    v' = v + h ( w(v; k_d) + r + n ),      p' = p + h v',      k_d' = k_d + n_k

  w(v) = (I + hK)^-1 (g - k_d |v| v) is the simulator's linearly-implicit step,

  r is an optional known input (the learned residual mean, M2; zero in M1), n is
  white acceleration noise with per-axis variance s_a^2 (unmodelled forces) and
  n_k a slow random walk that lets k_d adapt within a throw.

Jacobian  F = d(p', v', k_d')/d(p, v, k_d) = [[I, hA, hc], [0, A, c], [0, 0, 1]],
  A = dv'/dv,  c = dv'/dk_d   (closed forms in physics.model_jacobian).
Process noise  Q = G diag(s_a^2) G^T + diag(0, 0, q_k),  G = [h^2 I; h I; 0].
Measurement  z = p + n,  n ~ N(0, s_m^2 I), at some steps only (rate, dropout).

Everything is batched over B independent filters (numpy), which is what the
multi-world environments need.

Forecast
--------
``forecast`` rolls the mean forward with the same map until the trajectory
crosses the catch plane, and propagates the covariance alongside.  The crossing
point of the piecewise-linear path is p_c = p_k + a (p_{k+1} - p_k).  To first
order, an error dp in the position *at the nominal crossing time* moves the
crossing point by

    dp_c = (I - v n^T / (n^T v)) dp,       dt_c = - n^T dp / (n^T v),

(the object arrives earlier/later and slides along its velocity v), so the
catch-point covariance is J S_p J^T with J = I - v n^T/(n^T v); it is singular
along n, and its in-plane 2x2 block is what M4's variance trigger uses.

Time-correlated model error.  The future acceleration error is not white (an
aerodynamic force that is wrong now is still wrong 50 ms later).  The forecast
therefore models it as e_j = s_j * eta_j with eta a unit-variance AR(1) process,
eta_{j+1} = phi eta_j + sqrt(1 - phi^2) xi_j, by augmenting the error state with
eta (state augmentation for coloured noise).  phi = 0 recovers the white-noise
assumption the filter itself makes; phi is fitted on held-out data in M2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from card_catcher.config import Plane
from card_catcher.estim.physics import model_accel, model_jacobian

NX = 7


@dataclass(frozen=True)
class EKFParams:
    sigma_a: float            # white acceleration noise std per axis [m/s^2]
    kd0: float                # prior mean of the drag coefficient [1/m]
    kd0_std: float            # prior std of the drag coefficient [1/m]
    sigma_meas: float         # measurement noise std per axis [m]
    q_kd: float = 0.0         # random-walk std of k_d per step [1/m]
    v0_std: float = 10.0      # diffuse prior std on the initial velocity [m/s]


def _jacobian(v: np.ndarray, kd: np.ndarray, h: float) -> np.ndarray:
    """F (B,7,7) of the step map at velocities v (B,3) and drag kd (B,)."""
    B = v.shape[0]
    A, c = model_jacobian(v, kd, h)
    F = np.zeros((B, NX, NX))
    F[:, :3, :3] = np.eye(3)
    F[:, :3, 3:6] = h * A
    F[:, :3, 6] = h * c
    F[:, 3:6, 3:6] = A
    F[:, 3:6, 6] = c
    F[:, 6, 6] = 1.0
    return F


def _step_mean(x: np.ndarray, h: float, r: np.ndarray | None) -> np.ndarray:
    v = x[:, 3:6]
    a = model_accel(v, x[:, 6], h)
    if r is not None:
        a = a + r
    out = x.copy()
    out[:, 3:6] = v + h * a
    out[:, :3] = x[:, :3] + h * out[:, 3:6]
    return out


class BatchEKF:
    """B independent EKFs advanced in lock-step with the simulator."""

    def __init__(self, B: int, h: float, params: EKFParams):
        self.B, self.h, self.prm = B, h, params
        self.x = np.zeros((B, NX))
        self.P = np.zeros((B, NX, NX))
        self.initialized = np.zeros(B, bool)
        self.nu = np.zeros((B, 3))          # last innovation (zero where no update happened)
        self.S = np.tile(np.eye(3), (B, 1, 1))
        G = np.zeros((NX, 3))
        G[:3] = h * h * np.eye(3)
        G[3:6] = h * np.eye(3)
        self._G = G
        self._Qk = np.zeros((NX, NX))
        self._Qk[6, 6] = params.q_kd**2

    def reset(self, ids=None) -> None:
        ids = slice(None) if ids is None else ids
        self.x[ids] = 0.0
        self.P[ids] = 0.0
        self.initialized[ids] = False

    def _init(self, ids: np.ndarray, z: np.ndarray) -> None:
        p = self.prm
        self.x[ids, :3] = z
        self.x[ids, 3:6] = 0.0
        self.x[ids, 6] = p.kd0
        P = np.zeros((len(ids), NX, NX))
        P[:, range(3), range(3)] = p.sigma_meas**2
        P[:, range(3, 6), range(3, 6)] = p.v0_std**2
        P[:, 6, 6] = p.kd0_std**2
        self.P[ids] = P
        self.initialized[ids] = True

    def step(self, z: np.ndarray, valid: np.ndarray, r: np.ndarray | None = None,
             r_var: np.ndarray | None = None) -> None:
        """Advance one simulator step and absorb the measurement taken after it.

        z (B,3): position fixes; valid (B,) bool: which filters received one.
        r (B,3): known residual acceleration over the step (mean); r_var (B,3): its
        variance, used in place of sigma_a^2 as the process noise of this step.
        Filters that are not yet initialised start from their first valid fix.
        """
        ini = self.initialized.copy()
        if ini.any():
            x, P = self.x[ini], self.P[ini]
            F = _jacobian(x[:, 3:6], x[:, 6], self.h)
            x = _step_mean(x, self.h, None if r is None else r[ini])
            var = np.full((x.shape[0], 3), self.prm.sigma_a**2) if r_var is None else r_var[ini]
            GQ = self._G[None] * var[:, None, :]
            P = F @ P @ F.transpose(0, 2, 1) + GQ @ self._G.T[None] + self._Qk[None]
            self.nu[ini] = 0.0
            upd = valid[ini]
            if upd.any():
                xu, Pu = x[upd], P[upd]
                S = Pu[:, :3, :3] + self.prm.sigma_meas**2 * np.eye(3)[None]
                K = np.linalg.solve(S, Pu[:, :3, :]).transpose(0, 2, 1)          # P H^T S^-1, (b,7,3)
                nu = z[ini][upd] - xu[:, :3]
                xu = xu + np.einsum("bij,bj->bi", K, nu)
                IKH = np.eye(NX)[None].repeat(len(xu), 0)
                IKH[:, :, :3] -= K
                Pu = IKH @ Pu @ IKH.transpose(0, 2, 1) + self.prm.sigma_meas**2 * (K @ K.transpose(0, 2, 1))  # Joseph
                x[upd], P[upd] = xu, Pu
                idx = np.flatnonzero(ini)[upd]
                self.nu[idx] = nu
                self.S[idx] = S
            x[:, 6] = np.maximum(x[:, 6], 0.0)       # drag cannot be negative
            self.x[ini], self.P[ini] = x, 0.5 * (P + P.transpose(0, 2, 1))
        new = (~ini) & valid
        if new.any():
            self._init(np.flatnonzero(new), z[new])


def forecast(x: np.ndarray, P: np.ndarray, h: float, plane: Plane, nmax: int, sigma: np.ndarray | float,
             r_seq: np.ndarray | None = None, phi: float = 0.0, return_path: bool = False) -> dict:
    """Predict the catch-plane crossing and its covariance from filter states.

    x (B,7), P (B,7,7): current estimates.  sigma: std of the future acceleration
    error, scalar or (B, nmax, 3).  r_seq (B, nmax, 3): future residual means.
    phi: AR(1) coefficient of the (unit-variance) acceleration-error process.

    Returns dict with found (B,), k (steps before crossing), t (time to crossing),
    p (B,3), v (B,3) at the crossing, cov (B,3,3) catch-point covariance,
    cov_uw (B,2,2) its in-plane block, var_t (B,) crossing-time variance.
    """
    B = x.shape[0]
    NA = NX + 3
    Pa = np.zeros((B, NA, NA))
    Pa[:, :NX, :NX] = P
    Pa[:, NX:, NX:] = np.eye(3)                       # eta_0 ~ N(0, I), independent of the state error
    sig = np.broadcast_to(np.asarray(sigma, float), (B, nmax, 3)) if np.ndim(sigma) else None
    xk = x.copy()
    n, c0 = plane.n, plane.c0
    s_prev = (xk[:, :3] - c0) @ n
    done = np.zeros(B, bool)
    out = {"found": np.zeros(B, bool), "k": np.zeros(B, int), "t": np.full(B, np.nan), "p": np.full((B, 3), np.nan),
           "v": np.full((B, 3), np.nan), "cov": np.full((B, 3, 3), np.nan), "cov_uw": np.full((B, 2, 2), np.nan),
           "var_t": np.full(B, np.nan)}
    path = [xk[:, :3].copy()] if return_path else None
    Fa = np.zeros((B, NA, NA))
    Fa[:, NX:, NX:] = phi * np.eye(3)
    Qeta = np.zeros((NA, NA))
    Qeta[NX:, NX:] = (1.0 - phi * phi) * np.eye(3)
    E = plane.basis
    for j in range(nmax):
        F = _jacobian(xk[:, 3:6], xk[:, 6], h)
        sj = np.full((B, 3), float(sigma)) if sig is None else sig[:, j]
        Fa[:, :NX, :NX] = F
        Fa[:, 0, NX], Fa[:, 1, NX + 1], Fa[:, 2, NX + 2] = h * h * sj[:, 0], h * h * sj[:, 1], h * h * sj[:, 2]
        Fa[:, 3, NX], Fa[:, 4, NX + 1], Fa[:, 5, NX + 2] = h * sj[:, 0], h * sj[:, 1], h * sj[:, 2]
        x_new = _step_mean(xk, h, None if r_seq is None else r_seq[:, j])
        s_new = (x_new[:, :3] - c0) @ n
        hit = (~done) & (s_prev > 0) & (s_new <= 0)
        if hit.any():
            a = s_prev[hit] / (s_prev[hit] - s_new[hit])
            pk, pn = xk[hit, :3], x_new[hit, :3]
            vel = (pn - pk) / h
            # error of the position at the nominal crossing instant: dp = M dx_k,
            # M = (1-a) [I 0 0 0] + a [row block of Fa for p]
            M = a[:, None, None] * Fa[hit, :3, :]
            M[:, range(3), range(3)] += 1.0 - a[:, None]
            Sp = M @ Pa[hit] @ M.transpose(0, 2, 1)
            nv = vel @ n
            J = np.eye(3)[None] - vel[:, :, None] * n[None, None, :] / nv[:, None, None]
            cov = J @ Sp @ J.transpose(0, 2, 1)
            out["found"][hit] = True
            out["k"][hit] = j
            out["t"][hit] = (j + a) * h
            out["p"][hit] = pk + a[:, None] * (pn - pk)
            out["v"][hit] = vel
            out["cov"][hit] = cov
            out["cov_uw"][hit] = E[None] @ cov @ E.T[None]
            out["var_t"][hit] = np.einsum("i,bij,j->b", n, Sp, n) / nv**2
            done |= hit
        if return_path:
            path.append(x_new[:, :3].copy())
        elif done.all():
            break
        Pa = Fa @ Pa @ Fa.transpose(0, 2, 1) + Qeta[None]
        xk, s_prev = x_new, s_new
    if return_path:
        out["path"] = np.stack(path, axis=1)
    return out


def fit_drag(V: np.ndarray, valid: np.ndarray, h: float) -> float:
    """Least-squares isotropic drag coefficient from simulated velocities.

    Minimises sum_k | (v_{k+1} - v_k)/h - w(v_k; k) |^2 over k (golden-section search;
    the objective is smooth and unimodal in k).   V (N,K+1,3), valid (N,K) bool.
    """
    from scipy.optimize import minimize_scalar

    a = (np.diff(V, axis=1) / h)[valid]
    v = V[:, :-1][valid]

    def cost(k: float) -> float:
        return float(((a - model_accel(v, k, h)) ** 2).sum())

    return float(minimize_scalar(cost, bounds=(0.0, 10.0), method="bounded", options={"xatol": 1e-7}).x)
