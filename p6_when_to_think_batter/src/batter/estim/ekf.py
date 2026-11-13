"""Ball EKF on delayed, range-dependent position fixes, and the forecast built on it.

State x = [p, v, w] (position, velocity, spin).  Process model = the simulator's own
step (estim/flight.py), so without surprises the model is exact and process noise is
only there to keep the filter numerically honest.  Spin is observed through the curvature
it causes (lift); its prior is the pitching machine's spin distribution.

Latency.  A camera frame read at simulator step k shows the ball at step k - n_d
(n_d = delay / h, the MuJoCo sensor delay).  The filter therefore lives n_d steps in the
past; the present is reached by n_d model steps, and the forecast continues from there.

Surprises.  A spin change or a gust violates the model.  The filter watches its own
normalised innovation squared (NIS, chi-square with 3 dof per fix): when the sum over the
last ``nis_window`` fixes exceeds ``nis_gate`` it adds ``reset_sigma_v`` / ``reset_sigma_w``
to the velocity and spin covariance (a covariance reset), so that it re-learns in a few
frames instead of averaging the surprise away.

Everything is batched over B worlds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from batter.estim.flight import FlightModel


@dataclass
class EKFParams:
    prior_v: tuple = (-10.5, 0.0, 1.6)            # m/s: a typical release velocity of the machine
    prior_sigma_v: tuple = (2.5, 0.8, 1.8)
    prior_sigma_w: tuple = (45.0, 90.0, 45.0)     # rad/s
    sigma_a: float = 0.05                         # m/s^2 process noise on the acceleration
    sigma_wdot: float = 5.0                       # rad/s^2 process noise on the spin
    nis_window: int = 6
    nis_gate: float = 50.0                        # chi-square(18) at 7e-5
    reset_sigma_v: float = 0.8
    reset_sigma_w: float = 150.0
    adaptive: bool = True


class BallEKF:
    def __init__(self, fm: FlightModel, B: int, meas: dict, params: EKFParams | None = None):
        self.fm, self.B, self.prm = fm, B, params or EKFParams()
        self.sigma0, self.sigma1 = meas["sigma0"], meas["sigma_per_m"]
        self.cam = np.asarray(meas["camera"], float)
        self.nd = int(round(meas["delay"] / fm.h))
        h, p = fm.h, self.prm
        q = np.zeros(9)
        q[0:3] = (p.sigma_a * h * h) ** 2
        q[3:6] = (p.sigma_a * h) ** 2
        q[6:9] = (p.sigma_wdot * h) ** 2
        self.Q = np.diag(q)
        self.reset()

    def reset(self) -> None:
        B = self.B
        self.x = np.zeros((B, 9))
        self.P = np.tile(np.eye(9), (B, 1, 1))
        self.started = np.zeros(B, bool)
        self.n_fix = np.zeros(B, int)
        self.nis = np.zeros(B)                    # NIS of the latest fix (0 when there was none)
        self._nis_hist = np.zeros((B, self.prm.nis_window))
        self.n_reset = np.zeros(B, int)
        # per-world priors for the next (re)start; None: the parameters' defaults
        self.prior_v_world: np.ndarray | None = None
        self.prior_w_world: np.ndarray | None = None
        self.prior_Pw_world: np.ndarray | None = None

    def restart(self, mask: np.ndarray, prior_v: np.ndarray, keep_spin: bool = True) -> None:
        """Start these worlds' tracks afresh at their next fix (rally: after a paddle hit).

        ``prior_v`` (B, 3) is the velocity prior per world.  With ``keep_spin``, the spin estimate and
        its covariance carry over: a frictionless paddle does not change the ball's spin.
        """
        mask = np.asarray(mask, bool)
        B = self.B
        if self.prior_v_world is None:
            self.prior_v_world = np.tile(np.asarray(self.prm.prior_v, float), (B, 1))
            self.prior_w_world = np.zeros((B, 3))
            self.prior_Pw_world = np.tile(np.diag(np.square(self.prm.prior_sigma_w)), (B, 1, 1))
        self.prior_v_world[mask] = prior_v[mask]
        if keep_spin:
            ok = mask & self.started
            self.prior_w_world[ok] = self.x[ok, 6:9]
            self.prior_Pw_world[ok] = self.P[ok, 6:9, 6:9]
        self.started[mask] = False
        self.n_fix[mask] = 0
        self._nis_hist[mask] = 0.0

    def meas_sigma(self, p: np.ndarray) -> np.ndarray:
        """Std per axis of a position fix of a ball at p: grows linearly with the range from the camera."""
        return self.sigma0 + self.sigma1 * np.linalg.norm(p - self.cam, axis=-1)

    # ------------------------------------------------------------------ filter
    def step(self, z: np.ndarray, valid: np.ndarray) -> None:
        """Advance the filter by one simulator step and fuse the fixes z (B, 3) where ``valid``."""
        prm = self.prm
        s = self.started
        if s.any():
            F = self.fm.jacobian(self.x[s])
            extra = self.fm.extra_process_noise(self.x[s]) if hasattr(self.fm, "extra_process_noise") else 0.0
            self.x[s] = self.fm.step(self.x[s])
            self.P[s] = F @ self.P[s] @ np.swapaxes(F, 1, 2) + self.Q + extra
        self.nis[:] = 0.0
        new = valid & ~s
        if new.any():                               # first fix: position from the camera, the rest from the prior
            self.x[new, 0:3] = z[new]
            self.x[new, 3:6] = prm.prior_v if self.prior_v_world is None else self.prior_v_world[new]
            self.x[new, 6:9] = 0.0 if self.prior_w_world is None else self.prior_w_world[new]
            r = self.meas_sigma(z[new]) ** 2
            P0 = np.zeros((int(new.sum()), 9, 9))
            idx = np.arange(3)
            P0[:, idx, idx] = r[:, None]
            P0[:, idx + 3, idx + 3] = np.square(prm.prior_sigma_v)
            if self.prior_Pw_world is None:
                P0[:, idx + 6, idx + 6] = np.square(prm.prior_sigma_w)
            else:
                P0[:, 6:9, 6:9] = self.prior_Pw_world[new]
            self.P[new] = P0
            self.started[new] = True
            self.n_fix[new] = 1
        upd = valid & s
        if upd.any():
            x, P = self.x[upd], self.P[upd]
            r = self.meas_sigma(x[:, 0:3]) ** 2
            S = P[:, 0:3, 0:3] + r[:, None, None] * np.eye(3)
            nu = z[upd] - x[:, 0:3]
            Sinv = np.linalg.inv(S)
            nis = np.einsum("bi,bij,bj->b", nu, Sinv, nu)
            K = P[:, :, 0:3] @ Sinv                                        # (b, 9, 3)
            x = x + np.einsum("bij,bj->bi", K, nu)
            # Joseph form with H = [I 0]
            IKH = np.tile(np.eye(9), (len(x), 1, 1))
            IKH[:, :, 0:3] -= K
            P = IKH @ P @ np.swapaxes(IKH, 1, 2) + r[:, None, None] * (K @ np.swapaxes(K, 1, 2))
            self.x[upd], self.P[upd] = x, P
            self.nis[upd] = nis
            self.n_fix[upd] += 1
            self._nis_hist[upd] = np.roll(self._nis_hist[upd], 1, axis=1)
            self._nis_hist[upd, 0] = nis
            if prm.adaptive:
                trip = np.zeros(self.B, bool)
                trip[upd] = (self._nis_hist[upd].sum(axis=1) > prm.nis_gate) & (self.n_fix[upd] > prm.nis_window + 2)
                if trip.any():
                    idx = np.arange(3)
                    self.P[np.ix_(trip, idx + 3, idx + 3)] += prm.reset_sigma_v**2 * np.eye(3)
                    self.P[np.ix_(trip, idx + 6, idx + 6)] += prm.reset_sigma_w**2 * np.eye(3)
                    self._nis_hist[trip] = 0.0
                    self.n_reset[trip] += 1

    # ---------------------------------------------------------------- forecast
    def forecast(self, K: int) -> tuple[np.ndarray, np.ndarray]:
        """Predicted states and position covariances from the filter's time onwards.

        Returns (traj (K + 1, B, 9), Ppp (K + 1, B, 3, 3)); index 0 is the filter's time
        (n_d steps ago), index n_d is now.  Process noise is ignored over the forecast.
        """
        B = self.B
        traj = np.empty((K + 1, B, 9))
        Ppp = np.empty((K + 1, B, 3, 3))
        traj[0] = self.x
        Ppp[0] = self.P[:, 0:3, 0:3]
        Phi = np.tile(np.eye(9), (B, 1, 1))
        for k in range(K):
            Phi = self.fm.jacobian(traj[k]) @ Phi
            traj[k + 1] = self.fm.step(traj[k])
            T = Phi[:, 0:3, :]
            Ppp[k + 1] = T @ self.P @ np.swapaxes(T, 1, 2)
        return traj, Ppp


def crossing_batch(traj: np.ndarray, Ppp: np.ndarray, x_c: float) -> dict:
    """Crossing of the plane x = x_c for every world of a forecast (K + 1, B, 9).

    Returns found (B,), steps (B,) fractional index into traj, p (B, 3), v (B, 3), w (B, 3) and
    cov (B, 2, 2): covariance of the crossing point's (y, z).  A position error dp moves the
    crossing point by dp - v dp_x / v_x.
    """
    K1, B, _ = traj.shape
    x = traj[:, :, 0]
    below = x[1:] <= x_c                                            # (K, B)
    found = below.any(axis=0) & (x[0] > x_c)
    i = np.where(found, below.argmax(axis=0), 0)
    b = np.arange(B)
    xi, xi1 = traj[i, b], traj[i + 1, b]
    a = (xi[:, 0] - x_c) / np.maximum(xi[:, 0] - xi1[:, 0], 1e-12)
    p = xi[:, 0:3] + a[:, None] * (xi1[:, 0:3] - xi[:, 0:3])
    v = xi1[:, 3:6]
    A = np.tile(np.eye(3), (B, 1, 1))
    A[:, :, 0] -= v / np.where(np.abs(v[:, 0:1]) > 1e-6, v[:, 0:1], -1e-6)
    C = A @ Ppp[i, b] @ np.swapaxes(A, 1, 2)
    return {"found": found, "steps": i + a, "p": p, "v": v, "w": xi[:, 6:9], "cov": C[:, 1:, 1:]}
