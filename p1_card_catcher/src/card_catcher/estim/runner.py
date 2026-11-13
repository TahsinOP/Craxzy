"""Run filters over recorded flights: synthetic measurements, filtering, forecasting."""
from __future__ import annotations

import numpy as np

from card_catcher.config import Plane
from card_catcher.estim.ekf import NX, BatchEKF, EKFParams, forecast


def simulate_measurements(P_true: np.ndarray, meas: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Noisy position fixes z (N,K+1,3) and their validity mask (N,K+1).

    One fix every ``meas['every']`` steps, each lost independently with probability
    ``meas['dropout']``; the first frame is always present (it starts the track).
    """
    N, K1, _ = P_true.shape
    z = P_true + rng.normal(0.0, meas["sigma"], P_true.shape)
    valid = np.zeros((N, K1), bool)
    valid[:, :: meas["every"]] = True
    valid &= rng.random((N, K1)) >= meas["dropout"]
    valid[:, 0] = True
    return z, valid


def run_filter(z: np.ndarray, valid: np.ndarray, h: float, params: EKFParams, r: np.ndarray | None = None,
               r_var: np.ndarray | None = None) -> dict:
    """Filter N tracks.  Returns posterior means X (N,K+1,7), covariances Pc (N,K+1,7,7),
    innovations nu (N,K+1,3) and their covariances S (N,K+1,3,3).

    r, r_var: optional (N,K,3) residual mean/variance for the step k -> k+1.
    """
    N, K1, _ = z.shape
    f = BatchEKF(N, h, params)
    X = np.zeros((N, K1, NX))
    Pc = np.zeros((N, K1, NX, NX))
    nu = np.zeros((N, K1, 3))
    S = np.zeros((N, K1, 3, 3))
    for k in range(K1):
        f.step(z[:, k], valid[:, k], None if (r is None or k == 0) else r[:, k - 1],
               None if (r_var is None or k == 0) else r_var[:, k - 1])
        X[:, k], Pc[:, k], nu[:, k], S[:, k] = f.x, f.P, f.nu, f.S
    return {"X": X, "P": Pc, "nu": nu, "S": S}


def innovation_nll(res: dict, valid: np.ndarray, burn_in: int = 3) -> float:
    """Mean negative log-likelihood of the innovations (per update), the quantity
    maximum-likelihood noise tuning minimises.  The first ``burn_in`` updates of a
    track are excluded (they are dominated by the diffuse velocity prior)."""
    used = valid.copy()
    used[:, 0] = False
    cnt = np.cumsum(used, axis=1)
    used &= cnt > burn_in
    nu, S = res["nu"][used], res["S"][used]
    sol = np.linalg.solve(S, nu[:, :, None])[:, :, 0]
    return float(np.mean(0.5 * (np.einsum("bi,bi->b", nu, sol) + np.log(np.linalg.det(S)) + 3 * np.log(2 * np.pi))))


def forecast_errors(res: dict, cross_k: np.ndarray, cross_t: np.ndarray, cross_p: np.ndarray, h: float, plane: Plane,
                    sigma, phi: float = 0.0, stride: int = 1, r_future=None, sigma_future=None,
                    min_updates: int = 4, valid: np.ndarray | None = None) -> dict:
    """Catch-point forecasts from every ``stride``-th step before the crossing.

    r_future / sigma_future: optional callables k -> (N, nmax, 3) giving the residual
    mean / error std for the steps after k (M2); otherwise ``sigma`` (scalar) is used.

    Returns flat arrays over all (track, step) pairs that produced a forecast:
    tau (time to the true crossing), err (3-D distance between predicted and true
    catch point), err_uw (in-plane error, 2), cov_uw (2x2), err_t, var_t, traj, k.
    """
    X, Pc = res["X"], res["P"]
    N, K1, _ = X.shape
    nupd = np.cumsum(valid, axis=1) if valid is not None else np.full((N, K1), min_updates)
    out = {k: [] for k in ("tau", "err", "err_uw", "cov_uw", "err_t", "var_t", "traj", "k")}
    for k in range(0, int(cross_k.max()) + 1, stride):
        act = np.flatnonzero((cross_k >= k) & (nupd[:, k] >= min_updates))
        if act.size == 0:
            continue
        nmax = int(cross_k[act].max() - k + 40)
        fc = forecast(X[act, k], Pc[act, k], h, plane, nmax,
                      sigma if sigma_future is None else sigma_future(k)[act][:, :nmax],
                      None if r_future is None else r_future(k)[act][:, :nmax], phi)
        ok = fc["found"]
        idx = act[ok]
        d = fc["p"][ok] - cross_p[idx]
        out["tau"].append(cross_t[idx] - k * h)
        out["err"].append(np.linalg.norm(d, axis=1))
        out["err_uw"].append(d @ plane.basis.T)
        out["cov_uw"].append(fc["cov_uw"][ok])
        out["err_t"].append(k * h + fc["t"][ok] - cross_t[idx])
        out["var_t"].append(fc["var_t"][ok])
        out["traj"].append(idx)
        out["k"].append(np.full(idx.size, k))
    return {k: np.concatenate(v) for k, v in out.items()}
