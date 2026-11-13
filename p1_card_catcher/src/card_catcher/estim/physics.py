"""The physics model shared by the throw generator, the EKF and all predictors.

Continuous model (ballistic flight with isotropic quadratic drag):

    p' = v,        v' = a(v) + r,        a(v) = g - k_d |v| v,

where r is the residual acceleration that M2 learns (zero for the pure physics
model).

Discretisation = the simulator's.  MuJoCo's ``implicitfast`` integrator treats
velocity-dependent forces linearly-implicitly: with K(v) = -da/dv, a free body
advances by

    v_{k+1} = v_k + h (I + h K(v_k))^-1 a(v_k) + h r_k,      p_{k+1} = p_k + h v_{k+1}.     (*)

For isotropic quadratic drag, K = k_d s (I + u u^T) with s = |v|, u = v/s, and
Sherman-Morrison gives the closed form (alpha = h k_d s)

    w(v) := (I + hK)^-1 a = g/(1+alpha) - alpha (u.g) u / ((1+alpha)(1+2 alpha)) - k_d s v / (1+2 alpha).

Statements that are tested (tests/test_ekf.py):

* (*) with r = 0 reproduces MuJoCo for a non-spinning sphere to 1e-9 m over 0.8 s
  when the air has no viscosity; with viscosity the only difference is the Stokes
  term 6 pi beta r v / m (0.3 mm over 0.8 s for the ball), which the fitted k_d
  absorbs.  Plain semi-implicit Euler (dropping the (I + hK)^-1) is off by 3 mm,
  and the continuous closed form by g h t / 2 = 3 cm, over the same flight.
* With r_k := (v_{k+1} - v_k)/h - w(v_k) computed from a simulated trajectory,
  (*) reproduces that trajectory exactly.  So "true minus physics" is a
  well-defined per-step quantity with no discretisation error hiding in it.
"""
from __future__ import annotations

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])


def drag_accel(v: np.ndarray, kd) -> np.ndarray:
    """a_drag = -k_d |v| v for v of shape (..., 3); kd scalar or (...,)."""
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    return -np.asarray(kd)[..., None] * speed * v


def model_accel(v: np.ndarray, kd, h: float) -> np.ndarray:
    """w(v) = (I + h K)^-1 (g - k_d |v| v): the effective acceleration of one step.

    v (..., 3); kd scalar or (...,).  Reduces to g - k_d |v| v as h -> 0.
    """
    kd = np.asarray(kd, float)[..., None]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    u = v / np.maximum(s, 1e-12)
    al = h * kd * s
    ug = u @ GRAVITY
    return GRAVITY / (1 + al) - (al / ((1 + al) * (1 + 2 * al))) * ug[..., None] * u - (kd * s / (1 + 2 * al)) * v


def model_jacobian(v: np.ndarray, kd: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Derivatives of v' = v + h w(v): A = dv'/dv (B,3,3) and c = dv'/dk_d (B,3).

    With alpha = h k s, c1 = 1/(1+alpha), c2 = alpha/((1+alpha)(1+2alpha)),
    f = k s/(1+2alpha), gamma = u.g:   w = c1 g - c2 gamma u - f v, and

      dw/dv = h k c1' g u^T - h k c2' gamma u u^T - c2 u (g - gamma u)^T / s
              - c2 gamma (I - u u^T)/s - f I - (k s/(1+2alpha)^2) u u^T,
      dw/dk = h s c1' g - h s c2' gamma u - s v/(1+2alpha)^2,

    c1' = -1/(1+alpha)^2,  c2' = (1 - 2 alpha^2)/((1+alpha)^2 (1+2alpha)^2).
    """
    B = v.shape[0]
    s = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
    u = v / s[:, None]
    al = h * kd * s
    c2 = al / ((1 + al) * (1 + 2 * al))
    c1p = -1.0 / (1 + al) ** 2
    c2p = (1 - 2 * al**2) / ((1 + al) ** 2 * (1 + 2 * al) ** 2)
    f = kd * s / (1 + 2 * al)
    gam = u @ GRAVITY
    eye = np.eye(3)[None]
    uu = u[:, :, None] * u[:, None, :]
    g = np.broadcast_to(GRAVITY, (B, 3))
    dw = ((h * kd * c1p)[:, None, None] * g[:, :, None] * u[:, None, :]
          - (h * kd * c2p * gam)[:, None, None] * uu
          - (c2 / s)[:, None, None] * u[:, :, None] * (g - gam[:, None] * u)[:, None, :]
          - (c2 * gam / s)[:, None, None] * (eye - uu)
          - f[:, None, None] * eye
          - (kd * s / (1 + 2 * al) ** 2)[:, None, None] * uu)
    dk = (h * s * c1p)[:, None] * g - (h * s * c2p * gam)[:, None] * u - (s / (1 + 2 * al) ** 2)[:, None] * v
    return eye + h * dw, h * dk


def step(p: np.ndarray, v: np.ndarray, kd, h: float, r: np.ndarray | None = None):
    """One step of the discrete physics model (*) (batched over leading dimensions)."""
    a = model_accel(v, kd, h)
    if r is not None:
        a = a + r
    v_new = v + h * a
    return p + h * v_new, v_new


def rollout_nominal(p0: np.ndarray, v0: np.ndarray, kd, h: float, nstep: int,
                    r: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Roll the discrete model forward.

    p0, v0: (N,3) or (3,).  r: optional residual accelerations (N, nstep, 3) or (nstep, 3).
    Returns P, V of shape (N, nstep+1, 3) (or (nstep+1, 3) for unbatched input).
    """
    single = p0.ndim == 1
    p = np.atleast_2d(p0).astype(float)
    v = np.atleast_2d(v0).astype(float)
    kd_arr = np.broadcast_to(np.asarray(kd, float), (p.shape[0],))
    if r is not None and r.ndim == 2:
        r = r[None]
    P = np.empty((p.shape[0], nstep + 1, 3))
    V = np.empty_like(P)
    P[:, 0], V[:, 0] = p, v
    for k in range(nstep):
        p, v = step(p, v, kd_arr, h, None if r is None else r[:, k])
        P[:, k + 1], V[:, k + 1] = p, v
    return (P[0], V[0]) if single else (P, V)


def residual_accel(V: np.ndarray, kd, h: float) -> np.ndarray:
    """Per-step residual r_k = (v_{k+1} - v_k)/h - w(v_k) for V of shape (..., K+1, 3)."""
    a = np.diff(V, axis=-2) / h
    kd = np.asarray(kd, float)
    if kd.ndim:
        kd = kd[..., None]
    return a - model_accel(V[..., :-1, :], kd, h)


def plane_crossing(P: np.ndarray, c0: np.ndarray, n: np.ndarray, h: float, V: np.ndarray | None = None):
    """First crossing of the plane n.(x - c0) = 0 from the positive side.

    P: (N, K+1, 3) piecewise-linear trajectories.  The crossing is located by
    linear interpolation inside the step where the signed distance changes sign,
    which is the definition of "catch point" used for truth and predictions alike.

    Returns dict with ``found`` (N,), ``k`` (step index before the crossing),
    ``t`` (crossing time from the first sample), ``p`` (N,3) and, if V is given,
    ``v`` (N,3) linearly interpolated.
    """
    s = (P - c0) @ n                                   # (N, K+1)
    cross = (s[:, :-1] > 0) & (s[:, 1:] <= 0)
    found = cross.any(axis=1)
    k = np.where(found, cross.argmax(axis=1), 0)
    idx = np.arange(P.shape[0])
    s0, s1 = s[idx, k], s[idx, k + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(found, s0 / (s0 - s1), 0.0)
    out = {
        "found": found,
        "k": k,
        "alpha": a,
        "t": (k + a) * h,
        "p": P[idx, k] + a[:, None] * (P[idx, k + 1] - P[idx, k]),
    }
    if V is not None:
        out["v"] = V[idx, k] + a[:, None] * (V[idx, k + 1] - V[idx, k])
    return out
