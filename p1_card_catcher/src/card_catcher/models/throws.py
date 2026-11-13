"""Throw generator: random launch position, velocity, orientation and spin.

A throw is aimed with the *nominal* physics model (discrete point mass with
isotropic quadratic drag, see ``card_catcher.estim.physics``): we draw a target
point in the catch-plane window and a flight time T, then solve for the launch
velocity that brings the nominal model to the target at time T (Newton on the
3x3 shooting problem).  The real flight then differs from the nominal one by the
aerodynamics the nominal model lacks -- by millimetres for the ball and by tens
of centimetres for the card.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from card_catcher.config import Plane
from card_catcher.estim.physics import GRAVITY, rollout_nominal


@dataclass(frozen=True)
class ThrowDist:
    x0: tuple[float, float]          # launch position ranges [m]
    y0: tuple[float, float]
    z0: tuple[float, float]
    flight_time: tuple[float, float]  # nominal time to the catch plane [s]
    kd_nominal: float                # nominal drag used only for aiming [1/m]
    tilt_std_deg: float              # std of the tilt of the body z-axis away from "edge-on"
    spin: tuple[float, float]        # |spin about body z| range [rad/s]
    wobble_std: float                # std of the other angular-velocity components [rad/s]
    random_attitude: bool = False    # ball: uniformly random attitude and spin axis


THROWS: dict[str, ThrowDist] = {
    "ball": ThrowDist((2.4, 3.2), (-0.5, 0.5), (0.9, 1.5), (0.60, 0.90), 0.02945, 0.0, (0.0, 0.0), 8.0, True),
    "plate": ThrowDist((2.0, 2.8), (-0.4, 0.4), (0.9, 1.4), (0.60, 0.90), 0.12, 8.0, (25.0, 50.0), 2.0),
    "card": ThrowDist((1.25, 1.75), (-0.3, 0.3), (1.15, 1.65), (0.45, 0.75), 1.0, 15.0, (20.0, 60.0), 3.0),
}


def solve_launch_velocity(p0: np.ndarray, target: np.ndarray, T: np.ndarray, kd: float, dt: float,
                          iters: int = 20) -> np.ndarray:
    """Launch velocities (N,3) such that the nominal discrete model is at ``target`` at time T.

    Newton iteration with a finite-difference Jacobian; T is rounded to whole steps.
    """
    n = p0.shape[0]
    steps = np.maximum(np.rint(T / dt).astype(int), 1)
    Ts = steps * dt
    kmax = int(steps.max())
    idx = np.arange(n)

    def end(v0):
        P, _ = rollout_nominal(p0, v0, kd, dt, kmax)
        return P[idx, steps]

    # drag-free initial guess (discrete semi-implicit Euler: p_K = p0 + T v0 + g T (T + dt) / 2)
    v = (target - p0 - 0.5 * GRAVITY * (Ts * (Ts + dt))[:, None]) / Ts[:, None]
    eps = 1e-4
    for _ in range(iters):
        r = end(v) - target
        if np.abs(r).max() < 1e-10:
            break
        J = np.zeros((n, 3, 3))
        for j in range(3):
            dv = np.zeros(3)
            dv[j] = eps
            J[:, :, j] = (end(v + dv) - end(v - dv)) / (2 * eps)
        v = v - np.linalg.solve(J, r[:, :, None])[:, :, 0]
    return v


def _random_unit(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.normal(size=(n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _quat_from_mat(R: np.ndarray) -> np.ndarray:
    """Rotation matrices (N,3,3) -> unit quaternions (N,4), w first (Shepperd's method)."""
    n = R.shape[0]
    q = np.zeros((n, 4))
    tr = np.trace(R, axis1=1, axis2=2)
    for i in range(n):
        m = R[i]
        if tr[i] > 0:
            s = np.sqrt(tr[i] + 1.0) * 2
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        else:
            k = int(np.argmax(np.diag(m)))
            j, l = (k + 1) % 3, (k + 2) % 3
            s = np.sqrt(1.0 + m[k, k] - m[j, j] - m[l, l]) * 2
            q[i, 0] = (m[l, j] - m[j, l]) / s
            q[i, 1 + k] = 0.25 * s
            q[i, 1 + j] = (m[j, k] + m[k, j]) / s
            q[i, 1 + l] = (m[l, k] + m[k, l]) / s
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _rodrigues(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotation matrices (N,3,3) from unit axes (N,3) and angles (N,)."""
    K = np.zeros((axis.shape[0], 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    s, c = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


def sample_throws(obj: str, n: int, plane: Plane, dt: float, rng: np.random.Generator,
                  dist: ThrowDist | None = None) -> dict[str, np.ndarray]:
    """Sample n throws.  Returns qpos (n,7), qvel (n,6) for the object's free joint.

    MuJoCo free-joint convention: qvel[:3] is the linear velocity in the world
    frame, qvel[3:] the angular velocity in the *body* frame.
    """
    d = dist or THROWS[obj]
    p0 = np.stack([rng.uniform(*d.x0, n), rng.uniform(*d.y0, n), rng.uniform(*d.z0, n)], axis=1)
    uw = np.stack([rng.uniform(*plane.u_range, n), rng.uniform(*plane.w_range, n)], axis=1)
    target = plane.point(uw)
    T = rng.uniform(*d.flight_time, n)
    v0 = solve_launch_velocity(p0, target, T, d.kd_nominal, dt)

    if d.random_attitude:
        q = rng.normal(size=(n, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        w_body = rng.normal(0.0, d.wobble_std, (n, 3))
    else:
        # body z (the thin axis) is perpendicular to the launch velocity ("edge-on",
        # as a thrown card or frisbee), as close to world-up as possible, then tilted.
        vhat = v0 / np.linalg.norm(v0, axis=1, keepdims=True)
        up = np.array([0.0, 0.0, 1.0])
        z = up - (vhat @ up)[:, None] * vhat
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        x = vhat
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=2)  # columns are body axes in world coordinates
        tilt = np.deg2rad(d.tilt_std_deg) * np.abs(rng.normal(size=n))
        phi = rng.uniform(0, 2 * np.pi, n)
        axis_body = np.stack([np.cos(phi), np.sin(phi), np.zeros(n)], axis=1)  # tilt axis in the body xy-plane
        R = R @ _rodrigues(axis_body, tilt)
        # random heading about the body z-axis (which edge leads)
        R = R @ _rodrigues(np.tile(up, (n, 1)), rng.uniform(0, 2 * np.pi, n))
        q = _quat_from_mat(R)
        spin = rng.uniform(*d.spin, n) * rng.choice([-1.0, 1.0], n)
        w_body = np.concatenate([rng.normal(0.0, d.wobble_std, (n, 2)), spin[:, None]], axis=1)

    return {
        "qpos": np.concatenate([p0, q], axis=1),
        "qvel": np.concatenate([v0, w_body], axis=1),
        "target": target,
        "T_nominal": T,
    }
