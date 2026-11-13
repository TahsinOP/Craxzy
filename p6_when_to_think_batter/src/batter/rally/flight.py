"""The ping-pong ball's flight with table bounces, in closed form.

Free flight is the batter's ``FlightModel`` (the simulator's ``Euler`` step, exact).  The
table-ball contact pair is frictionless, with a direct restitution law (zero stiffness,
damping (1 + e)/h) and speculative contacts.  Measured on thousands of simulated bounces
(scripts/rally_bounce.py), a bounce then takes exactly one step, the step whose ballistic
prediction would end below the surface, and in that step

    v'_xy = the ballistic step's v_xy,      w' = the ballistic step's w,
    v'_z  = -e v_z + b                      (v_z: the vertical velocity at the start of the step)
    p'    = p + h v'                        (the integrator's position update)

with e equal to the pair's restitution and a tiny offset b; ``BounceModel.identify`` fits both
on simulated bounces.  A bounce happens only over the table's surface; the net and the edges
are not modelled (a ball that touches them ends the rally anyway).

``TableFlight`` has the interface the EKF, the forecast and the planner expect (``h``,
``radius``, ``g``, ``step``, ``jacobian``, ``rollout``), so all of them predict through the
bounce unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mujoco import batch as mjbatch

from batter.estim.flight import FlightModel

STAGE_NONE, STAGE_VEL = int(mujoco.mjtStage.mjSTAGE_NONE), int(mujoco.mjtStage.mjSTAGE_VEL)


@dataclass(frozen=True)
class BounceModel:
    e: float              # normal restitution
    b: float              # offset [m/s]
    z_surface: float      # height of the ball centre at contact (table top + radius)
    x_range: tuple        # table extent (world frame)
    y_range: tuple

    @staticmethod
    def identify(model: mujoco.MjModel, fm: FlightModel, table: dict, n: int = 256, seed: int = 0) -> "BounceModel":
        """Fit (e, b) on ``n`` simulated bounces (one batch step each, so this takes a second)."""
        x0, x1 = table["gap"], table["gap"] + table["length"]
        zt = table["height"] + fm.radius
        rng = np.random.default_rng(seed)
        a = int(model.jnt_qposadr[model.joint("ball").id])
        dd = int(model.jnt_dofadr[model.joint("ball").id])
        h = float(model.opt.timestep)
        v = np.stack([rng.uniform(2, 7, n) * rng.choice([-1, 1], n), rng.uniform(-1, 1, n), -rng.uniform(1, 5, n)], 1)
        w = rng.normal(0, 60, (n, 3))
        z = zt + rng.uniform(0.05, 0.95, n) * h * np.abs(v[:, 2]) + 1e-4     # the next step crosses the surface
        xy = np.stack([rng.uniform(x0 + 0.3, 0.5 * (x0 + x1) - 0.3, n), rng.uniform(-0.5, 0.5, n)], 1)
        b = mjbatch.Batch(model, nsim=n, nthread=0, persistent=True)
        qpos, qvel = b.bind("qpos"), b.bind("qvel")
        b.reset(stage=STAGE_NONE)
        qpos[:, a:a + 3] = np.concatenate([xy, z[:, None]], 1)
        qpos[:, a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]
        qvel[:] = 0.0
        qvel[:, dd:dd + 3] = v
        qvel[:, dd + 3:dd + 6] = w
        b.forward(stage=STAGE_VEL)
        b.step(stage=STAGE_VEL)
        vz_post = qvel[:, dd + 2].copy()
        A = np.stack([-v[:, 2], np.ones(n)], 1)
        (e, off), *_ = np.linalg.lstsq(A, vz_post, rcond=None)
        return BounceModel(float(e), float(off), zt, (x0, x1), (-table["width"] / 2, table["width"] / 2))


class TableFlight:
    """FlightModel + table bounces, in the frame the bounce model was given in."""

    def __init__(self, fm: FlightModel, bounce: BounceModel):
        self.fm, self.bm = fm, bounce
        self.h, self.radius, self.g = fm.h, fm.radius, fm.g

    def with_frame(self, bounce: BounceModel) -> "TableFlight":
        return TableFlight(self.fm, bounce)

    def bounces(self, x: np.ndarray, x_ballistic: np.ndarray) -> np.ndarray:
        """Which states (..., 9) bounce in the coming step."""
        bm = self.bm
        p = x[..., 0:3]
        over = ((p[..., 0] >= bm.x_range[0]) & (p[..., 0] <= bm.x_range[1]) &
                (p[..., 1] >= bm.y_range[0]) & (p[..., 1] <= bm.y_range[1]))
        return over & (p[..., 2] >= bm.z_surface - 0.02) & (x_ballistic[..., 2] < bm.z_surface) & (x[..., 5] < 0)

    def step(self, x: np.ndarray, extra_accel: np.ndarray | None = None) -> np.ndarray:
        y = self.fm.step(x, extra_accel)
        bo = self.bounces(x, y)
        if np.any(bo):
            vz = -self.bm.e * x[..., 5] + self.bm.b
            y[..., 5] = np.where(bo, vz, y[..., 5])
            y[..., 2] = np.where(bo, x[..., 2] + self.h * vz, y[..., 2])
        return y

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        F = self.fm.jacobian(x)
        y = self.fm.step(x)
        bo = self.bounces(x, y)
        if np.any(bo):
            Fb = F[bo]
            Fb[:, 5, :] = 0.0
            Fb[:, 5, 5] = -self.bm.e                        # v_z' = -e v_z + b
            Fb[:, 2, :] = 0.0
            Fb[:, 2, 2] = 1.0                               # p_z' = p_z + h v_z'
            Fb[:, 2, 5] = -self.h * self.bm.e
            F[bo] = Fb
        return F

    def extra_process_noise(self, x: np.ndarray, sigma_vz: float = 0.5) -> np.ndarray:
        """Process noise (..., 9, 9) for the filter around a bounce.

        The bounce is a discrete event: a few millimetres of error in the estimated height decide which
        10 ms step it falls in, and putting it in the wrong step makes the vertical speed wrong by about
        (1 + e) |v_z|.  For a state within about one step of the surface (descending), the filter is told
        that its height and vertical speed may be off by that much, so the next fixes can correct it.
        """
        Q = np.zeros(x.shape[:-1] + (9, 9))
        bm = self.bm
        p, vz = x[..., 0:3], x[..., 5]
        over = ((p[..., 0] >= bm.x_range[0]) & (p[..., 0] <= bm.x_range[1]) &
                (p[..., 1] >= bm.y_range[0]) & (p[..., 1] <= bm.y_range[1]))
        near = over & (np.abs(p[..., 2] - bm.z_surface) < 1.5 * self.h * np.abs(vz) + 0.005)
        if np.any(near):
            sig = np.maximum(sigma_vz, (1.0 + bm.e) * np.abs(vz) * 0.5)
            Q[near, 5, 5] = sig[near] ** 2
            Q[near, 2, 2] = (self.h * sig[near]) ** 2
            Q[near, 2, 5] = Q[near, 5, 2] = self.h * sig[near] ** 2
        return Q

    def rollout(self, x0: np.ndarray, n: int) -> np.ndarray:
        out = np.empty((n + 1,) + x0.shape)
        out[0] = x0
        for k in range(n):
            out[k + 1] = self.step(out[k])
        return out

    # the batter's Aimer only uses these of a flight model
    def accel(self, v, w):
        return self.fm.accel(v, w)

    def spin_accel(self, w):
        return self.fm.spin_accel(w)

    def with_step(self, h: float) -> FlightModel:
        return self.fm.with_step(h)


def mirrored(bm: BounceModel, L: float) -> BounceModel:
    """The bounce model in arm B's frame: (x, y) -> (L - x, -y); the table is symmetric."""
    return BounceModel(bm.e, bm.b, bm.z_surface, (L - bm.x_range[1], L - bm.x_range[0]), (-bm.y_range[1], -bm.y_range[0]))
