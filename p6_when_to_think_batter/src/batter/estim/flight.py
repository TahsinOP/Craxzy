"""The ball's flight model: the simulator's own discrete step, in closed form.

State x = [p (3), v (3), w (3)]: position, linear velocity and spin (angular velocity,
world frame) of the ball.  MuJoCo's ellipsoid fluid model applied to a sphere reduces to

    m dv/dt = m g - (k1 + k2 |v|) v + cm (w x v)        drag (Stokes + quadratic) and lift
    I dw/dt = -(k3 + k4 |w|) w                           spin decay

where the lift term is the Magnus force (rho V w x v) minus the gyroscopic part of the
added-mass force (rho V/2 w x v).  With the ``Euler`` integrator fluid forces are explicit,
so one simulator step is exactly

    v+ = v + h a(v, w),     p+ = p + h v+,     w+ = w + h alpha(w).                  (*)

The five coefficients are read off the engine by probing ``qfrc_passive`` of the
ball-only model (``FlightModel.identify``); tests/test_flight.py checks that (*) then
reproduces a simulated spinning pitch to 1e-9 m.

Everything is batched over leading dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])


def skew(a: np.ndarray) -> np.ndarray:
    """[a]x for a (..., 3) -> (..., 3, 3)."""
    z = np.zeros(a.shape[:-1])
    return np.stack([np.stack([z, -a[..., 2], a[..., 1]], -1),
                     np.stack([a[..., 2], z, -a[..., 0]], -1),
                     np.stack([-a[..., 1], a[..., 0], z], -1)], -2)


@dataclass(frozen=True)
class FlightModel:
    h: float          # timestep [s]
    k1: float         # linear drag / mass            [1/s]
    k2: float         # quadratic drag / mass         [1/m]
    cm: float         # lift coefficient / mass       [1/rad]
    k3: float         # linear spin damping / inertia [1/s]
    k4: float         # quadratic spin damping / inertia [1/rad]
    radius: float
    g: np.ndarray = None

    def __post_init__(self):
        if self.g is None:
            object.__setattr__(self, "g", GRAVITY.copy())

    # ------------------------------------------------------------ identification
    @staticmethod
    def identify(model: mujoco.MjModel) -> "FlightModel":
        """Read the coefficients off a model that contains the free body ``ball``."""
        m = model
        d = mujoco.MjData(m)
        bid = m.body("ball").id
        adr = int(m.jnt_dofadr[m.body_jntadr[bid]])
        qadr = int(m.jnt_qposadr[m.body_jntadr[bid]])
        mass = float(m.body_mass[bid])
        inertia = float(m.body_inertia[bid][0])

        def passive(v, w):
            mujoco.mj_resetData(m, d)
            d.qpos[qadr:qadr + 3] = [0.0, 0.0, 50.0]          # far from everything
            d.qvel[adr:adr + 3] = v
            d.qvel[adr + 3:adr + 6] = w
            mujoco.mj_forward(m, d)
            return d.qfrc_passive[adr:adr + 6].copy()

        f1, f2 = -passive([1.0, 0, 0], [0, 0, 0])[0], -passive([2.0, 0, 0], [0, 0, 0])[0]
        k2 = (f2 - 2 * f1) / 2.0                 # f(s) = k1 s + k2 s^2
        k1 = f1 - k2
        t1, t2 = -passive([0, 0, 0], [0, 10.0, 0])[4], -passive([0, 0, 0], [0, 20.0, 0])[4]
        k4 = (t2 - 2 * t1) / 200.0               # t(s) = k3 s + k4 s^2
        k3 = (t1 - 100.0 * k4) / 10.0
        cm = passive([1.0, 0, 0], [0, 1.0, 0])[2] / -1.0     # w x v = (0, 1, 0) x (1, 0, 0) = (0, 0, -1)
        radius = float(m.geom_size[m.geom("ball").id][0])
        return FlightModel(h=float(m.opt.timestep), k1=k1 / mass, k2=k2 / mass, cm=cm / mass, k3=k3 / inertia,
                           k4=k4 / inertia, radius=radius, g=np.array(m.opt.gravity))

    def with_step(self, h: float) -> "FlightModel":
        return FlightModel(h, self.k1, self.k2, self.cm, self.k3, self.k4, self.radius, self.g)

    # ------------------------------------------------------------------ dynamics
    def accel(self, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        s = np.linalg.norm(v, axis=-1, keepdims=True)
        return self.g - (self.k1 + self.k2 * s) * v + self.cm * np.cross(w, v)

    def spin_accel(self, w: np.ndarray) -> np.ndarray:
        s = np.linalg.norm(w, axis=-1, keepdims=True)
        return -(self.k3 + self.k4 * s) * w

    def step(self, x: np.ndarray, extra_accel: np.ndarray | None = None) -> np.ndarray:
        """One simulator step (*) for x (..., 9)."""
        p, v, w = x[..., :3], x[..., 3:6], x[..., 6:9]
        a = self.accel(v, w)
        if extra_accel is not None:
            a = a + extra_accel
        v2 = v + self.h * a
        return np.concatenate([p + self.h * v2, v2, w + self.h * self.spin_accel(w)], axis=-1)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """F = d step / dx, (..., 9, 9)."""
        h = self.h
        v, w = x[..., 3:6], x[..., 6:9]
        s = np.linalg.norm(v, axis=-1)[..., None, None]
        sw = np.linalg.norm(w, axis=-1)[..., None, None]
        I = np.eye(3)
        vv = v[..., :, None] * v[..., None, :]
        ww = w[..., :, None] * w[..., None, :]
        Avv = I + h * (-(self.k1 + self.k2 * s) * I - self.k2 * vv / np.maximum(s, 1e-9) + self.cm * skew(w))
        Avw = -h * self.cm * skew(v)
        Aww = I + h * (-(self.k3 + self.k4 * sw) * I - self.k4 * ww / np.maximum(sw, 1e-9))
        F = np.zeros(x.shape[:-1] + (9, 9))
        F[..., 0:3, 0:3] = I
        F[..., 0:3, 3:6] = h * Avv
        F[..., 0:3, 6:9] = h * Avw
        F[..., 3:6, 3:6] = Avv
        F[..., 3:6, 6:9] = Avw
        F[..., 6:9, 6:9] = Aww
        return F

    def rollout(self, x0: np.ndarray, n: int) -> np.ndarray:
        """States x_0 .. x_n, shape (n + 1, ..., 9)."""
        out = np.empty((n + 1,) + x0.shape)
        out[0] = x0
        for k in range(n):
            out[k + 1] = self.step(out[k])
        return out

    # ------------------------------------------------------------------- landing
    def landing(self, x0: np.ndarray, max_steps: int = 400) -> tuple[np.ndarray, np.ndarray]:
        """Where and when the ball first reaches the ground (centre at z = radius).

        Returns (xy (..., 2), time (...,)); linear interpolation inside the landing step.
        NaN where the ball does not come down within ``max_steps``.
        """
        x = np.array(x0, float)
        shape = x.shape[:-1]
        xy = np.full(shape + (2,), np.nan)
        t = np.full(shape, np.nan)
        done = x[..., 2] <= self.radius
        xy[done], t[done] = x[done, :2], 0.0
        for k in range(max_steps):
            if done.all():
                break
            x2 = self.step(x)
            hit = ~done & (x2[..., 2] <= self.radius)
            if hit.any():
                a = (x[hit, 2] - self.radius) / np.maximum(x[hit, 2] - x2[hit, 2], 1e-12)
                xy[hit] = x[hit, :2] + a[:, None] * (x2[hit, :2] - x[hit, :2])
                t[hit] = (k + a) * self.h
                done = done | hit
            x = x2
        return xy, t
