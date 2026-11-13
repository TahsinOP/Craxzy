"""Model-based arm control: kinematics, inverse kinematics and computed torque.

Discrete-time computed torque
-----------------------------
MuJoCo's ``implicitfast`` integrator advances the joint velocity by

    (M + h D) (v_{k+1} - v_k) / h = tau + f_passive - c,      q_{k+1} = q_k + h v_{k+1},

where M is the joint-space inertia (including armature), D = diag(joint damping)
(the velocity derivative of the passive force, treated implicitly), c the bias
(Coriolis + gravity) and f_passive = -D v_k + (fluid forces on the links).  So the
torque that makes the arm realise a commanded acceleration ``a`` *exactly* over
one step, ``v_{k+1} = v_k + h a``, is

    tau = (M + h D) a + c - f_passive.                                    (1)

(Continuous-time computed torque, tau = M a + c - f_passive, is off by h D a.)

With a_cmd = a_ref + Kd (v_ref - v) + Kp (q_ref - q), and a reference that
obeys the same integrator, the tracking error e = q_ref - q follows

    e_{k+1} - (2 - h Kd - h^2 Kp) e_k + (1 - h Kd) e_{k-1} = 0,

whose characteristic polynomial has a double root z0 = exp(-w h) (critically
damped, bandwidth w) for

    Kd = (1 - z0^2) / h,        Kp = (1 - z0)^2 / h^2.                    (2)

These are the gains used here; they tend to the familiar 2w and w^2 as h -> 0.
"""
from __future__ import annotations

import mujoco
import numpy as np

from card_catcher.models.scene import ARM_JOINTS, CatchIndex, TAU_MAX


def discrete_pd_gains(omega: float, h: float) -> tuple[float, float]:
    """(Kp, Kd) placing both closed-loop poles of the sampled error dynamics at exp(-omega h)."""
    z0 = np.exp(-omega * h)
    return (1.0 - z0) ** 2 / h**2, (1.0 - z0**2) / h


def dense_mass_index(m: mujoco.MjModel) -> np.ndarray:
    """Index map such that M_dense = where(idx >= 0, d.M[idx], 0) (lower+upper triangle)."""
    d = mujoco.MjData(m)
    d.M[:] = np.arange(1, d.M.size + 1)
    idx = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, idx)
    return idx.astype(int) - 1


def rotvec(R_des: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotation vector phi (world frame) with exp([phi]x) R = R_des."""
    E = R_des @ R.T
    cos = np.clip((np.trace(E) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(cos)
    w = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    if ang < 1e-9:
        return 0.5 * w
    if np.pi - ang < 1e-6:  # near pi: take the axis from the symmetric part
        A = (E + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        k = int(np.argmax(axis))
        axis = A[:, k] / axis[k]
        return ang * axis / np.linalg.norm(axis)
    return ang / (2.0 * np.sin(ang)) * w


class Arm:
    """Kinematics/dynamics helper for the 7 arm joints of the catch model."""

    def __init__(self, model: mujoco.MjModel, omega: float = 25.0):
        self.m = model
        self.ix = CatchIndex.of(model)
        self.h = model.opt.timestep
        self.d = mujoco.MjData(model)          # scratch data for kinematics
        self.qadr, self.dadr = self.ix.arm_qadr, self.ix.arm_dadr
        jid = [model.joint(j).id for j in ARM_JOINTS]
        self.q_min = model.jnt_range[jid, 0].copy()
        self.q_max = model.jnt_range[jid, 1].copy()
        self.damping = model.dof_damping[self.dadr].copy()
        self.kp, self.kd = discrete_pd_gains(omega, self.h)
        full = dense_mass_index(model)
        self.midx = full[np.ix_(self.dadr, self.dadr)]
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # ------------------------------------------------------------ kinematics
    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TCP position (3,) and rotation matrix (3,3) for arm configuration q."""
        d = self.d
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        return d.site_xpos[self.ix.tcp_site].copy(), d.site_xmat[self.ix.tcp_site].reshape(3, 3).copy()

    def jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(J 6x7 [linear; angular], pos, R) of the TCP site."""
        d = self.d
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        mujoco.mj_jacSite(self.m, d, self._jacp, self._jacr, self.ix.tcp_site)
        J = np.vstack([self._jacp[:, self.dadr], self._jacr[:, self.dadr]])
        return J, d.site_xpos[self.ix.tcp_site].copy(), d.site_xmat[self.ix.tcp_site].reshape(3, 3).copy()

    def ik(self, pos: np.ndarray, R: np.ndarray, q0: np.ndarray, q_rest: np.ndarray | None = None,
           w_rot: float = 0.3, iters: int = 60, damp: float = 0.02, null_gain: float = 0.05,
           tol: float = 1e-4, limit_margin: float = 0.02) -> tuple[np.ndarray, float, float]:
        """Damped-least-squares IK with joint-limit clamping and a null-space posture task.

        Minimises |p(q) - pos|^2 + w_rot^2 |rotvec(R, R(q))|^2 (orientation is traded
        against position only when the pose is not exactly reachable).  Returns
        (q, position error [m], orientation error [rad]).
        """
        q = np.clip(np.asarray(q0, float).copy(), self.q_min + limit_margin, self.q_max - limit_margin)
        q_rest = q if q_rest is None else q_rest
        W = np.array([1, 1, 1, w_rot, w_rot, w_rot], float)
        ep = er = np.inf
        for _ in range(iters):
            J, p, Rc = self.jacobian(q)
            e = np.concatenate([pos - p, rotvec(R, Rc)])
            ep, er = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
            if ep < tol and er < 10 * tol:
                break
            Jw, ew = W[:, None] * J, W * e
            # limit the task step (trust region) so the linearisation stays valid
            nrm = np.linalg.norm(ew)
            if nrm > 0.2:
                ew = ew * (0.2 / nrm)
            JJt = Jw @ Jw.T + damp**2 * np.eye(6)
            dq = Jw.T @ np.linalg.solve(JJt, ew)
            Npr = np.eye(7) - Jw.T @ np.linalg.solve(JJt, Jw)
            dq += Npr @ (null_gain * (q_rest - q))
            q = np.clip(q + dq, self.q_min + limit_margin, self.q_max - limit_margin)
        return q, ep, er

    # -------------------------------------------------------------- dynamics
    def mass_matrix(self, M_sparse: np.ndarray) -> np.ndarray:
        """Dense 7x7 arm inertia from mjData.M; M_sparse has shape (..., nM)."""
        return np.where(self.midx >= 0, M_sparse[..., np.maximum(self.midx, 0)], 0.0)

    def torque(self, M_sparse: np.ndarray, bias: np.ndarray, passive: np.ndarray, q: np.ndarray, v: np.ndarray,
               q_ref: np.ndarray, v_ref: np.ndarray, a_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Discrete computed torque, eq. (1)-(2); batched over leading dimensions.

        ``bias``/``passive`` are mjData.qfrc_bias/qfrc_passive (full nv).  Returns the
        torque clipped to the actuator limits and a boolean mask of saturated joints.
        """
        a_cmd = a_ref + self.kd * (v_ref - v) + self.kp * (q_ref - q)
        Md = self.mass_matrix(M_sparse) + self.h * np.diag(self.damping)
        tau = np.einsum("...ij,...j->...i", Md, a_cmd) + bias[..., self.dadr] - passive[..., self.dadr]
        sat = np.abs(tau) > TAU_MAX
        return np.clip(tau, -TAU_MAX, TAU_MAX), sat


def hand_rotation(approach: np.ndarray, closing_hint: np.ndarray) -> np.ndarray:
    """Rotation matrix [x y z] of the hand: z = approach axis (out of the palm, towards
    the incoming object), y = jaw-closing axis = ``closing_hint`` made orthogonal to z."""
    z = np.asarray(approach, float)
    z = z / np.linalg.norm(z)
    y = np.asarray(closing_hint, float)
    y = y - (y @ z) * z
    y = y / np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], axis=1)


READY_SEED = np.array([0.0, -0.2, 0.0, -1.9, 0.0, 2.4, 0.785])


def ready_configuration(arm: Arm, plane, closing: str = "vertical", approach: np.ndarray | None = None) -> np.ndarray:
    """Arm configuration with the TCP at the centre of the catch plane and the palm
    facing ``approach`` (default: the plane normal).  ``closing``: jaw axis along the
    plane's up-slope direction ("vertical": one finger above, one below) or along
    world y ("horizontal")."""
    axis = plane.ew if closing == "vertical" else plane.eu
    a = plane.n if approach is None else approach
    q, ep, er = arm.ik(plane.c0, hand_rotation(a, axis), READY_SEED, READY_SEED, iters=300, null_gain=0.02)
    if ep > 1e-3 or er > 1e-2:
        raise RuntimeError(f"ready pose not reachable (position error {ep:.2e} m, orientation error {er:.2e} rad)")
    return q
