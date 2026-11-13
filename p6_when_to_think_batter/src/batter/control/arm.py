"""Model-based arm control: kinematics of the bat, inverse kinematics, computed torque.

Discrete-time computed torque
-----------------------------
With the ``Euler`` integrator (joint damping implicit) the joint velocity advances by

    (M + h D) (v_{k+1} - v_k) / h = tau + f_passive - c,      q_{k+1} = q_k + h v_{k+1},

M the joint-space inertia (with armature), D = diag(joint damping), c the bias force and
f_passive = -D v_k + (fluid forces on the links).  The torque that realises a commanded
acceleration ``a`` exactly over one step, v_{k+1} = v_k + h a, is

    tau = (M + h D) a + c - f_passive.                                    (1)

With a_cmd = a_ref + Kd (v_ref - v) + Kp (q_ref - q) and a reference that obeys the same
integrator, the tracking error has both poles at z0 = exp(-w h) for

    Kd = (1 - z0^2) / h,        Kp = (1 - z0)^2 / h^2.                    (2)

Bat task
--------
The bat's task frame is the ``sweet`` site at the centre of the blade; its x axis is the
face normal.  A strike needs the sweet spot at a point (3 constraints) and the face
normal along a direction (2 constraints): 5 of the arm's 7 degrees of freedom.  The roll
of the blade about its own normal and the elbow are left to a posture task.

Ready pose
----------
``NOMINAL_STRIKE`` is the strike posture for a ball through the middle of the zone that
gives the highest bat speed within the joint speed limits (scripts/design_ready.py, a
random-restart search).  ``READY`` is that posture wound back along the swing, each joint
by 1.5 times the distance it needs to reach its strike speed at its acceleration limit,
so a swing starts with its run-up already in place.
"""
from __future__ import annotations

import mujoco
import numpy as np

from batter.models.scene import ARM_JOINTS, CageIndex, QD_MAX, TAU_MAX

# produced by scripts/design_ready.py
NOMINAL_STRIKE = np.array([0.2050, 0.1209, 0.4362, -1.7363, -1.0080, 1.3968, -1.4129])
READY = np.array([0.2100, 0.1243, 0.4437, -1.8705, -0.7135, 1.1754, -1.4129])


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


class Arm:
    """Kinematics/dynamics helper for the 7 arm joints of the cage model."""

    def __init__(self, model: mujoco.MjModel, omega: float = 30.0):
        self.m = model
        self.ix = CageIndex.of(model)
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
        self.n_fk = 0

    # ------------------------------------------------------------ kinematics
    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sweet-spot position (3,) and rotation matrix (3,3) (columns: normal, width, length axes)."""
        d = self.d
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        self.n_fk += 1
        return d.site_xpos[self.ix.sweet_site].copy(), d.site_xmat[self.ix.sweet_site].reshape(3, 3).copy()

    def jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(Jv 3x7, Jw 3x7, pos, R) of the sweet-spot site."""
        d = self.d
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        mujoco.mj_jacSite(self.m, d, self._jacp, self._jacr, self.ix.sweet_site)
        self.n_fk += 1
        return (self._jacp[:, self.dadr].copy(), self._jacr[:, self.dadr].copy(),
                d.site_xpos[self.ix.sweet_site].copy(), d.site_xmat[self.ix.sweet_site].reshape(3, 3).copy())

    def ik(self, pos: np.ndarray, normal: np.ndarray, q0: np.ndarray, q_rest: np.ndarray | None = None,
           w_rot: float = 0.3, iters: int = 40, damp: float = 0.02, null_gain: float = 0.1,
           tol: float = 2e-4, limit_margin: float = 0.03) -> tuple[np.ndarray, float, float]:
        """Damped least squares for the 5-dof strike task: sweet spot at ``pos``, face normal ``normal``.

        The remaining freedom is pulled towards ``q_rest`` in the null space.  Returns
        (q, position error [m], normal error [rad]).
        """
        lo, hi = self.q_min + limit_margin, self.q_max - limit_margin
        q = np.clip(np.asarray(q0, float).copy(), lo, hi)
        q_rest = q.copy() if q_rest is None else q_rest
        n_des = normal / np.linalg.norm(normal)
        ep = er = np.inf
        I7 = np.eye(7)
        for _ in range(iters):
            Jv, Jw, p, R = self.jacobian(q)
            n = R[:, 0]
            phi = np.cross(n, n_des)                       # rotation vector (small angle) taking n to n_des
            c = float(n @ n_des)
            ang = float(np.arctan2(np.linalg.norm(phi), c))
            if np.linalg.norm(phi) > 1e-9:
                phi = phi / np.linalg.norm(phi) * ang
            elif c < 0:                                    # antiparallel: rotate about the blade's long axis
                phi = R[:, 2] * np.pi
            ep, er = float(np.linalg.norm(pos - p)), ang
            if ep < tol and er < 5 * tol:
                break
            P = np.eye(3) - np.outer(n, n)                 # rotations about n do not change n
            J = np.vstack([Jv, w_rot * (P @ Jw)])
            e = np.concatenate([pos - p, w_rot * phi])
            nrm = np.linalg.norm(e)
            if nrm > 0.25:                                 # trust region
                e = e * (0.25 / nrm)
            JJt = J @ J.T + damp**2 * np.eye(6)
            Jp = J.T @ np.linalg.inv(JJt)
            dq = Jp @ e + (I7 - Jp @ J) @ (null_gain * (q_rest - q))
            q = np.clip(q + dq, lo, hi)
        return q, ep, er

    def strike_velocity(self, q: np.ndarray, normal: np.ndarray, speed: float, mu: float = 1.0,
                        w_limits: np.ndarray = QD_MAX, mu_rot: float = 0.0) -> np.ndarray:
        """Joint velocity that moves the sweet spot at ``speed`` along the face normal.

        Only the normal component is constrained (the Panda's joint speed limits leave no
        room for more: a pure translation of the blade tops out near 0.9 m/s, this near
        2 m/s).  The freedom is spent on  min sum (qd_j / qd_max_j)^2 + mu |v_tangential|^2,
        so the contact point slides little along the face if the timing is off.  ``mu_rot`` also
        penalises turning the face normal (angular velocity about the two in-face axes), so a
        contact a step early or late leaves at the same angle (the rally's small paddle needs it).
        """
        Jv, Jw, _, R = self.jacobian(q)
        Jn = normal @ Jv
        Jt = R[:, 1:3].T @ Jv
        H = np.diag(1.0 / w_limits**2) + mu * Jt.T @ Jt
        if mu_rot > 0.0:
            Jr = R[:, 1:3].T @ Jw
            H = H + mu_rot * Jr.T @ Jr
        x = np.linalg.solve(H, Jn)
        return x * (speed / (Jn @ x))

    # -------------------------------------------------------------- dynamics
    def mass_matrix(self, M_sparse: np.ndarray) -> np.ndarray:
        """Dense 7x7 arm inertia from mjData.M; M_sparse has shape (..., nM)."""
        return np.where(self.midx >= 0, M_sparse[..., np.maximum(self.midx, 0)], 0.0)

    def torque(self, M_sparse: np.ndarray, bias: np.ndarray, passive: np.ndarray, q: np.ndarray, v: np.ndarray,
               q_ref: np.ndarray, v_ref: np.ndarray, a_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Discrete computed torque, eq. (1)-(2); batched over leading dimensions.

        Returns the torque clipped to the actuator limits and a mask of saturated joints.
        """
        a_cmd = a_ref + self.kd * (v_ref - v) + self.kp * (q_ref - q)
        Md = self.mass_matrix(M_sparse) + self.h * np.diag(self.damping)
        tau = np.einsum("...ij,...j->...i", Md, a_cmd) + bias[..., self.dadr] - passive[..., self.dadr]
        sat = np.abs(tau) > TAU_MAX
        return np.clip(tau, -TAU_MAX, TAU_MAX), sat
