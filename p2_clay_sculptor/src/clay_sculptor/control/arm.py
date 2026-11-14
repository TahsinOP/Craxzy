"""Arm kinematics: tool-down inverse kinematics on the arm-only model.

The tool tip (site ``tool_tip``, centre of the bottom face of the tool head) is
commanded as a position ``p`` and a yaw ``psi`` about the world z axis, with the
tool axis pointing straight down.  The target rotation of the tip frame is

    R(psi) = [[cos psi,  sin psi,  0],
              [sin psi, -cos psi,  0],
              [0,        0,       -1]]

(columns: tip x axis in the table plane at angle psi, tip y = z x x, tip z down).

Inverse kinematics is damped least squares on the 6D error, with a null-space
term that pulls towards the home configuration so the solution is unique and a
continuous function of the target (7 joints, 6 constraints).

All tool heads are symmetric under a half turn about the tool axis, so the yaw is
used modulo pi and mapped to the branch nearest the home wrist angle.
"""
from __future__ import annotations

import mujoco
import numpy as np

from clay_sculptor.models.scene import HOME_Q, Index, SimCfg, build_model

HOME_YAW = np.pi          # yaw of the tip x axis at HOME_Q (measured, tested in tests/test_arm.py)


def tip_rotation(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def wrap_yaw(psi: float) -> float:
    """Map a yaw given modulo pi to the representative in (HOME_YAW - pi/2, HOME_YAW + pi/2]."""
    return HOME_YAW + (psi - HOME_YAW + np.pi / 2) % np.pi - np.pi / 2


class Kinematics:
    """Forward and inverse kinematics of arm + tool (the tip is the same for every tool)."""

    def __init__(self, sim: SimCfg = SimCfg()):
        self.m = build_model("press", clay=None, sim=sim)
        self.d = mujoco.MjData(self.m)
        self.ix = Index.of(self.m)
        self.lo = self.m.jnt_range[:7, 0].copy()
        self.hi = self.m.jnt_range[:7, 1].copy()
        self._jp = np.zeros((3, self.m.nv))
        self._jr = np.zeros((3, self.m.nv))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[:7] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.ix.tip_site].copy(), self.d.site_xmat[self.ix.tip_site].reshape(3, 3).copy()

    def ik(self, p: np.ndarray, psi: float, q0: np.ndarray | None = None, tol: float = 1e-7,
           max_iter: int = 200, damping: float = 1e-9, null_gain: float = 0.05) -> np.ndarray:
        """Joint angles that put the tip at ``p`` with yaw ``psi`` (tool axis down).

        Raises ``RuntimeError`` if the error does not fall below ``tol`` (metres and
        radians, max norm)."""
        m, d = self.m, self.d
        q = np.array(HOME_Q if q0 is None else q0, dtype=float)
        r_des = tip_rotation(wrap_yaw(psi))
        err = np.zeros(6)
        for _ in range(max_iter):
            d.qpos[:7] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            r = d.site_xmat[self.ix.tip_site].reshape(3, 3)
            err[:3] = p - d.site_xpos[self.ix.tip_site]
            # rotation error as a world-frame rotation vector: R_des R^T = exp([w]x)
            re = r_des @ r.T
            w = np.array([re[2, 1] - re[1, 2], re[0, 2] - re[2, 0], re[1, 0] - re[0, 1]]) / 2.0
            sin_a = np.linalg.norm(w)
            cos_a = (np.trace(re) - 1.0) / 2.0
            if sin_a > 1e-12:
                w *= np.arctan2(sin_a, cos_a) / sin_a
            err[3:] = w
            if np.abs(err).max() < tol:
                return q
            mujoco.mj_jacSite(m, d, self._jp, self._jr, self.ix.tip_site)
            jac = np.vstack([self._jp[:, :7], self._jr[:, :7]])
            jjt = jac @ jac.T + damping * np.eye(6)
            dq = jac.T @ np.linalg.solve(jjt, err)
            # null-space pull towards home, projected with the exact pseudo-inverse so that
            # it leaves the task error untouched to first order
            n0 = null_gain * (HOME_Q - q)
            dq += n0 - np.linalg.pinv(jac) @ (jac @ n0)
            q = np.clip(q + dq, self.lo, self.hi)
        raise RuntimeError(f"IK did not converge: p={p}, psi={psi:.3f}, err={np.abs(err).max():.2e}")
