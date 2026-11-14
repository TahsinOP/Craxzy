"""Tool action primitives and the joint reference trajectories that execute them.

One primitive for all three tools, with five parameters:

    a = (x, y, psi, depth, length)

* ``(x, y)``: centre of the action on the table plane, relative to the clay centre [m];
* ``psi``: yaw of the tool about the vertical [rad], modulo pi;
* ``depth``: how far the bottom of the tool head goes below the *undeformed* top
  surface [m] (an absolute height, so it does not depend on the current shape);
* ``length``: horizontal travel at depth [m], along the tool's y axis (perpendicular
  to the roller axis), centred on ``(x, y)``.  Zero for the press and the stamp.

Timeline (fixed number of steps per tool, so every rollout of a tool has the same
length): the tool starts at rest at hover height above the start point, descends,
dwells, slides (roller only), ascends and waits for the clay to settle.  Each
moving segment follows a straight Cartesian line with a minimum-jerk time law.

Between two primitives the arm repositions in free space at hover height.  The clay
is at rest and is not touched during that move, so the move is not simulated in
rollouts: the arm state is set directly to the hover pose of the next primitive
(``ClayEnv``).  ``transit_reference`` produces the same move as a real trajectory
for continuous execution and video; ``scripts/m0_bench.py`` measures the difference
between the two (at most 0.13 mm of height after three primitives, 0.03 mm RMS: the
arm arrives with a little residual motion and the clay has half a second more to settle).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clay_sculptor.control.arm import Kinematics
from clay_sculptor.models.scene import ClayCfg

TOOL_ID = {"press": 0, "roller": 1, "stamp": 2}


@dataclass(frozen=True)
class PrimCfg:
    hover: float = 0.015          # tip height above the undeformed top while travelling [m]
    t_down: float = 0.40          # s
    t_dwell: float = 0.10
    t_slide: float = 0.80         # roller only
    t_up: float = 0.30
    t_settle: float = 0.10
    knot: float = 0.004           # spacing of IK knots along a Cartesian segment [m]
    # action bounds (CEM samples and the random data policy stay inside these)
    xy_max: float = 0.055
    depth_min: float = 0.002
    depth_max: float = 0.014
    length_min: float = 0.02
    length_max: float = 0.10

    def nsteps(self, tool: str, dt: float) -> int:
        t = self.t_down + self.t_dwell + self.t_up + self.t_settle + (self.t_slide if tool == "roller" else 0.0)
        return int(round(t / dt))


def bounds(tool: str, cfg: PrimCfg = PrimCfg()) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bounds of the 5 action parameters for ``tool``.

    Yaw ranges use the symmetry of each head: the press and the cross stamp are
    invariant under quarter turns, the roller under half turns."""
    if tool == "roller":
        lo = [-cfg.xy_max, -cfg.xy_max, -np.pi / 2, cfg.depth_min, cfg.length_min]
        hi = [cfg.xy_max, cfg.xy_max, np.pi / 2, cfg.depth_max, cfg.length_max]
    else:
        lo = [-cfg.xy_max, -cfg.xy_max, -np.pi / 4, cfg.depth_min, 0.0]
        hi = [cfg.xy_max, cfg.xy_max, np.pi / 4, cfg.depth_max, 0.0]
    return np.array(lo), np.array(hi)


def yaw_period(tool: str) -> float:
    return np.pi if tool == "roller" else np.pi / 2


def minjerk(n: int) -> np.ndarray:
    """Minimum-jerk path parameter at the ends of ``n`` equal time steps: s(k/n), k = 1..n."""
    tau = np.arange(1, n + 1) / n
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def travel_direction(psi: float) -> np.ndarray:
    """Unit vector of the tool's y axis in the table plane (see ``arm.tip_rotation``)."""
    return np.array([np.sin(psi), -np.cos(psi)])


def waypoints(a: np.ndarray, clay: ClayCfg, cfg: PrimCfg) -> dict:
    """Cartesian key points of primitive ``a`` in world coordinates."""
    x, y, psi, depth, length = a
    c = np.array(clay.centre)
    half = 0.5 * length * travel_direction(psi)
    p0, p1 = c + np.array([x, y]) - half, c + np.array([x, y]) + half
    z_hi, z_lo = clay.top + cfg.hover, clay.top - depth
    return dict(start_hi=np.array([*p0, z_hi]), start_lo=np.array([*p0, z_lo]),
                end_lo=np.array([*p1, z_lo]), end_hi=np.array([*p1, z_hi]), psi=psi)


class Planner:
    """Turns primitives into joint reference sequences ``(nstep, 14)``:
    for every joint ``[q_ref, qd_ref]`` at the *end* of each step."""

    def __init__(self, kin: Kinematics, clay: ClayCfg = ClayCfg(), cfg: PrimCfg = PrimCfg(), dt: float = 0.005):
        self.kin, self.clay, self.cfg, self.dt = kin, clay, cfg, dt

    def _segment(self, q_start: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, psi_a: float, psi_b: float,
                 n: int) -> np.ndarray:
        """Joint positions at the end of each of ``n`` steps along the straight line p_a -> p_b."""
        dist = np.linalg.norm(p_b - p_a) + 0.05 * abs(psi_b - psi_a)
        nk = max(2, int(np.ceil(dist / self.cfg.knot)) + 1)
        sk = np.linspace(0.0, 1.0, nk)
        qk = np.empty((nk, 7))
        q = q_start
        for i, s in enumerate(sk):
            q = self.kin.ik(p_a + s * (p_b - p_a), psi_a + s * (psi_b - psi_a), q0=q)
            qk[i] = q
        s = minjerk(n)
        return np.stack([np.interp(s, sk, qk[:, j]) for j in range(7)], axis=1)

    def hover_q(self, a: np.ndarray) -> np.ndarray:
        w = waypoints(a, self.clay, self.cfg)
        return self.kin.ik(w["start_hi"], w["psi"])

    def reference(self, tool: str, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(q_hover, ctrl)``: the start configuration and the ``(nstep, 14)`` control."""
        cfg, dt = self.cfg, self.dt
        w = waypoints(a, self.clay, cfg)
        n = lambda t: int(round(t / dt))
        q0 = self.kin.ik(w["start_hi"], w["psi"])
        parts = [self._segment(q0, w["start_hi"], w["start_lo"], w["psi"], w["psi"], n(cfg.t_down))]
        parts.append(np.repeat(parts[-1][-1:], n(cfg.t_dwell), axis=0))
        if tool == "roller":
            parts.append(self._segment(parts[-1][-1], w["start_lo"], w["end_lo"], w["psi"], w["psi"], n(cfg.t_slide)))
        parts.append(self._segment(parts[-1][-1], w["end_lo"], w["end_hi"], w["psi"], w["psi"], n(cfg.t_up)))
        parts.append(np.repeat(parts[-1][-1:], n(cfg.t_settle), axis=0))
        q = np.concatenate(parts)
        return q0, self._to_ctrl(q0, q)

    def end_hover_q(self, a: np.ndarray) -> np.ndarray:
        w = waypoints(a, self.clay, self.cfg)
        return self.kin.ik(w["end_hi"], w["psi"])

    def transit_reference(self, a_prev: np.ndarray | None, a_next: np.ndarray, t: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Free-space move at hover height from the end of ``a_prev`` to the start of ``a_next``."""
        w1 = waypoints(a_next, self.clay, self.cfg)
        if a_prev is None:
            q0 = self.kin.ik(w1["start_hi"], w1["psi"])
            return q0, self._to_ctrl(q0, np.repeat(q0[None], int(round(t / self.dt)), axis=0))
        w0 = waypoints(a_prev, self.clay, self.cfg)
        # yaw modulo pi: take the short way between the two wrapped yaws
        from clay_sculptor.control.arm import wrap_yaw

        psi0, psi1 = wrap_yaw(w0["psi"]), wrap_yaw(w1["psi"])
        q0 = self.kin.ik(w0["end_hi"], psi0)
        q = self._segment(q0, w0["end_hi"], w1["start_hi"], psi0, psi1, int(round(t / self.dt)))
        return q0, self._to_ctrl(q0, q)

    def _to_ctrl(self, q0: np.ndarray, q: np.ndarray) -> np.ndarray:
        qd = np.diff(np.vstack([q0[None], q]), axis=0) / self.dt
        ctrl = np.empty((len(q), 14))
        ctrl[:, 0::2], ctrl[:, 1::2] = q, qd
        return ctrl
