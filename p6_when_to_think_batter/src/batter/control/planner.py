"""The expensive planner: interception with an analytic model ("analytic MPC").

Input: the predicted ball trajectory (states at the coming simulator steps, from the EKF
and the flight model), the arm's joint state and the target on the ground.
Output: a joint-space swing (control/traj.py) and what it is built on.

One planner call searches a small grid of strikes.  A candidate is a pair
(strike plane x_c, time of flight T_f of the hit ball):

  1. crossing   where and when the predicted ball crosses x = x_c: time t_i, position
                p_b, velocity v_b, spin w;
  2. aim        the velocity v_out that carries the ball from p_b to the target in T_f
                under the flight model (drag and lift of the spin it keeps), by shooting;
  3. impact     the bat-ball pair is frictionless with restitution e, so the impulse is
                along the face normal n:

                    v_out = v_b - (1 + e) ((v_b - v_c) . n) n
                    =>  n = (v_out - v_b) / |v_out - v_b|,
                        v_c . n = v_b . n + |v_out - v_b| / (1 + e).

                The bat moves along its normal: v_c = s n.  The contact point is on the
                face, so the blade centre is at p_b - (r + thickness / 2) n;
  4. pose       inverse kinematics for the 5-dof strike task, warm-started;
  5. velocity   joint velocity with the sweet spot moving at s along the normal;
  6. swing      the joint QPs from the current state.

The cheapest candidate wins: cost = swing effort + joint-speed margin + what is missed,
with a small bonus for the candidate the previous plan chose (no flip-flopping between
strikes).  Two things can be missed.  If the strike speed is beyond the joint speed limits
it is scaled down (the ball is hit but falls short).  If the strike cannot be reached in
the time left, the swing QP returns the closest approach and its terminal miss; this is
what makes a late replan useful: it still moves the bat towards the corrected strike.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from batter.control.arm import NOMINAL_STRIKE, READY, Arm
from batter.control.traj import Plan, SwingQP
from batter.estim.flight import FlightModel


@dataclass
class PlannerParams:
    strike_x: tuple = (0.56, 0.68)             # candidate strike planes [m]
    flight_time: tuple = (0.75, 0.95, 1.15)    # candidate times of flight of the hit ball [s]
    restitution: float = 0.585                 # effective restitution of the impact model (calibrated, M1)
    shoot_step: float = 0.05                   # coarse step of the aim shooting [s]
    shoot_iters: int = 4
    n_pre: int = 1                             # constant-velocity steps before / after the impact step
    n_post: int = 2
    min_steps: int = 3                         # no strike closer than this many steps
    hysteresis: float = 0.15                   # cost bonus for keeping the previous candidate
    w_miss_q: float = 200.0                    # cost per rad of terminal position miss
    w_miss_v: float = 2.0                      # cost per rad/s of terminal velocity miss
    w_rot: float = 0.3
    mu_rot: float = 0.0                        # penalty on turning the face through the stroke (see Arm.strike_velocity)


def crossing(traj: np.ndarray, x_c: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | None:
    """First crossing of the plane x = x_c by a trajectory (K+1, 9) moving towards -x.

    Returns (steps from traj[0] (fractional), position, velocity, spin), or None.
    The ball moves with v_{i+1} during step i (semi-implicit Euler), linearly.
    """
    x = traj[:, 0]
    below = np.flatnonzero(x[1:] <= x_c)
    if below.size == 0 or x[0] <= x_c:
        return None
    i = int(below[0])
    a = (x[i] - x_c) / max(x[i] - x[i + 1], 1e-12)
    return i + a, traj[i, :3] + a * (traj[i + 1, :3] - traj[i, :3]), traj[i + 1, 3:6].copy(), traj[i, 6:9].copy()


class Aimer:
    """Launch velocity that lands the ball on a ground target after a given time of flight."""

    def __init__(self, fm: FlightModel, step: float = 0.05, iters: int = 4, ground_z: float | None = None):
        self.fm, self.H, self.iters = fm, step, iters
        self.coarse = fm.with_step(step)
        self.ground_z = fm.radius if ground_z is None else ground_z     # ball-centre height at landing

    def fly(self, p0: np.ndarray, v0: np.ndarray, w: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Position after n[i] coarse steps for each row i (batched)."""
        # the simulator's semi-implicit Euler with step h equals, to first order, the same scheme
        # with step H started from v0 - (H - h)/2 a(v0)
        v = v0 - 0.5 * (self.H - self.fm.h) * self.coarse.accel(v0, w)
        x = np.concatenate([p0, v, w], axis=-1)
        out = np.zeros_like(p0)
        for k in range(1, int(n.max()) + 1):
            x = self.coarse.step(x)
            sel = n == k
            if sel.any():
                out[sel] = x[sel, :3]
        return out

    def solve(self, p0: np.ndarray, w: np.ndarray, target_xy: np.ndarray, T: np.ndarray) -> np.ndarray:
        """v_out (C, 3) for start points p0 (C, 3), spins w (C, 3), targets (C, 2) and times T (C,)."""
        n = np.maximum(np.round(T / self.H).astype(int), 1)
        Tn = (n * self.H)[:, None]
        goal = np.concatenate([target_xy, np.full((len(T), 1), self.ground_z)], axis=1)
        v = (goal - p0) / Tn - 0.5 * self.fm.g * Tn                       # drag-free start
        for _ in range(self.iters):
            v = v + (goal - self.fly(p0, v, w, n)) / Tn                   # fixed point with the drag-free Jacobian T I
        return v


def planner_params(cfg: dict, **kw) -> PlannerParams:
    """PlannerParams from the ``planner`` section of the YAML config."""
    pc = dict(cfg.get("planner", {}))
    for key in ("strike_x", "flight_time"):
        if key in pc:
            pc[key] = tuple(pc[key])
    return PlannerParams(**{**pc, **kw})


class SwingPlanner:
    def __init__(self, arm: Arm, fm: FlightModel, bat_thickness: float, params: PlannerParams | None = None,
                 q_nominal: np.ndarray | None = None):
        self.arm, self.fm = arm, fm
        self.q_nominal = NOMINAL_STRIKE if q_nominal is None else np.asarray(q_nominal, float)
        self.prm = params or PlannerParams()
        self.qp = SwingQP(arm.h, arm.q_min, arm.q_max)
        self.aimer = Aimer(fm, self.prm.shoot_step, self.prm.shoot_iters)
        self.offset = fm.radius + bat_thickness / 2.0
        self.cands = [(x, T) for x in self.prm.strike_x for T in self.prm.flight_time]
        self.wall = 0.0            # cumulative wall-clock in plan() [s]
        self.calls = 0
        self.times: list[float] = []   # wall-clock of every call [s]

    # ----------------------------------------------------------------- strikes
    def strikes(self, traj: np.ndarray, target_xy: np.ndarray) -> list[dict]:
        """Steps 1-3 for every candidate: the bat pose and velocity each strike needs."""
        prm = self.prm
        cross = {x: crossing(traj, x) for x in prm.strike_x}
        rows = [(ci, x, T) for ci, (x, T) in enumerate(self.cands) if cross[x] is not None and cross[x][0] >= prm.min_steps]
        if not rows:
            return []
        p0 = np.array([cross[x][1] for _, x, _ in rows])
        w = np.array([cross[x][3] for _, x, _ in rows])
        T = np.array([T for _, _, T in rows])
        v_out = self.aimer.solve(p0, w, np.tile(target_xy, (len(rows), 1)), T)
        out = []
        for r, (ci, x, _) in enumerate(rows):
            steps, p_b, v_b, _ = cross[x]
            dv = v_out[r] - v_b
            mag = float(np.linalg.norm(dv))
            n = dv / mag
            s = float(v_b @ n) + mag / (1.0 + prm.restitution)
            out.append({"cand": ci, "steps": steps, "p_b": p_b, "v_b": v_b, "v_out": v_out[r], "n": n,
                        "speed": s, "pos": p_b - self.offset * n})
        return out

    # -------------------------------------------------------------------- plan
    def plan_ctx(self, ctx: dict) -> Plan | None:
        """Planner interface used by the environment (shared with the learned planner)."""
        return self.plan(ctx["traj"], ctx["k"], ctx["q0"], ctx["v0"], ctx["target"], ctx["prev"])

    def plan(self, traj: np.ndarray, k_now: int, q0: np.ndarray, v0: np.ndarray, target_xy: np.ndarray,
             prev: Plan | None = None) -> Plan | None:
        """One planner call.  ``traj[0]`` is the ball now; returns None if nothing can be planned."""
        t0 = time.perf_counter()
        prm, arm = self.prm, self.arm
        prev_info = prev.info if prev is not None else {}
        warm = prev_info.get("q_cand", {})
        best, best_cost, q_cand = None, np.inf, {}
        for st in self.strikes(traj, target_xy):
            ci = st["cand"]
            q_imp, ep, er = arm.ik(st["pos"], st["n"], warm.get(ci, self.q_nominal), self.q_nominal, w_rot=prm.w_rot)
            q_cand[ci] = q_imp
            if ep > 5e-3 or er > 3e-2:
                continue                                   # the strike pose is out of reach
            v_full = arm.strike_velocity(q_imp, st["n"], st["speed"], mu_rot=prm.mu_rot)
            ratio = float(np.max(np.abs(v_full) / self.qp.v_max))
            scale = min(1.0, 1.0 / max(ratio, 1e-9))       # stay within the joint speed limits
            plan, miss_q, miss_v, effort = self.qp.swing(q0, v0, q_imp, scale * v_full, st["steps"], k0=k_now,
                                                         n_pre=prm.n_pre, n_post=prm.n_post)
            mq, mv = float(np.abs(miss_q).max()), float(np.abs(miss_v).max())
            cost = (effort + 0.5 * (scale * ratio) ** 2 + 10.0 * (1.0 - scale) + prm.w_miss_q * mq + prm.w_miss_v * mv
                    + st.get("penalty", 0.0))
            if ci == prev_info.get("cand"):
                cost -= prm.hysteresis
            if cost < best_cost:
                best_cost = cost
                plan.info = {**st, "scale": scale, "q_imp": q_imp, "v_imp": scale * v_full, "ratio": scale * ratio,
                             "cost": cost, "k_impact": k_now + st["steps"], "miss_q": mq, "miss_v": mv}
                best = plan
        if best is not None:
            best.info["q_cand"] = q_cand
        dt = time.perf_counter() - t0
        self.wall += dt
        self.calls += 1
        self.times.append(dt)
        if best is not None:
            best.info["wall"] = dt
        return best
