"""The rally planner: the batter's analytic planner aiming at the far half of the table.

Everything is computed in the hitting arm's own frame, where the arm is at the origin facing +x
and the opponent's base is at x = L.  One call:

1. **Candidates.**  Strike plane x_c (2) × time from the hit to the bounce T (3) × landing point
   on the far half (2 depths × 3 lateral positions): 36 candidates.
2. **Aim.**  For each, the launch velocity that lands the ball on its landing point after T,
   under the flight model (the batter's ``Aimer``, aiming at the table surface).
3. **Check.**  Each launch is rolled out through the bounce with the table-aware flight model,
   all candidates in one vectorised rollout.  A return must clear the net by
   ``net_clearance`` and, after bouncing, cross the opponent's strike plane inside its reach
   (``zone_z``, ``zone_y``).  The score is the squared distance from the opponent's comfortable
   crossing point (``comfort_z``, centre line), plus a large penalty for an infeasible return.
4. **Strike.**  The best ``top_k`` candidates (and the previous plan's) go through the batter's
   impact inversion, inverse kinematics and swing QP, which pick the cheapest feasible swing.  The
   stroke keeps a constant joint velocity from 3 steps before to 4 steps after the planned impact,
   so a contact a step or two early or late still meets a paddle at the planned speed.
"""
from __future__ import annotations

import numpy as np

from batter.control.arm import Arm
from batter.control.planner import Aimer, PlannerParams, SwingPlanner, crossing
from batter.control.traj import Plan
from batter.rally.flight import TableFlight

INFEASIBLE = 50.0


class RallyPlanner(SwingPlanner):
    def __init__(self, arm: Arm, tf: TableFlight, cfg: dict, q_nominal: np.ndarray, restitution: float, top_k: int = 4):
        rc, t = cfg["rally"], cfg["table"]
        # a wider constant-velocity window: with the filter, the contact can come a step or two off the
        # planned one, and the paddle must still be moving at the planned speed.  (Also penalising the
        # face's rotation through the stroke, Arm.strike_velocity's mu_rot, made rallies shorter even with
        # the true ball state: 8.9 returns against 12.0.  It is not used.)
        prm = PlannerParams(strike_x=tuple(rc["strike_x"]), flight_time=tuple(rc["bounce_time"]), restitution=restitution,
                            min_steps=3, n_pre=3, n_post=4)
        super().__init__(arm, tf.fm, cfg["paddle"]["thickness"], prm, q_nominal=q_nominal)
        self.tf, self.rc, self.t = tf, rc, t
        self.L = t["L"]
        self.aimer = Aimer(tf.fm, prm.shoot_step, prm.shoot_iters, ground_z=t["height"] + tf.radius)
        far_end = t["gap"] + t["length"]
        self.lands = np.array([(far_end - dx, dy) for dx in rc["land_x"] for dy in rc["land_y"]])
        self.cands = [(x, T, j) for x in prm.strike_x for T in prm.flight_time for j in range(len(self.lands))]
        self.x_net = t["gap"] + t["length"] / 2
        self.z_net = t["height"] + t["net_height"] + tf.radius + rc["net_clearance"]
        self.x_opp = self.L - float(np.mean(rc["strike_x"]))          # the opponent's strike plane, in this frame
        self.top_k = top_k
        self._prev_cand = None
        self._lock = None                      # candidate index the plan is locked to (late, committed replans)
        self.last_scores: dict = {}

    # ---------------------------------------------------------------- strikes
    def strikes(self, traj: np.ndarray, target_xy=None) -> list[dict]:
        prm, rc = self.prm, self.rc
        cross = {x: crossing(traj, x) for x in prm.strike_x}
        rows = [(ci, x, T, j) for ci, (x, T, j) in enumerate(self.cands) if cross[x] is not None and cross[x][0] >= prm.min_steps
                and (self._lock is None or ci == self._lock)]
        if not rows:
            return []
        p0 = np.array([cross[x][1] for _, x, _, _ in rows])
        w = np.array([cross[x][3] for _, x, _, _ in rows])
        T = np.array([T for _, _, T, _ in rows])
        land = self.lands[[j for _, _, _, j in rows]]
        v_out = self.aimer.solve(p0, w, land, T)
        # roll every launch out through the bounce, all at once
        x0 = np.concatenate([p0, v_out, w], axis=1)
        steps = int(1.2 / self.tf.h)
        path = self.tf.rollout(x0, steps)                             # (steps + 1, C, 9)
        pen = np.zeros(len(rows))
        info = []
        for r in range(len(rows)):
            pr = path[:, r]
            k_net = int(np.argmax(pr[:, 0] >= self.x_net))
            ok = pr[:, 0].max() >= self.x_net and pr[k_net, 2] >= self.z_net
            up = np.flatnonzero(np.diff(np.sign(pr[:, 5])) > 0)       # the bounce reverses v_z
            bounced_far = up.size > 0 and pr[up[0], 0] > self.x_net
            ok &= bounced_far
            c = None
            if ok:
                after = pr[up[0]:]
                beyond = np.flatnonzero(after[:, 0] >= self.x_opp)
                if beyond.size:
                    i = int(beyond[0])
                    a = (self.x_opp - after[i - 1, 0]) / max(after[i, 0] - after[i - 1, 0], 1e-9)
                    c = after[i - 1, :3] + a * (after[i, :3] - after[i - 1, :3])
            if c is None:
                pen[r] = INFEASIBLE
            else:
                z_lo, z_hi = rc["zone_z"]
                inside = z_lo <= c[2] <= z_hi and abs(c[1]) <= rc["zone_y"]
                pen[r] = 20.0 * ((c[2] - rc["comfort_z"]) ** 2 + c[1] ** 2) + (0.0 if inside else INFEASIBLE)
            info.append(c)
        order = list(np.argsort(pen)[: self.top_k])
        prev = [r for r, row in enumerate(rows) if row[0] == self._prev_cand]
        for r in prev:
            if r not in order:
                order.append(r)
        out = []
        for r in order:
            ci, x, _, j = rows[r]
            steps_c, p_b, v_b, _ = cross[x]
            dv = v_out[r] - v_b
            mag = float(np.linalg.norm(dv))
            n = dv / mag
            s = float(v_b @ n) + mag / (1.0 + prm.restitution)
            out.append({"cand": ci, "steps": steps_c, "p_b": p_b, "v_b": v_b, "v_out": v_out[r], "n": n, "speed": s,
                        "pos": p_b - self.offset * n, "penalty": float(pen[r]), "land": self.lands[j],
                        "opp_cross": info[r], "feasible_return": bool(pen[r] < INFEASIBLE)})
        self.last_scores = {"n_candidates": len(rows), "n_feasible": int((pen < INFEASIBLE).sum())}
        return out

    def plan_ctx(self, ctx: dict) -> Plan | None:
        """``ctx["lock"]``: only re-plan the previous plan's candidate (same strike plane, bounce time and
        landing point): a late correction of the same stroke, never a switch to another one."""
        prev = ctx["prev"]
        self._prev_cand = prev.info.get("cand") if prev is not None else None
        self._lock = self._prev_cand if ctx.get("lock") else None
        try:
            return self.plan(ctx["traj"], ctx["k"], ctx["q0"], ctx["v0"], None, prev)
        finally:
            self._lock = None
