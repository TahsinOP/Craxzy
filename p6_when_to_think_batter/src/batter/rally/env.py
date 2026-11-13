"""Batched two-arm rally on ``mujoco.batch`` (CPU; speculative contacts).

Per world: arm A, arm B, one ball, served by the feeder to arm A.  Per simulator step:

  1. plan     : the receiving arm (the one the ball is heading to) runs the rally planner on the
                EKF forecast, transformed into its own frame; first plan after 5 fixes, then
                at ``rate_hz`` (1000: every step).  Within ``commit_time`` of the cached swing's
                impact the stroke is committed: replans may only correct the same candidate
                (strike plane, bounce time, landing point), never switch to another (strike
                planes are 60 ms apart at ping-pong speeds, and a switch that late pulls the
                paddle off a swing that is about to land);
  2. track    : both arms track their cached joint trajectories with discrete computed torque;
                an arm that has hit plays out its follow-through and then returns to its ready pose;
  3. referee  : contact sensors decide the rally (below);
  4. sense    : the delayed camera fixes feed one world-frame EKF whose process model bounces on
                the table; after a paddle hit it restarts from the first fix taken after the hit,
                keeping its spin estimate (the paddle is frictionless).

Confidence gate.  The filter is unsure of the ball's vertical speed for a few steps after each
bounce (the bounce's timing is a discrete event; see rally/flight.py).  Replanning on those
transient forecasts made the arm abandon good strokes for ones it could no longer reach; the
arm therefore keeps its cached swing while the filter's v_z standard deviation exceeds
``replan_vz_sd``.  On the same 48 serves this doubled the mean rally length (4.7 -> 9.3 returns).

Referee.  A return counts when the receiver hits the ball after it bounced exactly once on the
receiver's half.  A rally ends when the ball touches the net, lands on the hitter's own half,
bounces twice on the receiver's half (a miss), is hit before it bounced, hits the floor, passes
an arm, or when ``max_hits`` returns or ``max_time`` are reached.
"""
from __future__ import annotations

import mujoco
import numpy as np
from mujoco import batch as mjbatch

from batter.control.arm import Arm, dense_mass_index, discrete_pd_gains
from batter.control.traj import Plan
from batter.estim.ekf import BallEKF, EKFParams, crossing_batch
from batter.estim.flight import FlightModel
from batter.models.scene import TAU_MAX
from batter.rally import poses
from batter.rally.flight import BounceModel, TableFlight
from batter.rally.planner import RallyPlanner
from batter.rally.scene import RallyIndex, build_arm_model, build_rally_model, to_arm_frame

STAGE_NONE, STAGE_VEL = int(mujoco.mjtStage.mjSTAGE_NONE), int(mujoco.mjtStage.mjSTAGE_VEL)
REASONS = ["max", "net", "own_side", "missed", "no_bounce", "floor", "passed", "time", "double_hit"]


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, u = q[:, 0:1], q[:, 1:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class RallyEnv:
    def __init__(self, cfg: dict, nsim: int, rate_hz: float = 1000.0, seed: int = 0, nthread: int = 0,
                 sim_override: dict | None = None, restitution: float | None = None, oracle: bool = False,
                 omega: float = 30.0, commit_time: float | None = None, late_corrections: bool = True,
                 max_vz_sd: float | None = None):
        self.cfg, self.B, self.rate, self.oracle = cfg, nsim, rate_hz, oracle
        self.commit_time = cfg["rally"]["commit_time"] if commit_time is None else commit_time   # closer than this to the impact [s] ...
        self.late_corrections = late_corrections  # ... replan only the same stroke (True) or not at all (False)
        # no replanning while the filter's v_z standard deviation exceeds this [m/s] (np.inf: always replan)
        self.max_vz_sd = cfg["rally"]["replan_vz_sd"] if max_vz_sd is None else max_vz_sd
        self.model = build_rally_model(cfg, sim_override)
        m = self.model
        self.h = m.opt.timestep
        self.ix = RallyIndex.of(m)
        self.L = cfg["table"]["L"]
        self.x_net = cfg["table"]["gap"] + cfg["table"]["length"] / 2
        fm = FlightModel.identify(m)
        self.bm = BounceModel.identify(m, fm, cfg["table"])
        self.tf = TableFlight(fm, self.bm)                 # the table is symmetric: the same model in both arms' frames
        self.arm = Arm(build_arm_model(cfg))
        P = poses.load(self.arm, cfg)
        self.q_ready, self.q_nominal = np.array(P["ready"]), np.array(P["nominal"])
        e = cfg["paddle"].get("restitution_model", cfg["paddle"]["restitution"]) if restitution is None else restitution
        self.planner = RallyPlanner(self.arm, self.tf, cfg, self.q_nominal, e)
        self.qp = self.planner.qp
        self.strike_x = float(np.mean(cfg["rally"]["strike_x"]))
        self.rc = cfg["rally"]
        # per-arm dynamics in the two-arm model
        full = dense_mass_index(m)
        self.midx = [full[np.ix_(d, d)] for d in self.ix.dadr]
        self.damping = [m.dof_damping[d].copy() for d in self.ix.dadr]
        self.kp, self.kd = discrete_pd_gains(omega, self.h)
        self.ekf = BallEKF(self.tf, nsim, cfg["measurement"], EKFParams(prior_v=(-4.0, 0.0, 0.5), prior_sigma_v=(2.0, 0.8, 1.5),
                                                                          prior_sigma_w=(40.0, 40.0, 40.0)))
        self.nd = self.ekf.nd
        self.meas = cfg["measurement"]
        self.rng = np.random.default_rng(seed)
        b = mjbatch.Batch(m, nsim=nsim, nthread=nthread, persistent=True)
        self.batch = b
        self.qpos, self.qvel, self.ctrl = b.bind("qpos"), b.bind("qvel"), b.bind("ctrl")
        self.M, self.bias, self.passive = b.bind("M"), b.bind("qfrc_bias"), b.bind("qfrc_passive")
        self.sens = b.bind("sensordata")
        s = self.ix.sensors
        self._s = {k: s[k].start for k in ("hit_a", "hit_b", "table", "net", "ground")}
        self._cam = s["ball_cam"]

    # ------------------------------------------------------------------ reset
    def reset(self, serves: np.ndarray) -> None:
        B, ix = self.B, self.ix
        assert len(serves) == B
        self.batch.reset(stage=STAGE_NONE)
        for a in range(2):
            self.qpos[:, ix.qadr[a]] = self.q_ready
        a, d = ix.ball_qadr, ix.ball_dadr
        self.qpos[:, a:a + 3] = serves[:, 0:3]
        self.qpos[:, a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]
        self.qvel[:] = 0.0
        self.qvel[:, d:d + 3] = serves[:, 3:6]
        self.qvel[:, d + 3:d + 6] = serves[:, 6:9]
        self.ctrl[:] = 0.0
        self.batch.forward(stage=STAGE_VEL)
        self.k = 0
        self.plans: list[list[Plan | None]] = [[None] * B, [None] * B]
        self.swing = np.zeros((2, B), bool)              # the arm's cached plan is a swing at the current ball
        self.turn = np.zeros(B, int)                     # receiving arm
        self.bounces = np.zeros(B, int)                  # bounces on the receiver's half since the last hit
        self.n_hits = np.zeros(B, int)
        self.hits_by = np.zeros((2, B), int)
        self.last_hit_k = np.full((2, B), -100)
        self.track_k0 = np.zeros(B, int)                 # fixes showing steps before this are of the previous flight
        self.finished = np.zeros(B, bool)
        self.reason = np.full(B, "", dtype=object)
        self.fault_by = np.full(B, -1)                   # which arm's shot ended the rally (-1: none / the feeder)
        self.n_plans = np.zeros(B, int)
        self.plan_wall = np.zeros(B)
        self.next_t = np.full(B, np.inf)
        self.prev_contact = {k: np.zeros(B, bool) for k in self._s}
        self.hit_log: list[list[dict]] = [[] for _ in range(B)]
        self.ekf.reset()
        self.ekf.restart(np.ones(B, bool), np.tile([-4.0, 0.0, 0.5], (B, 1)), keep_spin=False)
        self._truth = []
        self._sense()

    # ------------------------------------------------------------------ state
    def ball_state(self) -> np.ndarray:
        a, d = self.ix.ball_qadr, self.ix.ball_dadr
        w = quat_rotate(self.qpos[:, a + 3:a + 7], self.qvel[:, d + 3:d + 6])
        return np.concatenate([self.qpos[:, a:a + 3], self.qvel[:, d:d + 3], w], axis=1)

    def _end(self, i: int, reason: str, by: int = -1) -> None:
        if not self.finished[i]:
            self.finished[i] = True
            self.reason[i] = reason
            self.fault_by[i] = by

    def _referee(self, xs: np.ndarray) -> None:
        B = self.B
        now = {k: self.sens[:, adr] > 0.5 for k, adr in self._s.items()}
        rise = {k: now[k] & ~self.prev_contact[k] for k in now}
        self.prev_contact = now
        for i in np.flatnonzero(~self.finished):
            r = self.turn[i]                              # receiver
            hitter = 1 - r
            if rise["net"][i]:
                self._end(i, "net", hitter)
                continue
            if rise["table"][i]:
                side = 0 if xs[i, 0] < self.x_net else 1
                if side != r:
                    self._end(i, "own_side", hitter)
                    continue
                self.bounces[i] += 1
                if self.bounces[i] >= 2:
                    self._end(i, "missed", r)
                    continue
            for arm, key in ((0, "hit_a"), (1, "hit_b")):
                if not rise[key][i]:
                    continue
                if arm != r:
                    if self.k - self.last_hit_k[arm, i] > 10:
                        self._end(i, "double_hit", arm)
                    continue
                if self.bounces[i] != 1:
                    self._end(i, "no_bounce", hitter)
                    break
                # a valid return
                self.n_hits[i] += 1
                self.hits_by[arm, i] += 1
                self.last_hit_k[arm, i] = self.k
                p = self.plans[arm][i]
                self.hit_log[i].append({"arm": arm, "k": int(self.k), "planned_k": float(p.info.get("k_impact", np.nan)) if p is not None else np.nan,
                                        "n_plans": int(self.n_plans[i])})
                self.turn[i] = 1 - arm
                self.bounces[i] = 0
                self.swing[arm, i] = False
                self.track_k0[i] = self.k + 1
                self.next_t[i] = np.inf
                pv = np.array([4.0, 0.0, 1.0]) if arm == 0 else np.array([-4.0, 0.0, 1.0])
                mask = np.zeros(B, bool)
                mask[i] = True
                self.ekf.restart(mask, np.tile(pv, (B, 1)), keep_spin=True)
                if self.n_hits[i] >= self.rc["max_hits"]:
                    self._end(i, "max")
                break
            if self.finished[i]:
                continue
            if rise["ground"][i]:
                self._end(i, "missed" if self.bounces[i] == 1 else "floor", r if self.bounces[i] == 1 else hitter)
            elif xs[i, 0] < -0.5 or xs[i, 0] > self.L + 0.5:
                self._end(i, "passed", r)
            elif self.k * self.h >= self.rc["max_time"]:
                self._end(i, "time")

    def _sense(self) -> None:
        B = self.B
        xs = self.ball_state()
        if self.k > 0:
            self._referee(xs)
        # camera: delayed, noisy fixes; only fixes taken after the last hit feed the current track
        delayed = self.sens[:, self._cam]
        sigma = self.ekf.meas_sigma(delayed)
        z = delayed + self.rng.normal(size=(B, 3)) * sigma[:, None]
        valid = (self.k >= self.nd) & (self.k - self.nd >= self.track_k0) & (self.rng.random(B) >= self.meas["dropout"]) & ~self.finished
        if self.oracle:
            self._truth.append(xs.copy())
            valid = (self.k >= self.nd) & (self.k - self.nd >= self.track_k0) & ~self.finished
            z = delayed
        self.ekf.step(z, valid)
        if self.oracle:
            j = max(len(self._truth) - 1 - self.nd, 0)
            ok = self.ekf.started
            self.ekf.x[ok] = self._truth[j][ok]
            self.ekf.P[ok] = 1e-10 * np.eye(9)
        K = self.nd + 110
        self.traj, _ = self.ekf.forecast(K)
        # the forecast in each world's receiver frame, and the crossing of its strike plane
        tr = self.traj.copy()
        bw = self.turn == 1
        if bw.any():
            tr[:, bw] = to_arm_frame(tr[:, bw], 1, self.L)
        self.traj_r = tr
        c = crossing_batch(tr, np.zeros(tr.shape[:2] + (3, 3)), self.strike_x)
        self.tau = (c["steps"] - self.nd) * self.h
        self.active = self.ekf.started & (self.ekf.n_fix >= 5) & c["found"] & (self.tau > 0.04) & ~self.finished

    # ------------------------------------------------------------------- plan
    def _ref(self, arm: int, i: int, k: int):
        p = self.plans[arm][i]
        if p is None:
            return self.q_ready, np.zeros(7), np.zeros(7)
        return p.ref(k)

    def _plan(self, i: int, lock: bool = False) -> None:
        r = self.turn[i]
        q0, v0, _ = self._ref(r, i, self.k)
        prev = self.plans[r][i] if self.swing[r, i] else None
        ctx = {"traj": self.traj_r[self.nd:, i], "k": self.k, "q0": q0.copy(), "v0": v0.copy(), "target": None, "prev": prev,
               "lock": lock}
        w0 = self.planner.wall
        plan = self.planner.plan_ctx(ctx)
        self.plan_wall[i] += self.planner.wall - w0
        self.n_plans[i] += 1
        if plan is not None:
            self.plans[r][i] = plan
            self.swing[r, i] = True

    def _return_to_ready(self, arm: int, i: int) -> None:
        """After a stroke has played out: a smooth return to the ready pose (0.4 s)."""
        q, v, _ = self._ref(arm, i, self.k)
        a, _, _, _ = self.qp.approach(q, v, self.q_ready, np.zeros(7), 40)
        A = np.vstack([a, np.zeros((1, 7))])
        V = np.vstack([v, v + self.h * np.cumsum(A, axis=0)])
        Q = np.vstack([q, q + self.h * np.cumsum(V[1:], axis=0)])
        self.plans[arm][i] = Plan(q=Q, v=V, a=A, k0=self.k, info={"return": True})

    # ------------------------------------------------------------------- step
    def step(self) -> bool:
        B, ix = self.B, self.ix
        t = self.k * self.h
        period = 0.0 if self.rate >= 100.0 else 1.0 / self.rate
        for i in np.flatnonzero(self.active):
            r = self.turn[i]
            p = self.plans[r][i]
            committed = self.swing[r, i] and p is not None and (p.info.get("k_impact", np.inf) - self.k) * self.h < self.commit_time
            if committed and not self.late_corrections:
                continue          # committed: too close to the cached swing's impact to change it
            if self.swing[r, i] and self.ekf.P[i, 5, 5] > self.max_vz_sd ** 2:
                continue          # the filter is just through a bounce and unsure of v_z: keep the cached swing
            if not self.swing[r, i]:
                self._plan(i)
                self.next_t[i] = t + period
            elif t >= self.next_t[i] - 1e-9:
                self._plan(i, lock=committed)
                self.next_t[i] = max(self.next_t[i] + period, t + 1e-9) if period > 0 else t
        for arm in range(2):
            for i in range(B):
                p = self.plans[arm][i]
                if p is not None and not self.swing[arm, i] and not p.info.get("return") and self.k >= p.k0 + len(p.a):
                    self._return_to_ready(arm, i)
                if self.swing[arm, i] and self.turn[i] != arm and p is not None and self.k >= p.k0 + len(p.a):
                    self.swing[arm, i] = False
        for arm in range(2):
            q_ref, v_ref, a_ref = np.empty((B, 7)), np.empty((B, 7)), np.empty((B, 7))
            for i in range(B):
                q_ref[i], v_ref[i], a_ref[i] = self._ref(arm, i, self.k)
            q, v = self.qpos[:, ix.qadr[arm]], self.qvel[:, ix.dadr[arm]]
            a_cmd = a_ref + self.kd * (v_ref - v) + self.kp * (q_ref - q)
            Mf = np.where(self.midx[arm] >= 0, self.M[:, np.maximum(self.midx[arm], 0)], 0.0) + self.h * np.diag(self.damping[arm])
            tau = np.einsum("bij,bj->bi", Mf, a_cmd) + self.bias[:, ix.dadr[arm]] - self.passive[:, ix.dadr[arm]]
            self.ctrl[:, ix.act[arm]] = np.clip(tau, -TAU_MAX, TAU_MAX)
        self.batch.step(stage=STAGE_VEL)
        self.k += 1
        self._sense()
        return bool(self.finished.all())

    def outcome(self) -> dict:
        return {"n_hits": self.n_hits.copy(), "hits_a": self.hits_by[0].copy(), "hits_b": self.hits_by[1].copy(),
                "reason": self.reason.copy(), "fault_by": self.fault_by.copy(), "n_plans": self.n_plans.copy(),
                "plan_wall": self.plan_wall.copy(), "duration": np.full(self.B, self.k * self.h)}


def run_rally(env: RallyEnv, serves: np.ndarray, frame_cb=None) -> dict:
    env.reset(serves)
    done = False
    while not done:
        if frame_cb is not None:
            frame_cb(env)
        done = env.step()
    return env.outcome()
