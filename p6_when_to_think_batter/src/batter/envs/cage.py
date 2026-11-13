"""Batched batting cage on ``mujoco.batch`` (multi-world CPU; speculative contacts).

One world = Panda + bat + one pitched ball + one target.  Per simulator step (10 ms):

  1. sense    : the delayed camera sensor gives a position fix (noise grows with range,
                frames drop out); the EKF fuses it and forecasts the ball;
  2. decide   : the caller (a trigger) says which worlds re-run the planner; the first
                plan of a pitch is made by the environment after ``first_fixes`` fixes;
  3. plan     : the expensive planner (control/planner.py or learn/wm_planner.py) builds a
                swing from the forecast; its wall-clock time is recorded per call;
  4. track    : discrete computed torque on the cached plan;
  5. surprise : spin change / gust for M4 (same pitches for every controller);
  6. step the batch.

Outcome of a pitch: did the bat touch the ball, and where does the ball come down.  The
landing point is extrapolated with the exact flight model from the ball's state
``post_steps`` after the contact (tests/test_env.py checks the extrapolated flight against the
simulated one).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mujoco import batch as mjbatch

from batter.control.arm import READY, Arm
from batter.control.planner import PlannerParams, SwingPlanner, planner_params
from batter.control.traj import Plan
from batter.estim.ekf import BallEKF, EKFParams, crossing_batch
from batter.estim.flight import FlightModel
from batter.models.pitch import Pitches
from batter.models.scene import build_cage_model

STAGE_NONE, STAGE_VEL = int(mujoco.mjtStage.mjSTAGE_NONE), int(mujoco.mjtStage.mjSTAGE_VEL)
KMAX = 160           # stored prediction length per plan (steps)
NFIX = 16            # camera fixes kept per world (the learned planner's observation)


@dataclass
class EnvParams:
    first_fixes: int = 5          # the first plan is made once the filter has this many fixes
    min_tau: float = 0.05         # no replanning closer than this to the predicted crossing [s]
    post_steps: int = 6           # steps simulated after the bat contact before the landing is extrapolated
    chi2_window: int = 4          # fixes averaged in the plan-innovation statistic
    latency_steps: int = 0        # a plan requested at step k takes effect at step k + latency_steps
    oracle: bool = False          # give the planner the true ball state (diagnostics)
    max_time: float = 1.10        # episode limit [s]
    horizon: float = 0.86         # forecast horizon at release [s] (the slowest pitch takes 0.78 s)
    q_start: tuple | None = None  # initial arm configuration (default: the ready pose)
    swing: bool = True            # False: the arm holds q_start (M0: a static bat)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (B, 3) by unit quaternions q (B, 4) (w, x, y, z)."""
    w, u = q[:, 0:1], q[:, 1:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_rotate(q * np.array([1.0, -1.0, -1.0, -1.0]), v)


class BatEnv:
    def __init__(self, cfg: dict, nsim: int, params: EnvParams | None = None, planner_params_: PlannerParams | None = None,
                 ekf_params: EKFParams | None = None, seed: int = 0, nthread: int = 0, sim_override: dict | None = None,
                 planner=None):
        self.cfg, self.B = cfg, nsim
        self.prm = params or EnvParams()
        self.model = build_cage_model(cfg, sim_override)
        self.h = self.model.opt.timestep
        self.arm = Arm(self.model)
        self.ix = self.arm.ix
        self.fm = FlightModel.identify(self.model)
        self.planner = planner or SwingPlanner(self.arm, self.fm, cfg["bat"]["thickness"], planner_params_ or globals()["planner_params"](cfg))
        self.ekf = BallEKF(self.fm, nsim, cfg["measurement"], ekf_params)
        self.nd = self.ekf.nd
        self.strike_x = cfg["pitch"]["strike_x"]
        self.meas = cfg["measurement"]
        self.tcfg = cfg["target"]
        self.rng = np.random.default_rng(seed)
        b = mjbatch.Batch(self.model, nsim=nsim, nthread=nthread, persistent=True)
        self.batch = b
        self.qpos, self.qvel, self.ctrl = b.bind("qpos"), b.bind("qvel"), b.bind("ctrl")
        self.xfrc, self.mocap = b.bind("xfrc_applied"), b.bind("mocap_pos")
        self.M, self.bias, self.passive = b.bind("M"), b.bind("qfrc_bias"), b.bind("qfrc_passive")
        self.sens, self.site_xpos, self.site_xmat = b.bind("sensordata"), b.bind("site_xpos"), b.bind("site_xmat")
        s = self.ix.sensors
        self._cam, self._hit = s["ball_cam"], s["hit"].start

    # ------------------------------------------------------------------ reset
    def reset(self, P: Pitches) -> dict:
        B, ix = self.B, self.ix
        assert len(P) == B
        self.P = P
        self.batch.reset(stage=STAGE_NONE)
        self.q_hold = READY if self.prm.q_start is None else np.asarray(self.prm.q_start, float)
        self.qpos[:, ix.arm_qadr] = self.q_hold
        self.qpos[:, ix.ball_qadr:ix.ball_qadr + 3] = P.x0[:, 0:3]
        self.qpos[:, ix.ball_qadr + 3:ix.ball_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.qvel[:] = 0.0
        self.qvel[:, ix.ball_dadr:ix.ball_dadr + 3] = P.x0[:, 3:6]
        self.qvel[:, ix.ball_dadr + 3:ix.ball_dadr + 6] = P.x0[:, 6:9]      # identity orientation: local = world
        self.ctrl[:] = 0.0
        self.xfrc[:] = 0.0
        self.mocap[:, ix.target_mocap, 0:2] = P.target
        self.batch.forward(stage=STAGE_VEL)

        self.k = 0
        self.plans: list[Plan | None] = [None] * B
        self.pending: list[tuple[int, Plan] | None] = [None] * B
        self.plan_k = np.zeros(B, int)
        self.plan_c = np.zeros((B, 3))               # crossing point the plan was built on
        self.plan_cov = np.zeros((B, 2, 2))
        self.plan_t = np.zeros(B)                    # absolute predicted crossing time [s]
        self.plan_j0 = np.zeros(B, int)              # filter-time step of the stored prediction's first element
        self.plan_len = np.zeros(B, int)
        self.plan_p = np.zeros((B, KMAX, 3))         # predicted ball positions the plan relies on
        self.plan_S = np.zeros((B, KMAX, 3, 3))      # and their covariances
        self.chi2_hist = np.zeros((B, self.prm.chi2_window))
        self.chi2_n = np.zeros(B, int)
        self.n_plans = np.zeros(B, int)
        self.n_fail = np.zeros(B, int)               # planner calls that returned no feasible swing
        self.plan_wall = np.zeros(B)
        self.plan_steps: list[list[int]] = [[] for _ in range(B)]
        self.hit = np.zeros(B, bool)
        self.hit_k = np.full(B, -1)
        self.finished = np.zeros(B, bool)
        self.x_after = np.full((B, 9), np.nan)       # ball state when the world finished
        self.finish_k = np.full(B, -1)               # ... and the step at which it did
        self.v_pre = np.zeros((B, 3))                # ball velocity one step before the contact
        self._v_prev = P.x0[:, 3:6].copy()
        self.v_post = np.full((B, 3), np.nan)        # ball velocity on the first step clear of the bat
        self.hit_off = np.full((B, 2), np.nan)       # contact offset from the sweet spot along (width, length)
        self.sat_steps = np.zeros(B, int)
        self.fix_p = np.zeros((B, NFIX, 3))          # the last NFIX camera fixes, newest first ...
        self.fix_k = np.full((B, NFIX), -1)          # ... and the simulator step each one shows (-1: empty)
        self.ekf.reset()
        self._sense()
        return self._obs()

    # ------------------------------------------------------------------ state
    def ball_state(self) -> np.ndarray:
        """True ball state (B, 9) with the spin in the world frame."""
        a, d = self.ix.ball_qadr, self.ix.ball_dadr
        w = quat_rotate(self.qpos[:, a + 3:a + 7], self.qvel[:, d + 3:d + 6])
        return np.concatenate([self.qpos[:, a:a + 3], self.qvel[:, d:d + 3], w], axis=1)

    def _sense(self) -> None:
        B, prm = self.B, self.prm
        touching = self.sens[:, self._hit] > 0.5
        new = touching & ~self.hit
        if new.any():
            self.hit_k[new] = self.k
            self.v_pre[new] = self._v_prev[new]
            xs = self.ball_state()
            rel = xs[new, 0:3] - self.site_xpos[new, self.ix.sweet_site]
            R = self.site_xmat[new, self.ix.sweet_site].reshape(-1, 3, 3)
            self.hit_off[new] = np.einsum("bij,bi->bj", R, rel)[:, 1:3]
        self.hit |= touching
        xs = self.ball_state()
        self._v_prev = xs[:, 3:6].copy()
        left = self.hit & ~touching & ~np.isfinite(self.v_post[:, 0])
        self.v_post[left] = xs[left, 3:6]            # first step clear of the bat
        fin = ~self.finished & ((self.hit & (self.k >= self.hit_k + prm.post_steps) & ~touching)
                                | (xs[:, 0] < -0.6) | (xs[:, 2] < self.fm.radius + 0.005) | (self.k * self.h >= prm.max_time))
        self.x_after[fin] = xs[fin]
        self.finish_k[fin] = self.k
        self.finished |= fin

        # camera: a delayed position fix (MuJoCo sensor delay), noisy, with dropouts
        truth_delayed = self.sens[:, self._cam]
        sigma = self.ekf.meas_sigma(truth_delayed)
        z = truth_delayed + self.rng.normal(size=(B, 3)) * sigma[:, None]
        seen_hit = self.hit & (self.k - self.nd >= self.hit_k)
        valid = (self.k >= self.nd) & (self.rng.random(B) >= self.meas["dropout"]) & ~seen_hit & (truth_delayed[:, 0] > 0.25)
        if self.k == self.nd:
            valid[:] = True
        if prm.oracle:
            valid = (self.k >= self.nd) & ~seen_hit
            z = truth_delayed
        self.z, self.z_valid = z, valid
        if valid.any():
            self.fix_p[valid] = np.roll(self.fix_p[valid], 1, axis=1)
            self.fix_k[valid] = np.roll(self.fix_k[valid], 1, axis=1)
            self.fix_p[valid, 0] = z[valid]
            self.fix_k[valid, 0] = self.k - self.nd
        self.ekf.step(z, valid)
        if prm.oracle and self.k >= self.nd:
            # the filter state is replaced by the truth n_d steps ago (kept exact by storing the history)
            self._truth_hist.append(xs.copy())
            self.ekf.x[:] = self._truth_hist[max(len(self._truth_hist) - 1 - self.nd, 0)]
            self.ekf.P[:] = 1e-10 * np.eye(9)
        elif prm.oracle:
            self._truth_hist = getattr(self, "_truth_hist", []) if self.k else []
            self._truth_hist.append(xs.copy())

        K = min(self.nd + max(8, int(round(prm.horizon / self.h)) - self.k), KMAX - 1)
        self.traj, self.Ppp = self.ekf.forecast(K)
        c = crossing_batch(self.traj, self.Ppp, self.strike_x)
        self.cross = c
        self.tau = (c["steps"] - self.nd) * self.h
        self.active = (self.ekf.started & (self.ekf.n_fix >= prm.first_fixes) & c["found"] & (self.tau > prm.min_tau)
                       & ~self.hit & ~self.finished & prm.swing)

        # innovation of the fix against the prediction the cached plan was built on
        has_plan = np.array([p is not None for p in self.plans])
        idx = (self.k - self.nd) - self.plan_j0
        ok = valid & has_plan & (idx > 0) & (idx < self.plan_len)
        if ok.any():
            w = np.flatnonzero(ok)
            nu = z[w] - self.plan_p[w, idx[w]]
            S = self.plan_S[w, idx[w]] + (sigma[w] ** 2)[:, None, None] * np.eye(3)
            chi2 = np.einsum("bi,bij,bj->b", nu, np.linalg.inv(S), nu)
            self.chi2_hist[w] = np.roll(self.chi2_hist[w], 1, axis=1)
            self.chi2_hist[w, 0] = chi2
            self.chi2_n[w] += 1

    def _obs(self) -> dict:
        c = self.cross
        has_plan = np.array([p is not None for p in self.plans])
        found = c["found"]
        t_cross = self.k * self.h + self.tau
        drift = np.where(has_plan & found, np.linalg.norm((c["p"] - self.plan_c)[:, 1:], axis=1), 0.0)
        dt_drift = np.where(has_plan & found, np.abs(t_cross - self.plan_t), 0.0)
        tr_now = np.where(found, np.trace(np.nan_to_num(c["cov"]), axis1=1, axis2=2), 0.0)
        tr_plan = np.where(has_plan, np.trace(self.plan_cov, axis1=1, axis2=2), 0.0)
        n = np.minimum(self.chi2_n, self.prm.chi2_window)
        chi2 = np.where(n > 0, self.chi2_hist.sum(axis=1) / np.maximum(n, 1), 0.0)
        return {"active": self.active.copy(), "has_plan": has_plan, "need_first": self.active & ~has_plan,
                "tau": np.where(found, self.tau, 0.0), "since": (self.k - self.plan_k) * self.h,
                "chi2": chi2, "chi2_n": n, "nis": self.ekf.nis.copy(), "drift": drift, "time_drift": dt_drift,
                "speed": np.where(found, np.linalg.norm(c["v"], axis=1), 0.0), "tr_now": tr_now, "tr_plan": tr_plan,
                "k": self.k, "h": self.h}

    # ------------------------------------------------------------------- plan
    def _reference(self, i: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.plans[i]
        if p is None:
            return self.q_hold, np.zeros(7), np.zeros(7)
        return p.ref(k)

    def _plan(self, i: int) -> None:
        L = self.prm.latency_steps
        k_start = self.k + L
        prev = self.plans[i]
        if prev is None:
            q0, v0 = self.qpos[i, self.ix.arm_qadr].copy(), self.qvel[i, self.ix.arm_dadr].copy()
        else:
            q0, v0, _ = prev.ref(k_start)            # continue from the reference: no jump on a replan
            q0, v0 = q0.copy(), v0.copy()
        ctx = {"traj": self.traj[self.nd + L:, i], "k": k_start, "q0": q0, "v0": v0, "target": self.P.target[i],
               "prev": prev, "env": self, "world": i}
        w0 = self.planner.wall
        plan = self.planner.plan_ctx(ctx)
        self.plan_wall[i] += self.planner.wall - w0
        self.n_plans[i] += 1
        self.plan_steps[i].append(self.k)
        # what the plan relies on, for the triggers (stored even if the planner found nothing new)
        c = self.cross
        self.plan_k[i] = self.k
        self.plan_c[i] = c["p"][i]
        self.plan_cov[i] = np.nan_to_num(c["cov"][i])
        self.plan_t[i] = self.k * self.h + self.tau[i]
        n = min(self.traj.shape[0], KMAX)
        self.plan_j0[i] = self.k - self.nd
        self.plan_len[i] = n
        self.plan_p[i, :n] = self.traj[:n, i, 0:3]
        self.plan_S[i, :n] = self.Ppp[:n, i]
        self.chi2_hist[i] = 0.0
        self.chi2_n[i] = 0
        if plan is None:
            self.n_fail[i] += 1
            return
        if L == 0:
            self.plans[i] = plan
        else:
            self.pending[i] = (k_start, plan)

    # ------------------------------------------------------------------- step
    def step(self, replan: np.ndarray | None = None) -> tuple[dict, bool]:
        """Replan where asked (and where a first plan is due), apply control, advance one step."""
        B, ix, P = self.B, self.ix, self.P
        has_plan = np.array([p is not None for p in self.plans])
        busy = np.array([p is not None for p in self.pending])
        want = self.active & ~busy & (~has_plan | (np.zeros(B, bool) if replan is None else np.asarray(replan, bool)))
        for i in np.flatnonzero(want):
            self._plan(i)
        for i in np.flatnonzero(busy):
            if self.pending[i][0] <= self.k:
                self.plans[i] = self.pending[i][1]
                self.pending[i] = None
        q_ref, v_ref, a_ref = np.empty((B, 7)), np.empty((B, 7)), np.empty((B, 7))
        for i in range(B):
            q_ref[i], v_ref[i], a_ref[i] = self._reference(i, self.k)
        tau, sat = self.arm.torque(self.M, self.bias, self.passive, self.qpos[:, ix.arm_qadr], self.qvel[:, ix.arm_dadr],
                                   q_ref, v_ref, a_ref)
        self.sat_steps += sat.any(axis=1) & ~self.hit
        self.ctrl[:, :7] = tau
        # surprises
        chg = (P.s_kind == 1) & (P.s_step == self.k)
        if chg.any():
            a, d = ix.ball_qadr, ix.ball_dadr
            self.qvel[chg, d + 3:d + 6] = quat_rotate_inv(self.qpos[chg, a + 3:a + 7], P.s_spin[chg])
        gust = (P.s_kind == 2) & (self.k >= P.s_step) & (self.k < P.s_step + P.s_len)
        self.xfrc[:, ix.ball_body, 0:3] = np.where(gust[:, None], P.s_force, 0.0)
        self.batch.step(stage=STAGE_VEL)
        self.k += 1
        self._sense()
        return self._obs(), bool(self.finished.all())

    # ---------------------------------------------------------------- outcome
    def outcome(self) -> dict:
        B = self.B
        land = np.full((B, 2), np.nan)
        ok = self.hit & np.isfinite(self.x_after[:, 0])
        if ok.any():
            land[ok], _ = self.fm.landing(self.x_after[ok])
        dist = np.linalg.norm(land - self.P.target, axis=1)
        dist = np.where(self.hit & np.isfinite(dist), dist, np.inf)
        quality = np.where(np.isfinite(dist), np.exp(-0.5 * (dist / self.tcfg["sigma"]) ** 2), 0.0)
        v_imp = np.array([p.info.get("speed", np.nan) * p.info.get("scale", np.nan) if p is not None else np.nan
                          for p in self.plans])
        return {"hit": self.hit.copy(), "landing": land, "dist": dist, "quality": quality,
                "on_target": dist < self.tcfg["radius"], "n_plans": self.n_plans.copy(), "n_fail": self.n_fail.copy(),
                "plan_wall": self.plan_wall.copy(), "sat_steps": self.sat_steps.copy(), "hit_k": self.hit_k.copy(),
                "hit_off": self.hit_off.copy(), "v_pre": self.v_pre.copy(), "v_post": self.v_post.copy(),
                "bat_speed": v_imp, "speed": self.P.speed.copy(), "ekf_resets": self.ekf.n_reset.copy()}


def run_episode(env: BatEnv, P: Pitches, trigger, frame_cb=None) -> dict:
    """Roll one batch of pitches with a compute trigger (batter.control.triggers)."""
    obs = env.reset(P)
    trigger.reset(env.B)
    done = False
    while not done:
        replan = trigger(obs)
        if frame_cb is not None:
            frame_cb(env)
        obs, done = env.step(replan)
    return env.outcome()
