"""Batched catch environment on ``mujoco.batch`` (multi-world CPU; speculative contacts).

One world = Panda + one thrown object.  Per simulator step (8 ms):

  1. sense   : noisy position fix of the object (rate / dropout from the config) until the
               object first touches the robot;
  2. predict : predictor.step + forecast -> catch point, time, velocity, covariance;
  3. decide  : the caller (a trigger, M4) says which worlds replan; the first available
               forecast always triggers a plan;
  4. plan    : IK for the hand pose at the predicted catch point + joint-space QP (planner.py);
  5. track   : discrete computed torque on the plan (arm.py); jaws close shortly before the
               predicted contact or on a contact-sensor event;
  6. step the batch.

The env is a stepper (``reset`` / ``step(replan)``) so the same code serves the fixed-rate
controller of M3 and the triggers and PPO policy of M4.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mujoco import batch as mjbatch

from card_catcher.config import Plane
from card_catcher.control.arm import Arm, hand_rotation, ready_configuration
from card_catcher.control.planner import Plan, TrajectoryQP
from card_catcher.envs.flight import FlightData
from card_catcher.models.scene import TABLE_TOP, build_catch_model
from card_catcher.models.objects import OBJECTS

STAGE_NONE, STAGE_VEL = int(mujoco.mjtStage.mjSTAGE_NONE), int(mujoco.mjtStage.mjSTAGE_VEL)


@dataclass
class CatchParams:
    closing: str = "vertical"     # jaw axis: "vertical" (one finger above, one below) or "horizontal"
    lead_steps: int = 2           # be at the catch pose this many steps before the object
    close_lead: float = 0.05      # close the jaws this long before the predicted contact [s]
    cone_deg: float = 35.0        # the palm faces the incoming object, within this cone of the plane normal
    depth: float = 0.0            # aim the TCP this far beyond the catch point along the approach axis [m]
    min_tau: float = 0.03         # no replanning closer than this to the catch [s]
    hold_time: float = 0.5        # the grasp is judged this long after the last catch time [s]
    grip_dist: float = 0.08       # "in hand": object centre within this distance of the TCP [m]
    w_rot: float = 0.3            # IK weight of orientation against position
    ready_pitch_deg: float = 50.0 # palm elevation in the ready pose (typical arrival direction)
    effector: str = "gripper"     # "gripper" (parallel jaws) or "suction" (stretch: adhesion disc, jaws stay open)
    suction_force: float = 1.0    # adhesion of the suction disc, newtons per contact point


# Per-object defaults: palm elevation of the ready pose = typical arrival direction of that
# object (ball ~55 deg, plate ~65 deg, card ~79 deg above the horizontal; the cone caps it at 65),
# and the predictor used for catching (the ball needs no residual: M1 is already at 3.5 mm).
DEFAULT_PARAMS = {"ball": CatchParams(ready_pitch_deg=50.0), "plate": CatchParams(ready_pitch_deg=60.0),
                  "card": CatchParams(ready_pitch_deg=65.0)}
DEFAULT_PREDICTOR = {"ball": "physics", "plate": "hybrid", "card": "hybrid"}


def clip_to_cone(a: np.ndarray, axis: np.ndarray, max_deg: float) -> np.ndarray:
    """Unit vector closest to ``a`` within ``max_deg`` of ``axis``."""
    a = a / np.linalg.norm(a)
    c = float(np.clip(a @ axis, -1.0, 1.0))
    if np.arccos(c) <= np.deg2rad(max_deg):
        return a
    perp = a - c * axis
    n = np.linalg.norm(perp)
    if n < 1e-9:
        return axis.copy()
    th = np.deg2rad(max_deg)
    return np.cos(th) * axis + np.sin(th) * perp / n


class CatchEnv:
    def __init__(self, obj: str, cfg: dict, nsim: int, predictor, params: CatchParams | None = None,
                 seed: int = 0, nthread: int = 0, sim_override: dict | None = None):
        self.obj, self.cfg, self.B = obj, cfg, nsim
        self.prm = params or CatchParams()
        sim = dict(cfg["sim"], **(sim_override or {}))
        self.model = build_catch_model(obj, sim, effector=self.prm.effector, suction_force=self.prm.suction_force)
        self.h = self.model.opt.timestep
        self.plane = Plane.from_cfg(cfg)
        self.arm = Arm(self.model)
        self.ix = self.arm.ix
        self.qp = TrajectoryQP(self.h, self.arm.q_min, self.arm.q_max)
        self.closing_axis = self.plane.ew if self.prm.closing == "vertical" else self.plane.eu
        # ready pose: TCP at the centre of the catch window, palm facing the direction objects
        # of this kind typically arrive from (``ready_pitch_deg`` above the horizontal)
        th = np.deg2rad(self.prm.ready_pitch_deg)
        a0 = clip_to_cone(np.array([np.cos(th), 0.0, np.sin(th)]), self.plane.n, self.prm.cone_deg)
        self.q_ready = ready_configuration(self.arm, self.plane, self.prm.closing, approach=a0)
        self.predictor = predictor
        self.meas = cfg["measurement"]
        self.rng = np.random.default_rng(seed)
        b = mjbatch.Batch(self.model, nsim=nsim, nthread=nthread, persistent=True)
        self.batch = b
        self.qpos, self.qvel, self.ctrl = b.bind("qpos"), b.bind("qvel"), b.bind("ctrl")
        self.M, self.bias, self.passive = b.bind("M"), b.bind("qfrc_bias"), b.bind("qfrc_passive")
        self.sens, self.site_xpos = b.bind("sensordata"), b.bind("site_xpos")
        s = self.ix.sensors
        self._found = {n: s[n].start for n in ("touch_left", "touch_right", "touch_hand", "touch_floor", "touch_table")}

    # ------------------------------------------------------------------ reset
    def reset(self, D: FlightData, idx: np.ndarray) -> dict:
        """Load throws ``idx`` of dataset D into the B worlds and take the first fix."""
        B, ix = self.B, self.ix
        assert len(idx) == B
        self.truth_t, self.truth_p, self.truth_v, self.truth_k = D.cross_t[idx], D.cross_p[idx], D.cross_v[idx], D.cross_k[idx]
        self.n_steps = int(np.ceil((self.truth_t.max() + self.prm.hold_time) / self.h)) + 1
        self.batch.reset(stage=STAGE_NONE)
        self.qpos[:, ix.arm_qadr] = self.q_ready
        self.qpos[:, ix.finger_qadr] = 0.04
        self.qpos[:, ix.obj_qadr:ix.obj_qadr + 3] = D.p[idx, 0]
        self.qpos[:, ix.obj_qadr + 3:ix.obj_qadr + 7] = D.quat[idx, 0]
        self.qvel[:] = 0.0
        self.qvel[:, ix.obj_dadr:ix.obj_dadr + 3] = D.v[idx, 0]
        self.qvel[:, ix.obj_dadr + 3:ix.obj_dadr + 6] = D.w[idx, 0]
        self.ctrl[:] = 0.0
        self.ctrl[:, ix.grip_act] = 255.0
        self.batch.forward(stage=STAGE_VEL)

        self.k = 0
        self.plans: list[Plan | None] = [None] * B
        self.plan_c = np.zeros((B, 3))
        self.plan_cov = np.zeros((B, 2, 2))
        self.plan_t = np.zeros(B)                 # absolute predicted catch time used by the plan
        self.plan_k = np.zeros(B, int)
        self.n_plans = np.zeros(B, int)
        self.q_goal = np.tile(self.q_ready, (B, 1))
        self.grip_closed = np.zeros(B, bool)
        self.contact_seen = np.zeros(B, bool)
        self.contact_k = np.full(B, -1)
        self.sat_steps = np.zeros(B, int)
        self.late_plans = np.zeros(B, int)
        self.min_dist = np.full(B, np.inf)
        self.tcp_err_at_cross = np.full(B, np.nan)
        self.pred_err_last = np.full(B, np.nan)   # error of the catch point the final plan used
        self.predictor.reset()
        if hasattr(self.predictor, "set_truth"):
            self.predictor.set_truth(slice(None), self.truth_t, self.truth_p, self.truth_v)
        self._sense()
        return self._obs()

    # ------------------------------------------------------------------ sense
    def obj_pos(self) -> np.ndarray:
        return self.qpos[:, self.ix.obj_qadr:self.ix.obj_qadr + 3]

    def tcp_pos(self) -> np.ndarray:
        return self.site_xpos[:, self.ix.tcp_site]

    def _touch(self, name: str) -> np.ndarray:
        return self.sens[:, self._found[name]] > 0.5

    def _sense(self) -> None:
        B = self.B
        touching = self._touch("touch_left") | self._touch("touch_right") | self._touch("touch_hand")
        new = touching & ~self.contact_seen
        self.contact_k[new] = self.k
        self.contact_seen |= touching
        scheduled = (self.k % self.meas["every"] == 0)
        valid = np.full(B, scheduled) & (self.rng.random(B) >= self.meas["dropout"]) & ~self.contact_seen
        if self.k == 0:
            valid[:] = True
        z = self.obj_pos() + self.rng.normal(0.0, self.meas["sigma"], (B, 3))
        self.predictor.step(z, valid)
        self.fc = self.predictor.forecast()
        fc = self.fc
        self.active = fc["found"] & (fc["t"] > self.prm.min_tau) & ~self.contact_seen
        # jaws: close just before the predicted contact, or as soon as the contact sensors fire
        t_now = self.k * self.h
        has_plan = np.array([p is not None for p in self.plans])
        due_fc = fc["found"] & (fc["t"] <= self.prm.close_lead)
        due_plan = has_plan & (self.plan_t - t_now <= self.prm.close_lead)
        self.grip_closed |= self.contact_seen | np.where(fc["found"] & ~self.contact_seen, due_fc, due_plan)
        d = np.linalg.norm(self.obj_pos() - self.tcp_pos(), axis=1)
        self.min_dist = np.minimum(self.min_dist, d)
        at = self.k == self.truth_k + 1
        self.tcp_err_at_cross[at] = np.linalg.norm(self.tcp_pos()[at] - self.truth_p[at], axis=1)

    def _obs(self) -> dict:
        fc = self.fc
        has_plan = np.array([p is not None for p in self.plans])
        drift = np.where(has_plan & fc["found"], np.linalg.norm(fc["p"] - self.plan_c, axis=1), 0.0)
        dt_drift = np.where(has_plan & fc["found"], np.abs(self.k * self.h + fc["t"] - self.plan_t), 0.0)
        tr_now = np.where(fc["found"], np.trace(np.nan_to_num(fc["cov_uw"]), axis1=1, axis2=2), 0.0)
        tr_plan = np.where(has_plan, np.trace(self.plan_cov, axis1=1, axis2=2), 0.0)
        return {"active": self.active.copy(), "has_plan": has_plan, "need_first": self.active & ~has_plan,
                "drift": drift, "time_drift": dt_drift, "tau": np.where(fc["found"], fc["t"], 0.0),
                "tr_now": tr_now, "tr_plan": tr_plan, "since": (self.k - self.plan_k) * self.h, "k": self.k, "h": self.h,
                "speed": np.where(fc["found"], np.linalg.norm(np.nan_to_num(fc["v"]), axis=1), 0.0)}

    # ------------------------------------------------------------------- plan
    def _plan(self, i: int) -> None:
        fc, ix, prm = self.fc, self.ix, self.prm
        a = clip_to_cone(-fc["v"][i], self.plane.n, prm.cone_deg)
        R = hand_rotation(a, self.closing_axis)
        target = fc["p"][i] - prm.depth * a
        qg, _, _ = self.arm.ik(target, R, self.q_goal[i], self.q_ready, w_rot=prm.w_rot)
        N = int(np.floor(fc["t"][i] / self.h)) - prm.lead_steps
        plan = self.qp.solve(self.qpos[i, ix.arm_qadr].copy(), self.qvel[i, ix.arm_dadr].copy(), qg, N, k0=self.k)
        self.plans[i] = plan
        self.q_goal[i] = qg
        self.plan_c[i] = fc["p"][i]
        self.plan_cov[i] = np.nan_to_num(fc["cov_uw"][i])
        self.plan_t[i] = self.k * self.h + fc["t"][i]
        self.plan_k[i] = self.k
        self.n_plans[i] += 1
        # late = some joint is still more than 0.02 rad (about 1 cm at the TCP) from its goal at the interception
        self.late_plans[i] += int(np.abs(plan.q_at_intercept - plan.q_goal).max() > 0.02)
        self.pred_err_last[i] = np.linalg.norm(fc["p"][i] - self.truth_p[i])

    # ------------------------------------------------------------------- step
    def step(self, replan: np.ndarray | None = None) -> tuple[dict, bool]:
        """Replan where asked (and where a first plan is due), apply control, advance one step."""
        B, ix = self.B, self.ix
        has_plan = np.array([p is not None for p in self.plans])
        want = self.active & (~has_plan | (np.zeros(B, bool) if replan is None else np.asarray(replan, bool)))
        for i in np.flatnonzero(want):
            self._plan(i)
        q_ref, v_ref, a_ref = np.tile(self.q_ready, (B, 1)), np.zeros((B, 7)), np.zeros((B, 7))
        for i, p in enumerate(self.plans):
            if p is not None:
                q_ref[i], v_ref[i], a_ref[i] = p.ref(self.k)
        tau, sat = self.arm.torque(self.M, self.bias, self.passive, self.qpos[:, ix.arm_qadr], self.qvel[:, ix.arm_dadr],
                                   q_ref, v_ref, a_ref)
        self.sat_steps += sat.any(axis=1)
        self.ctrl[:, :7] = tau
        self.ctrl[:, ix.grip_act] = np.where(self.grip_closed & (self.prm.effector == "gripper"), 0.0, 255.0)
        self.batch.step(stage=STAGE_VEL)
        self.k += 1
        self._sense()
        return self._obs(), self.k >= self.n_steps

    # ---------------------------------------------------------------- outcome
    @property
    def thickness(self) -> float:
        return 2.0 * float(OBJECTS[self.obj].semi_axes.min())

    def final_penetration(self) -> np.ndarray:
        """Deepest overlap [m] between the object and any other geom, per world, in the current state."""
        m, d = self.model, mujoco.MjData(self.model)
        pen = np.zeros(self.B)
        for i in range(self.B):
            self.batch.get_data(i, d)
            mujoco.mj_forward(m, d)
            for c in d.contact[: d.ncon]:
                if self.ix.obj_body in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                    pen[i] = max(pen[i], -c.dist)
        return pen

    def outcome(self) -> dict:
        """Judged at the end of the episode (>= hold_time after the catch)."""
        ix = self.ix
        left, right, hand = self._touch("touch_left"), self._touch("touch_right"), self._touch("touch_hand")
        d = np.linalg.norm(self.obj_pos() - self.tcp_pos(), axis=1)
        above = self.obj_pos()[:, 2] > TABLE_TOP + 0.05
        near = d < self.prm.grip_dist
        off_ground = ~(self._touch("touch_floor") | self._touch("touch_table")) & above
        pen = self.final_penetration()
        caught = left & right & near & off_ground
        return {
            "caught": caught,                                           # pinched between both fingers
            # ... and the contact is physical: no geometry overlaps the object by more than its own
            # thickness.  Without speculative contacts a thin object can end up lodged inside a finger.
            "clean": caught & (pen <= self.thickness),
            "final_pen": pen,
            "retained": (left | right | hand) & near & off_ground,      # still in the hand, any contact
            "touched": self.contact_seen.copy(),
            "n_plans": self.n_plans.copy(), "late_plans": self.late_plans.copy(), "sat_steps": self.sat_steps.copy(),
            "min_dist": self.min_dist.copy(), "tcp_err_at_cross": self.tcp_err_at_cross.copy(),
            "pred_err_last": self.pred_err_last.copy(), "final_dist": d,
        }


def run_episode(env: CatchEnv, D: FlightData, idx: np.ndarray, trigger, frame_cb=None) -> dict:
    """Roll one batch of throws with a replanning trigger (card_catcher.control.triggers)."""
    obs = env.reset(D, idx)
    trigger.reset(env.B)
    done = False
    while not done:
        replan = trigger(obs)
        if frame_cb is not None:
            frame_cb(env)
        obs, done = env.step(replan)
    return env.outcome()


def run_fixed_rate(env: CatchEnv, D: FlightData, idx: np.ndarray, rate_hz: float = 50.0, frame_cb=None) -> dict:
    """Periodic replanning at ``rate_hz`` (the M3 controller and M4's baseline)."""
    from card_catcher.control.triggers import Periodic

    return run_episode(env, D, idx, Periodic(rate_hz), frame_cb)
