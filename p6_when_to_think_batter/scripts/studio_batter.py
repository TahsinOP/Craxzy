"""MuJoCo Studio plugins: watch the batter think.

A ``SimPlugin`` runs the batting cage (one world of the batched environment, with the EKF,
the planner and a compute trigger) and a ``ViewerPlugin`` draws what the controller
believes: the ball's trail (white), the EKF forecast of its path (green), the forecast the
cached plan was built on (orange), the planned strike point (red), and a marker on the
ball's path every time the planner ran (yellow).  A panel lets you pick the trigger, its
threshold, surprises and slow motion.

    env -u PYTHONPATH .venv/bin/python scripts/studio_batter.py                 # native window
    env -u PYTHONPATH .venv/bin/python scripts/studio_batter.py --gfx web       # browser, http://localhost:8080
    env -u PYTHONPATH .venv/bin/python -m mujoco.studio.viewer --plugins=scripts/studio_batter.py --hot_reload=plugins

(no ``from __future__ import annotations`` here: Studio finds handlers by their annotation.)
"""
import argparse
import dataclasses
import time

import mujoco
import numpy as np
from mujoco.studio import app as studio_app
from mujoco.studio import launch_passive, messages, plugin, sim_history, viewer_protocol
from mujoco.studio.dear_imgui import dear_imgui as imgui

from batter import config
from batter.envs.cage import BatEnv, EnvParams
from batter.eval.runner import make_trigger
from batter.eval.sweeps import policy_path
from batter.models.pitch import generate

TRIGGERS = ["fixed rate [Hz]", "innovation [chi2 gate]", "variance [cm]", "drift [cm]", "learned [lambda]"]
DEFAULTS = [100.0, 4.5, 3.0, 3.0, 0.02]
SURPRISES = ["none", "spin", "gust", "mixed"]


@dataclasses.dataclass(frozen=True)
class BatterSettings(messages.Event):
    """Viewer -> sim: what the panel is set to."""

    trigger: int
    value: float
    surprise: int
    slow: float
    paused: bool


@dataclasses.dataclass(frozen=True)
class BatterSnapshot(messages.Snapshot):
    """Sim -> viewer: what the controller believes right now."""

    trail: np.ndarray        # (n, 3) where the ball has been
    forecast: np.ndarray     # (n, 3) EKF forecast from now
    assumed: np.ndarray      # (n, 3) the forecast the cached plan was built on
    calls: np.ndarray        # (n, 3) ball positions at which the planner ran
    strike: np.ndarray       # (3,) planned impact point of the ball (nan: no plan)
    stats: dict


class BatterSim(plugin.SimPlugin):
    """Runs the pitch loop and replaces the default stepping."""

    def __init__(self):
        self.cfg = config.load()
        self.env = None
        self.settings = BatterSettings(0, DEFAULTS[0], 0, 4.0, False)
        self.seed = 0
        self.score = {"pitches": 0, "hits": 0, "on_target": 0, "calls": 0}

    def _trigger(self):
        s = self.settings
        if s.trigger == 0:
            return make_trigger(("periodic", 1000.0 if s.value >= 100 else s.value))
        if s.trigger == 1:
            return make_trigger(("innovation", s.value))
        if s.trigger == 2:
            return make_trigger(("variance", s.value / 100.0))
        if s.trigger == 3:
            return make_trigger(("drift", s.value / 100.0))
        name = "surprise" if policy_path("surprise").exists() else "nominal"
        if not policy_path(name).exists():
            return make_trigger(("periodic", 10.0))            # no trained policy yet (run scripts/m3_triggers.py)
        return make_trigger(("learned", (str(policy_path(name)), s.value)), self.seed)

    def _new_pitch(self):
        env = self.env
        P = generate(env.fm, self.cfg, 1, 9000 + self.seed, surprise=SURPRISES[self.settings.surprise])
        self.seed += 1
        self.obs = env.reset(P)
        self.trigger = self._trigger()
        self.trigger.reset(1)
        self.trail, self.calls, self._n = [], [], 0

    @messages.handler
    def on_load_model(self, event: messages.LoadModelEvent) -> bool:
        if event.path:
            return False                                       # a model file was given: not our scene
        self.env = BatEnv(self.cfg, 1, EnvParams(post_steps=80, max_time=1.7), nthread=1)
        self.handle.set_model(self.env.model)
        self._new_pitch()
        return True

    @messages.handler
    def on_settings(self, event: BatterSettings) -> None:
        changed = (event.trigger, event.value, event.surprise) != (self.settings.trigger, self.settings.value, self.settings.surprise)
        self.settings = event
        if changed and self.env is not None:
            self._new_pitch()

    @messages.handler(priority=messages.Priority.USER)
    def on_step(self, event: messages.StepEvent) -> bool:
        env = self.env
        if env is None:
            return False
        t0 = time.perf_counter()
        if not self.settings.paused:
            ball = env.ball_state()[0, 0:3].copy()
            self.trail.append(ball)
            self.obs, done = env.step(self.trigger(self.obs))
            if env.n_plans[0] > self._n:
                self._n = int(env.n_plans[0])
                self.calls.append(ball)
            env.batch.get_data(0, event.data)
            mujoco.mj_forward(event.model, event.data)
            if done:
                out = env.outcome()
                self.score["pitches"] += 1
                self.score["hits"] += int(out["hit"][0])
                self.score["on_target"] += int(out["on_target"][0])
                self.score["calls"] += int(out["n_plans"][0])
                self.last = {"hit": bool(out["hit"][0]), "dist": float(out["dist"][0])}
                self._send()
                time.sleep(0.6)
                self._new_pitch()
                return True
        self._send()
        time.sleep(max(0.0, env.h * self.settings.slow - (time.perf_counter() - t0)))
        return True

    def _send(self):
        env = self.env
        plan = env.plans[0]
        n = int(env.plan_len[0])
        flying = bool(env.ekf.started[0]) and not env.hit[0]
        empty = np.zeros((0, 3))
        stats = {**self.score, "k": int(env.k), "calls_now": int(env.n_plans[0]), "chi2": float(self.obs["chi2"][0]),
                 "tau": float(self.obs["tau"][0]), "sd_now": float(np.sqrt(max(self.obs["tr_now"][0], 0.0))),
                 "wall_ms": float(1e3 * env.plan_wall[0]), "resets": int(env.ekf.n_reset[0]), "last": getattr(self, "last", None)}
        self.handle.send_to_viewer(BatterSnapshot(
            trail=np.array(self.trail[-80:]) if self.trail else empty,
            forecast=env.traj[env.nd::2, 0, 0:3].copy() if flying else empty,
            assumed=env.plan_p[0, :n:2].copy() if (plan is not None and flying) else empty,
            calls=np.array(self.calls) if self.calls else empty,
            strike=np.array(plan.info["p_b"]) if (plan is not None and "p_b" in plan.info) else np.full(3, np.nan),
            stats=stats))


def _sphere(pos, radius, rgba):
    g = mujoco.MjvGeom()
    mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_SPHERE), np.array([radius, 0.0, 0.0]), np.asarray(pos, float),
                        np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
    return g


def _line(a, b, width, rgba):
    g = mujoco.MjvGeom()
    mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), width, np.asarray(a, float), np.asarray(b, float))
    return g


class BatterOverlay(plugin.ViewerPlugin):
    """Draws the plan and the trigger events; a small panel to steer the experiment."""

    def __init__(self):
        self.snap = None
        self.trigger, self.values, self.surprise, self.slow, self.paused = 0, list(DEFAULTS), 0, 4.0, False

    @messages.handler
    def on_snapshot(self, snap: BatterSnapshot) -> None:
        self.snap = snap

    @messages.handler
    def on_update(self, _: messages.UpdateEvent) -> None:
        geoms = self.viewer.extra_geoms
        geoms.clear()
        s = self.snap
        if s is None:
            return
        for p in s.trail:
            geoms.append(_sphere(p, 0.012, (1.0, 1.0, 1.0, 0.5)))
        for path, width, rgba in [(s.forecast, 0.006, (0.11, 0.69, 0.48, 0.9)), (s.assumed, 0.004, (0.92, 0.41, 0.20, 0.8))]:
            for a, b in zip(path[:-1], path[1:]):
                if a[2] > 0.03 and a[0] > 0.3:
                    geoms.append(_line(a, b, width, rgba))
        for p in s.calls:
            geoms.append(_sphere(p, 0.022, (0.93, 0.63, 0.0, 0.9)))
        if np.isfinite(s.strike[0]):
            geoms.append(_sphere(s.strike, 0.03, (0.9, 0.1, 0.1, 0.6)))

    @messages.handler
    def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
        before = (self.trigger, tuple(self.values), self.surprise, self.slow, self.paused)
        if imgui.Begin("When to think")[0]:
            _, self.trigger = imgui.Combo("trigger", self.trigger, TRIGGERS)
            lo, hi = [(1.0, 100.0), (0.5, 30.0), (0.3, 30.0), (0.3, 30.0), (0.002, 0.2)][self.trigger]
            _, self.values[self.trigger] = imgui.SliderFloat("setting", self.values[self.trigger], lo, hi)
            _, self.surprise = imgui.Combo("surprise", self.surprise, SURPRISES)
            _, self.slow = imgui.SliderFloat("slow motion x", self.slow, 1.0, 20.0)
            _, self.paused = imgui.Checkbox("pause", self.paused)
            imgui.Separator()
            s = self.snap
            if s is not None:
                st = s.stats
                imgui.TextUnformatted(f"this pitch: step {st['k']}, planner calls {st['calls_now']} ({st['wall_ms']:.0f} ms)")
                imgui.TextUnformatted(f"time to plate {1e3 * st['tau']:.0f} ms   crossing sd {100 * st['sd_now']:.1f} cm")
                imgui.TextUnformatted(f"innovation chi2 {st['chi2']:.1f}   filter resets {st['resets']}")
                if st["last"] is not None:
                    last = st["last"]
                    imgui.TextUnformatted("last pitch: " + (f"hit, {last['dist']:.2f} m from the target" if last["hit"] else "missed"))
                n = max(st["pitches"], 1)
                imgui.TextUnformatted(f"{st['pitches']} pitches: hit {100 * st['hits'] / n:.0f} %, on target "
                                      f"{100 * st['on_target'] / n:.0f} %, {st['calls'] / n:.1f} calls/pitch")
            imgui.Separator()
            imgui.TextUnformatted("white: ball   green: forecast now   orange: forecast the plan used")
            imgui.TextUnformatted("yellow: planner ran here   red: planned strike point")
        imgui.End()
        after = (self.trigger, tuple(self.values), self.surprise, self.slow, self.paused)
        if after != before:
            self.viewer.send_to_sim(BatterSettings(self.trigger, float(self.values[self.trigger]), self.surprise,
                                                   float(self.slow), bool(self.paused)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfx", default="", choices=viewer_protocol.GFX_MODES)
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()
    cfg = viewer_protocol.ViewerConfig(width=1400, height=900, gfx=args.gfx, http_port=args.port)
    launch_passive.run(cfg, plugins=[studio_app.App(), sim_history.SimHistory(), BatterSim(), BatterOverlay()])


if __name__ == "__main__":
    main()
