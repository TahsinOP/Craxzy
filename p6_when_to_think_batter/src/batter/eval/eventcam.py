"""Stretch: an event-camera emulator and a trigger that fires on events.

An event camera reports, per pixel and asynchronously, when the log-brightness has changed
by more than a contrast threshold C since that pixel's last event.  The emulator renders
the scene offscreen each simulator step, keeps a per-pixel reference log-intensity, and
emits (row, column, polarity) wherever |log I - reference| >= C, resetting the reference
there (the standard frame-based emulation; events are therefore timed to the 10 ms step,
not to microseconds).

The camera looks down on the flight corridor from the side, so the white ball is seen
against the grass; shadows are switched off in its renderer and the image columns that
show the robot are masked.  The ball is about 6 pixels across.

Event trigger.  The leading edge of the moving ball produces ON events.  Their centroid is
a (latency-free) image measurement of the ball.  It is compared with the image position
the cached plan's forecast predicts for that instant; the squared distance, normalised by
the projected forecast covariance plus a pixel noise, is chi-square with 2 dof under the
model.  The trigger fires when its mean over the last few steps exceeds a gate: the
innovation trigger of M3, in pixels, on events instead of on delayed state estimates.
"""
from __future__ import annotations

import os
import subprocess
import sys

import mujoco
import numpy as np

from batter.eval.render import free_camera

_PROBE = """
import mujoco
from OpenGL import GL
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 8, 8)
print("RENDERER=" + GL.glGetString(GL.GL_RENDERER).decode())
"""


def use_software_rendering(max_devices: int = 6) -> int | None:
    """Point MuJoCo's EGL at Mesa's software rasteriser (llvmpipe), one thread per process.

    Call it before the process pool starts and before anything renders.  Every pool worker of the
    event-camera study renders; two dozen contexts on one GPU is more than a desktop driver takes
    (nouveau rejected them: ``kernel rejected pushbuf``).  The EGL device list is probed in
    subprocesses; returns the device index, or None if there is no llvmpipe device.
    (``LIBGL_ALWAYS_SOFTWARE=1`` is not an alternative: with MuJoCo's device-based EGL it segfaults.)
    """
    for i in range(max_devices):
        env = {**os.environ, "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(i)}
        env.pop("PYTHONPATH", None)
        try:
            out = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=60).stdout
        except subprocess.TimeoutExpired:
            continue
        if "RENDERER=" in out and "llvmpipe" in out.split("RENDERER=")[1]:
            os.environ["MUJOCO_GL"] = "egl"
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(i)
            os.environ["LP_NUM_THREADS"] = "1"
            return i
    return None


class EventCamera:
    def __init__(self, model: mujoco.MjModel, width: int = 480, height: int = 240, contrast: float = 0.25,
                 lookat=(3.9, 0.0, 1.0), distance: float = 3.6, azimuth: float = 90.0, elevation: float = -38.0,
                 x_min: float = 1.25):
        self.model, self.W, self.H, self.C = model, width, height, contrast
        self.renderer = mujoco.Renderer(model, height, width)
        self.cam = free_camera(lookat, distance, azimuth, elevation)
        self.data = mujoco.MjData(model)
        self.opt = mujoco.MjvOption()
        # pinhole geometry of the free camera
        az, el = np.deg2rad(azimuth), np.deg2rad(elevation)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        self.pos = np.asarray(lookat, float) - distance * fwd
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        self.R = np.stack([right, up, fwd])                     # world -> camera (x right, y up, z forward)
        self.f = 0.5 * height / np.tan(0.5 * np.deg2rad(model.vis.global_.fovy))
        self.ref: np.ndarray | None = None
        # the robot (x < x_min) is masked out: events are only read where the ball flies alone
        self.col_min = int(np.ceil(self.project(np.array([x_min, 0.0, 1.0]))[0]))
        mujoco.mj_forward(model, self.data)
        self.log_image(self.data)                              # warm-up: the renderer's first frame differs slightly

    def reset(self) -> None:
        self.ref = None

    def log_image(self, data: mujoco.MjData) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.cam, scene_option=self.opt)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        img = self.renderer.render().astype(np.float32)
        grey = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        return np.log(grey + 8.0)

    def events(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rows, columns, polarity) of the events since the previous call."""
        L = self.log_image(data)
        if self.ref is None:
            self.ref = L
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
        d = L - self.ref
        fired = np.abs(d) >= self.C
        fired[:, : self.col_min] = False
        rows, cols = np.nonzero(fired)
        pol = np.sign(d[fired]).astype(int)
        self.ref[fired] = L[fired]
        return rows, cols, pol

    def project(self, p: np.ndarray) -> np.ndarray:
        """World points (..., 3) -> pixel coordinates (..., 2) as (column, row)."""
        c = (np.asarray(p, float) - self.pos) @ self.R.T
        u = self.W / 2.0 + self.f * c[..., 0] / c[..., 2]
        v = self.H / 2.0 - self.f * c[..., 1] / c[..., 2]
        return np.stack([u, v], axis=-1)

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        """d(pixel)/d(world point), (2, 3)."""
        c = self.R @ (np.asarray(p, float) - self.pos)
        Jc = self.f * np.array([[1.0 / c[2], 0.0, -c[0] / c[2] ** 2], [0.0, -1.0 / c[2], c[1] / c[2] ** 2]])
        return Jc @ self.R

    def close(self) -> None:
        self.renderer.close()


class EventRig:
    """Per-world event cameras on a BatEnv, and the event-innovation statistic."""

    def __init__(self, env, window: int = 4, pixel_sigma: float = 0.5, min_events: int = 2, **cam_kw):
        self.env, self.window, self.pixel_sigma, self.min_events = env, window, pixel_sigma, min_events
        self.cams = [EventCamera(env.model, **cam_kw) for _ in range(env.B)]
        self.n_events = 0

    def reset(self) -> None:
        B = self.env.B
        for c in self.cams:
            c.reset()
        self.hist = np.zeros((B, self.window))
        self.count = np.zeros(B, int)
        self.plan_seen = np.zeros(B, int)
        self.centroid = np.full((B, 2), np.nan)

    def update(self) -> None:
        """Render every world, extract events, update the statistic against the cached plans' forecasts."""
        env = self.env
        for i, cam in enumerate(self.cams):
            env.batch.get_data(i, cam.data)
            mujoco.mj_forward(env.model, cam.data)
            rows, cols, pol = cam.events(cam.data)
            self.n_events += len(rows)
            if env.n_plans[i] != self.plan_seen[i]:              # a new plan: start a new window
                self.plan_seen[i] = env.n_plans[i]
                self.hist[i], self.count[i] = 0.0, 0
            on = pol > 0
            self.centroid[i] = np.nan
            if on.sum() < self.min_events or env.plans[i] is None or env.hit[i]:
                continue
            # ON events mark where the ball has arrived: its position now (no latency)
            uv = np.array([cols[on].mean() + 0.5, rows[on].mean() + 0.5])     # pixel centres
            self.centroid[i] = uv
            idx = env.k - env.plan_j0[i]                         # the plan's forecast for this instant
            if not (0 < idx < env.plan_len[i]):
                continue
            p_hat = env.plan_p[i, idx]
            J = cam.jacobian(p_hat)
            S = J @ env.plan_S[i, idx] @ J.T + self.pixel_sigma**2 * np.eye(2)
            nu = uv - cam.project(p_hat)
            self.hist[i] = np.roll(self.hist[i], 1)
            self.hist[i, 0] = float(nu @ np.linalg.solve(S, nu))
            self.count[i] += 1

    def chi2(self) -> np.ndarray:
        n = np.minimum(self.count, self.window)
        return np.where(n > 0, self.hist.sum(axis=1) / np.maximum(n, 1), 0.0)

    def close(self) -> None:
        for c in self.cams:
            c.close()


def run_event_episode(env, rig: EventRig, P, gate: float, min_count: int = 2) -> dict:
    """One batch of pitches with the event trigger."""
    obs = env.reset(P)
    rig.reset()
    done = False
    while not done:
        rig.update()
        fire = obs["active"] & obs["has_plan"] & (rig.count >= min_count) & (rig.chi2() > gate)
        obs, done = env.step(fire)
    return env.outcome()


def _event_work(args):
    from batter.eval.runner import make_env

    cfg, gate, P, seed = args
    global _RIG
    key = (len(P),)
    if _RIG.get("key") != key:
        env = make_env(cfg, len(P), seed)
        _RIG.update(key=key, env=env, rig=EventRig(env))
    env, rig = _RIG["env"], _RIG["rig"]
    env.rng = np.random.default_rng(seed)
    out = run_event_episode(env, rig, P, gate)
    steps = np.empty(len(P), dtype=object)
    for i, st in enumerate(env.plan_steps):
        steps[i] = np.array(st, int)
    out["plan_steps"], out["s_step"], out["s_kind"] = steps, P.s_step.copy(), P.s_kind.copy()
    return out


_RIG: dict = {}


def evaluate_events(cfg: dict, gate: float, P, seed: int = 0, B: int = 4) -> dict:
    from batter.eval.runner import pool

    n = len(P)
    jobs = [(cfg, gate, P.take(np.arange(i, min(i + B, n))), seed * 1000 + i) for i in range(0, n, B)]
    outs = list(pool().map(_event_work, jobs))
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
