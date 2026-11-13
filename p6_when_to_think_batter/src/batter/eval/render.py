"""Offscreen rendering helpers (EGL) and mp4 writing."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def free_camera(lookat, distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def label(img: np.ndarray, text: str, color=(255, 255, 255), size: int = 20, xy=(10, 8)) -> np.ndarray:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    f = _font(size)
    box = dr.textbbox(xy, text, font=f)
    dr.rectangle((box[0] - 5, box[1] - 4, box[2] + 5, box[3] + 4), fill=(0, 0, 0))
    dr.text(xy, text, fill=color, font=f)
    return np.asarray(im)


def grid(frames: list[np.ndarray], ncol: int) -> np.ndarray:
    rows = [np.concatenate(frames[i:i + ncol], axis=1) for i in range(0, len(frames), ncol)]
    return np.concatenate(rows, axis=0)


def side_by_side(panels: list[list[list[np.ndarray]]]) -> list[np.ndarray]:
    """Frames of several recordings next to each other, pitch by pitch.

    ``panels[p][i]`` is the list of frames of pitch i in panel p.  A pitch lasts different numbers
    of steps in different panels (a miss ends sooner than a hit), so within each pitch the shorter
    recordings hold their last frame; every pitch then starts at the same moment in all panels.
    """
    out = []
    for pitch in zip(*panels):
        n = max(len(f) for f in pitch)
        for k in range(n):
            out.append(np.concatenate([f[min(k, len(f) - 1)] for f in pitch], axis=1))
    return out


def write_mp4(path: Path, frames: list[np.ndarray], fps: int = 30) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=None) as w:
        for f in frames:
            w.append_data(f)
    return path


def add_sphere(scene: mujoco.MjvScene, pos, radius: float, rgba) -> None:
    """Append a decorative sphere to a scene (trails, predicted paths, markers)."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0], float), np.asarray(pos, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def add_line(scene: mujoco.MjvScene, a, b, width: float, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    scene.ngeom += 1


class Recorder:
    """Frame callback for ``run_episode``: renders world ``world`` of a BatEnv with the ball's trail,
    the EKF forecast (predicted path), the path the cached plan was built on and the planner calls."""

    def __init__(self, env, camera, width: int = 960, height: int = 540, world: int = 0, every: int = 1,
                 text: str | None = None, show_forecast: bool = True, repeat: int = 1):
        self.env, self.cam, self.world, self.every, self.text = env, camera, world, every, text
        self.show_forecast, self.repeat = show_forecast, repeat
        self.renderer = mujoco.Renderer(env.model, height, width)
        self.data = mujoco.MjData(env.model)
        self.frames: list[np.ndarray] = []
        self.pitches: list[list[np.ndarray]] = []      # the same frames, one list per pitch
        self.trail: list[np.ndarray] = []
        self.calls: list[np.ndarray] = []
        self._n_plans = 0

    def new_pitch(self) -> None:
        self.trail, self.calls, self._n_plans = [], [], 0
        self.pitches.append([])

    def __call__(self, env) -> None:
        i = self.world
        if env.k == 0:
            self.new_pitch()
        ball = env.ball_state()[i, 0:3].copy()
        self.trail.append(ball)
        if env.n_plans[i] > self._n_plans:              # the planner ran since the last frame
            self._n_plans = int(env.n_plans[i])
            self.calls.append(ball)
        if env.k % self.every:
            return
        env.batch.get_data(i, self.data)
        mujoco.mj_forward(env.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam)
        sc = self.renderer.scene
        for p in self.trail[-60:]:
            add_sphere(sc, p, 0.012, (1.0, 1.0, 1.0, 0.55))
        if self.show_forecast and env.ekf.started[i] and not env.hit[i]:
            path = env.traj[env.nd::2, i, 0:3]
            for a, b in zip(path[:-1], path[1:]):
                if a[2] > 0.03 and a[0] > 0.3:
                    add_line(sc, a, b, 0.006, (0.11, 0.69, 0.48, 0.9))           # forecast now: aqua
            n = int(env.plan_len[i])
            if env.plans[i] is not None and n > 2:
                path = env.plan_p[i, :n:2]
                for a, b in zip(path[:-1], path[1:]):
                    if a[2] > 0.03 and a[0] > 0.3:
                        add_line(sc, a, b, 0.004, (0.92, 0.41, 0.20, 0.8))       # what the cached plan assumed: orange
        for p in self.calls:
            add_sphere(sc, p, 0.022, (0.93, 0.63, 0.0, 0.9))                     # planner calls: yellow
        img = self.renderer.render()
        if self.text:
            img = label(img, self.text)
        img = label(img, f"t = {env.k * env.h * 1e3:4.0f} ms   planner calls {int(env.n_plans[i])}", size=16,
                    xy=(10, img.shape[0] - 30))
        self.frames.extend([img] * self.repeat)
        self.pitches[-1].extend([img] * self.repeat)

    def close(self) -> None:
        self.renderer.close()


def make_recorder(renderer: str, env, camera, width: int = 960, height: int = 540, **kw):
    """A frame recorder for ``renderer`` "filament" (MuJoCo 4.0's physically based renderer, the
    default for videos) or "classic"; falls back to classic if no Filament context can be made."""
    if renderer == "filament":
        try:
            from batter.eval.filament import FilamentRecorder

            return FilamentRecorder(env, camera, width, height, **kw)
        except Exception as e:  # noqa: BLE001  (missing Vulkan device or driver: keep the video)
            print(f"  Filament unavailable ({e}); rendering with the classic renderer", flush=True)
    elif renderer != "classic":
        raise ValueError(renderer)
    return Recorder(env, camera, width, height, **kw)
