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


def write_mp4(path: Path, frames: list[np.ndarray], fps: int = 30) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=None) as w:
        for f in frames:
            w.append_data(f)
    return path


# ---- clay scene helpers ----------------------------------------------------------------
_renderers: dict = {}


def renderer(model, height: int = 360, width: int = 640) -> mujoco.Renderer:
    key = (id(model), height, width)
    if key not in _renderers:
        _renderers[key] = mujoco.Renderer(model, height, width)
    return _renderers[key]


def frame(model, data, camera: str = "close", height: int = 360, width: int = 640) -> np.ndarray:
    r = renderer(model, height, width)
    r.update_scene(data, camera=camera)
    return r.render().copy()


def snapshot(env, state, tool: str, camera: str = "close", height: int = 360, width: int = 480, park: bool = True) -> np.ndarray:
    """Render a state.  With ``park`` the arm is drawn at its home pose, out of the way of the clay."""
    from clay_sculptor.models.scene import HOME_Q

    m, d = env.models[tool], env._data[tool]
    s = env.prepare(state, HOME_Q) if park else state
    env.set_state(d, s, tool)
    mujoco.mj_forward(m, d)
    return frame(m, d, camera, height, width)


def save_png(path: Path, img: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return path


def record(env, state, tool: str, actions, title: str = "", every: int = 6, cameras=("wide", "close"),
           height: int = 360, width: int = 640) -> list[np.ndarray]:
    """Frames of a continuous execution (free-space moves included), two cameras side by side.

    ``every`` simulation steps per frame: 6 steps of 5 ms at 30 fps is real time."""
    frames, count = [], [0]

    def on_step(m, d):
        count[0] += 1
        if count[0] % every == 0:      # positions in ``d`` are one step old after mj_step: fine for video
            img = np.concatenate([frame(m, d, c, height, width) for c in cameras], axis=1)
            frames.append(label(img, title) if title else img)

    final = env.run_continuous(state, tool, actions, on_step=on_step)
    record.last_state = final
    return frames
