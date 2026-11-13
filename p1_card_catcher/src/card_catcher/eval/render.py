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
