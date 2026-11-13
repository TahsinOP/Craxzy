"""P6 when-to-think batter: model what you can, learn the rest, and know when to think."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"
DATA = ROOT / "data"

# Offscreen rendering through EGL.  MuJoCo reads this when it is first imported, so it is set here,
# before any module of the package imports mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
