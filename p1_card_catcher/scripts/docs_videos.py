"""Render the MuJoCo clips embedded in the documentation site (docs/public/video).

    env -u PYTHONPATH .venv/bin/python scripts/docs_videos.py
"""
from __future__ import annotations

import dataclasses
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import m3_catch  # noqa: E402

from card_catcher import config  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "public" / "video"


def main():
    cfg = config.load()
    OUT.mkdir(parents=True, exist_ok=True)
    for obj in ("plate", "card"):
        prm = dataclasses.replace(m3_catch.PARAMS[obj], effector="suction", suction_force=m3_catch.SUCTION_FORCE[obj])
        m3_catch.video(obj, m3_catch.PREDICTOR[obj], cfg, 0, OUT / f"suction_{obj}.mp4", params=prm, key="retained", tag=" + suction disc")
        print("video:", OUT / f"suction_{obj}.mp4", flush=True)
    for obj in ("ball", "card"):
        m3_catch.video(obj, m3_catch.PREDICTOR[obj], cfg, 0, OUT / f"catch_{obj}.mp4")
        print("video:", OUT / f"catch_{obj}.mp4", flush=True)
    shutil.copy(m3_catch.RESULTS / "m0_tunnel.mp4", OUT / "tunnel.mp4")


if __name__ == "__main__":
    main()
