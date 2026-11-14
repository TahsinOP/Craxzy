"""Clips and renders for the documentation site (docs/public/video, results/targets_render.png).

    env -u PYTHONPATH .venv/bin/python scripts/docs_videos.py

Replays, with the free-space moves simulated, the actions that the planners chose
in the milestone runs (read from results/*.json), so the clips show exactly the
episodes behind the tables.
"""
import json
import shutil
from pathlib import Path

import numpy as np

from clay_sculptor import RESULTS, ROOT, config
from clay_sculptor.envs.targets import DEMOS, make_target
from clay_sculptor.eval import render

OUT = ROOT / "docs" / "public" / "video"
OUT.mkdir(parents=True, exist_ok=True)
cfg = config.load()
env = cfg.env(nsim=4)


def replay(pairs, title):
    """Frames of a sequence of (tool, action); consecutive actions of one tool run without resetting the arm."""
    groups, state, frames = [], env.state0, []
    for tool, a in pairs:
        if groups and groups[-1][0] == tool:
            groups[-1][1].append(a)
        else:
            groups.append((tool, [a]))
    for tool, actions in groups:
        frames += render.record(env, state, tool, actions, title=title)
        state = render.record.last_state
    frames += [frames[-1]] * 20
    return frames, state


def episode_pairs(ep, default_tool):
    tools = ep.get("tools") or [default_tool] * len(ep["actions"])
    return list(zip(tools, ep["actions"]))


if (RESULTS / "m0_bench.mp4").exists():
    shutil.copy(RESULTS / "m0_bench.mp4", OUT / "bench.mp4")

shots = []
m1 = RESULTS / "m1_oracle.json"
if m1.exists():
    r = json.loads(m1.read_text())
    frames = []
    for name, t in r["targets"].items():
        ep = t["episodes"][0]
        f, state = replay(episode_pairs(ep, t["tool"]), f"oracle MPC: {name}")
        frames += f
        tgt = make_target(env, name)
        shots.append(render.label(render.snapshot(env, tgt.state, t["tool"], "close"), f"{name}: target"))
        shots.append(render.label(render.snapshot(env, state, t["tool"], "close"), f"{name}: oracle MPC"))
    render.write_mp4(OUT / "oracle.mp4", frames, fps=30)
    print("wrote oracle.mp4", flush=True)
    render.save_png(RESULTS / "targets_render.png", render.grid(shots, 2))

m3 = RESULTS / "m3_mpc.json"
if m3.exists():
    r = json.loads(m3.read_text())
    frames = []
    for name, res in r["methods"]["structured"].items():
        if name == "overall":
            continue
        f, _ = replay(episode_pairs(res["episodes"][0], DEMOS[name]["tool"]), f"structured model: {name}")
        frames += f
    render.write_mp4(OUT / "structured.mp4", frames, fps=30)
    print("wrote structured.mp4", flush=True)

st = RESULTS / "stretch.json"
if st.exists():
    r = json.loads(st.read_text())
    frames = []
    for name, t in r["targets"].items():
        key = "fallback" if "fallback" in t["planners"] else next(iter(t["planners"]))
        ep = t["planners"][key]["episodes"][0]
        f, _ = replay(episode_pairs(ep, DEMOS[name]["tool"]), f"{'letter T' if name == 't' else 'three tools'}")
        frames += f
    render.write_mp4(OUT / "stretch.mp4", frames, fps=30)
    print("wrote stretch.mp4", flush=True)
