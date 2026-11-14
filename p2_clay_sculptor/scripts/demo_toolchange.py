"""Tool changes made visible: the arm fetches each tool the planner chose from a rack.

    env -u PYTHONPATH .venv/bin/python scripts/demo_toolchange.py
    env -u PYTHONPATH .venv/bin/python scripts/demo_toolchange.py --planner oracle --seed 0
    env -u PYTHONPATH .venv/bin/python scripts/demo_toolchange.py --source house --planner oracle

In the planner (``control/mpc.py: run_episode`` with ``target.tool = None``), a tool change
is free and instant: every tool is its own compiled model and the state moves between
them unchanged.  This script replays a planned three-tool episode (``results/stretch.json``,
target ``combo``) with the tool changes executed by the arm, as an automatic tool changer
would: the three tools stand on a rack beside the clay, and before every action whose
tool differs from the one held, the arm returns the held tool to its pedestal and picks
up the next one.  All motions are simulated: Cartesian straight lines for every move
near the rack or the clay, joint-space minimum-jerk moves at a safe height in between,
tracked by the same pid servos as the primitives.  Docking is kinematic (the tool is
attached when the flange is on it), and the rack is visual only, so the clay is exactly
the clay of the planner's own physics.

``--source house`` replays an episode of ``scripts/demo_unseen.py --target house`` (a house on
the 245 mm block) instead.

Output: results/demo_toolchange[_house].mp4 and docs/public/video/toolchange[_house].mp4.
"""
import argparse
import json
import shutil
import warnings

import imageio.v2 as imageio
import matplotlib
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from clay_sculptor import RESULTS, ROOT, config
from clay_sculptor.control.arm import wrap_yaw
from clay_sculptor.control.primitives import minjerk, waypoints
from clay_sculptor.envs.clay_env import StateLayout
from clay_sculptor.envs.shape import map_rmse
from clay_sculptor.envs.targets import make_target
from clay_sculptor.eval import render
from clay_sculptor.models.scene import HOME_Q, RACK_SLOTS, RACK_TOP, RACK_Y, TOOLS, build_spec

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="stretch", choices=["stretch", "house"])
ap.add_argument("--planner", default=None, help="default: fallback (stretch), oracle (house)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--every", type=int, default=6, help="simulation steps per frame (6 = real time at 30 fps)")
args = ap.parse_args()

BIG = ["clay.size=[0.245, 0.245, 0.04]", "clay.cells=[21, 21, 3]", "clay.count=[43, 43, 4]",
       "map.size=56", "map.half=0.110", "prim.xy_max=0.10"]
SRC = dict(stretch=dict(target="combo", world=[], planner="fallback", suffix="", title="Three tools, one shape",
                        cam=([0.50, 0.13, 0.05], 0.78, 155, -50), up_is_plus_x=True),
           house=dict(target="house", world=BIG, planner="oracle", suffix="_house", title="A house from three tools",
                      cam=([0.50, 0.12, 0.04], 0.95, 165, -52), up_is_plus_x=False))[args.source]
args.planner = args.planner or SRC["planner"]
cfg = config.load(overrides=SRC["world"])
env = cfg.env(nsim=1)
tgt = make_target(env, SRC["target"])
if args.source == "stretch":
    st = json.loads((RESULTS / "stretch.json").read_text())["targets"]["combo"]
else:
    st = json.loads((RESULTS / "demo_house.json").read_text())
ep = next(e for e in st["planners"][args.planner]["episodes"] if e["seed"] == args.seed)
plan = [(t, np.asarray(a)) for t, a in zip(ep["tools"], ep["actions"])]
print(f"{args.planner} seed {args.seed}: tools {ep['tools']}, planned result {ep['map_rmse_mm']:.2f} mm", flush=True)

# ---- models with the rack, and state conversion -----------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    M = {t: build_spec(t, env.clay, env.sim, rack=True).compile() for t in TOOLS}
D = {t: mujoco.MjData(M[t]) for t in TOOLS}
L_rack, L_env = StateLayout(M["press"]), env.layout
base = env.models["press"]
for i in range(base.nbody):        # the rack's bodies come after every existing body
    assert mujoco.mj_id2name(M["press"], mujoco.mjtObj.mjOBJ_BODY, i) == mujoco.mj_id2name(base, mujoco.mjtObj.mjOBJ_BODY, i)


def to_rack(s: np.ndarray) -> np.ndarray:
    """A planner state -> the same state in the rack models (extra bodies: zero applied force)."""
    out = np.zeros(L_rack.size)
    for k, sl in L_env.slices.items():
        r = L_rack.slices[k]
        out[r.start:r.start + (sl.stop - sl.start)] = s[sl]
    return out


def from_rack(s: np.ndarray) -> np.ndarray:
    out = np.zeros(L_env.size)
    for k, sl in L_env.slices.items():
        r = L_rack.slices[k]
        out[sl] = s[r.start:r.start + (sl.stop - sl.start)]
    return out


def get(t):
    s = np.empty(L_rack.size)
    mujoco.mj_getState(M[t], D[t], s, mujoco.mjtState.mjSTATE_INTEGRATION)
    return s


def put(t, s):
    mujoco.mj_setState(M[t], D[t], np.ascontiguousarray(s), mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(M[t], D[t])


def body_geoms(m, body):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    return np.where(m.geom_bodyid == b)[0]


RGBA = {t: M[t].geom_rgba.copy() for t in TOOLS}


def set_visibility(t: str, held: str | None) -> None:
    """In model ``t``: the flange tool shown only if a tool is held; each rack copy hidden
    while its tool is on the flange."""
    m = M[t]
    m.geom_rgba[:] = RGBA[t]
    if held is None:
        m.geom_rgba[body_geoms(m, "tool_tool"), 3] = 0.0
    for r in TOOLS:
        if r == held:
            m.geom_rgba[body_geoms(m, f"rack_{r}_tool"), 3] = 0.0


# ---- rendering --------------------------------------------------------------------------
W_MAIN, H = 1280, 720
W_SIDE = 640
HM = 290
cam_main = render.free_camera(*SRC["cam"])
cmap = matplotlib.colormaps["viridis"]
h0 = env.geom.heightmap(env.nodes(env.state0))
top0 = float(h0.max())
LO, HI = -12.0, 1.0
font = render._font
OUT = RESULTS / f"demo_toolchange{SRC['suffix']}.mp4"
writer = imageio.get_writer(OUT, fps=30, codec="libx264", quality=8,
                            macro_block_size=None, ffmpeg_params=["-movflags", "+faststart"])
LABEL = dict(roller="roller", press="flat press", stamp="cross stamp")


def hmap_img(h):
    z = np.clip((1e3 * (h - top0) - LO) / (HI - LO), 0, 1)
    rgb = (255 * cmap(z)[..., :3]).astype(np.uint8)
    rgb = rgb[::-1] if SRC["up_is_plus_x"] else rgb               # rows x, columns y; match the camera's "up"
    return Image.fromarray(np.ascontiguousarray(rgb)).resize((HM, HM), Image.NEAREST)


state_txt = dict(phase="", step=-1)


def frame(t, nframes=1):
    m, d = M[t], D[t]
    r = render.renderer(m, H, W_MAIN)
    r.update_scene(d, camera=cam_main)
    main = r.render().copy()
    s = from_rack(get(t))
    h = env.geom.heightmap(env.nodes(s))
    err = 1e3 * float(map_rmse(h, tgt.hmap))
    side = Image.new("RGB", (W_SIDE, H), (18, 18, 18))
    dr = ImageDraw.Draw(side)
    dr.text((20, 16), SRC["title"], fill=(255, 255, 255), font=font(24))
    dr.text((20, 50), f"planner: {args.planner} (seed {args.seed}), tools chosen at every step", fill=(190, 190, 190), font=font(14))
    dr.text((20, 92), "target", fill=(230, 230, 230), font=font(16))
    dr.text((20 + HM + 20, 92), "clay now", fill=(230, 230, 230), font=font(16))
    side.paste(hmap_img(tgt.hmap), (20, 116))
    side.paste(hmap_img(h), (20 + HM + 20, 116))
    dr.text((20 + HM + 20, 116 + HM + 8), f"error to target: {err:.2f} mm", fill=(255, 220, 120), font=font(16))
    dr.text((20, 116 + HM + 8), f"untouched: {1e3 * tgt.initial_cost:.2f} mm", fill=(170, 170, 170), font=font(14))
    y = 116 + HM + 48
    dr.text((20, y), "plan", fill=(230, 230, 230), font=font(16))
    y += 26
    for k, (tool, a) in enumerate(plan):
        col = (140, 230, 140) if k < state_txt["step"] else (255, 220, 120) if k == state_txt["step"] else (150, 150, 150)
        what = f"{k + 1}. {LABEL[tool]:12s} depth {1e3 * a[3]:4.1f} mm" + (f", {1e3 * a[4]:.0f} mm roll" if tool == "roller" else "")
        dr.text((30, y), what, fill=col, font=font(15))
        y += 22
    dr.text((20, H - 40), state_txt["phase"], fill=(120, 200, 255), font=font(18))
    img = np.concatenate([main, np.asarray(side)], axis=1)
    for _ in range(nframes):
        writer.append_data(img)


# ---- motion -----------------------------------------------------------------------------
kin, pl, dt = env.kin, env.planner, env.sim.timestep
arm_q = env.ix.arm_qpos
cur = dict(tool="press", held=None, count=0, psi=wrap_yaw(0.0))
Z_SAFE = 0.20                                      # tip height for moves between the rack and the clay [m]
Z_MID = env.clay.top + 0.05                        # tip height for moves between two strokes


def run(q_path: np.ndarray, q0: np.ndarray) -> None:
    t = cur["tool"]
    m, d = M[t], D[t]
    for u in pl._to_ctrl(q0, q_path):
        d.ctrl[:] = u
        mujoco.mj_step(m, d)
        cur["count"] += 1
        if cur["count"] % args.every == 0:
            frame(t)


def q_now() -> np.ndarray:
    return D[cur["tool"]].qpos[arm_q].copy()


def line(p_to, psi=None, speed=0.12, t_min=0.4):
    """Straight Cartesian move of the tip to ``p_to``, tool down, yaw kept (or turned to ``psi``)."""
    psi0 = cur["psi"]
    psi1 = psi0 if psi is None else psi
    q0 = q_now()
    p0, _ = kin.fk(q0)
    n = int(round(max(t_min, np.linalg.norm(p_to - p0) / speed) / dt))
    run(pl._segment(q0, p0, np.asarray(p_to, float), psi0, psi1, n), q0)
    cur["psi"] = psi1


def joint_move(q1, psi, t_min=0.8, speed=1.2):
    """Minimum-jerk joint-space move; only used well above the clay and the rack."""
    q0 = q_now()
    n = int(round(max(t_min, np.abs(q1 - q0).max() / speed) / dt))
    s = minjerk(n)[:, None]
    run(q0 + s * (q1 - q0), q0)
    cur["psi"] = psi


def lift(z):
    p, _ = kin.fk(q_now())
    if p[2] < z - 1e-3:
        line(np.array([p[0], p[1], z]))


def hold(t_s):
    q0 = q_now()
    run(np.repeat(q0[None], int(round(t_s / dt)), axis=0), q0)


def switch_model(t_new):
    s = get(cur["tool"])
    cur["tool"] = t_new
    put(t_new, s)


def slot(t, z=RACK_TOP):
    return np.array([RACK_SLOTS[t], RACK_Y, z])


PSI_RACK = wrap_yaw(0.0)


def to_rack_slot(t):
    lift(Z_SAFE)
    joint_move(kin.ik(slot(t, Z_SAFE), PSI_RACK, q0=q_now()), PSI_RACK)
    line(slot(t), PSI_RACK)
    hold(0.25)


def drop(t):
    state_txt["phase"] = f"tool change: returning the {LABEL[t]} to the rack"
    to_rack_slot(t)
    cur["held"] = None
    for k in TOOLS:
        set_visibility(k, None)
    line(slot(t, Z_SAFE))


def pick(t):
    state_txt["phase"] = f"tool change: picking up the {LABEL[t]}"
    to_rack_slot(t)
    switch_model(t)
    cur["held"] = t
    for k in TOOLS:
        set_visibility(k, t)
    line(slot(t, Z_SAFE))


def do_action(t, a, k):
    state_txt["phase"], state_txt["step"] = f"step {k + 1} of {len(plan)}: {LABEL[t]}", k
    w = waypoints(a, env.clay, env.prim)
    psi = wrap_yaw(w["psi"])
    q_hover, ctrl = pl.reference(t, a)
    p, _ = kin.fk(q_now())
    z_up = Z_SAFE if p[2] > Z_SAFE - 1e-3 else Z_MID
    lift(z_up)
    joint_move(kin.ik(np.array([w["start_hi"][0], w["start_hi"][1], z_up]), psi, q0=q_hover), psi)
    line(w["start_hi"])
    joint_move(q_hover, psi, t_min=0.3)            # same tip pose; aligns the redundant joint with the primitive
    m, d = M[t], D[t]
    for u in ctrl:
        d.ctrl[:] = u
        mujoco.mj_step(m, d)
        cur["count"] += 1
        if cur["count"] % args.every == 0:
            frame(t)
    cur["psi"] = psi


# initial: the planner's start state, arm at home, nothing on the flange
for k in TOOLS:
    put(k, to_rack(env.state0))
    set_visibility(k, None)
state_txt["phase"] = "start: all three tools on the rack"
frame("press", 45)
changes = 0
for k, (t, a) in enumerate(plan):
    if cur["held"] != t:
        if cur["held"] is not None:
            drop(cur["held"])
            changes += 1
        pick(t)
    do_action(t, a, k)
    print(f"step {k + 1} {t}: done", flush=True)
state_txt["step"] = len(plan)
drop(cur["held"])
state_txt["phase"] = "done"
joint_move(np.asarray(HOME_Q, float), PSI_RACK, t_min=1.5)
final = from_rack(get(cur["tool"]))
err = 1e3 * float(map_rmse(env.geom.heightmap(env.nodes(final)), tgt.hmap))
state_txt["phase"] = f"done: {err:.2f} mm off ({100 * err / (1e3 * tgt.initial_cost):.0f} % of the start), {changes} tool changes"
frame(cur["tool"], 120)
writer.close()
dst = ROOT / "docs" / "public" / "video" / f"toolchange{SRC['suffix']}.mp4"
shutil.copy(OUT, dst)
print(f"final {err:.2f} mm (planned {ep['map_rmse_mm']:.2f}), {changes} tool changes, {cur['count'] * dt:.1f} s simulated; wrote {dst}", flush=True)
