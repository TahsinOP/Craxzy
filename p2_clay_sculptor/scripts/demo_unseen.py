"""One demo video: every planner on a target that no model or setting was tuned on.

    env -u PYTHONPATH .venv/bin/python scripts/demo_unseen.py                  # staircase, 3 seeds
    env -u PYTHONPATH .venv/bin/python scripts/demo_unseen.py --target smiley  # smiley on a bigger block, 1 seed
    env -u PYTHONPATH .venv/bin/python scripts/demo_unseen.py --video-only     # render from the json

``--target smiley`` runs on a 245 mm block (1.75 times the side of the training block)
with a 21 x 21 x 3 cage, so the cells keep the size the models were trained on.  The
trained networks are reused unchanged: their weights do not depend on the grid size, only
their reference buffers (node positions, edges, pixel coordinates), which are rebuilt
for the bigger block.  The U-Net takes absolute pixel coordinates as an input, which lie
outside its training range there.  Results: results/demo_smiley.{json,mp4}, docs/public/video/smiley.mp4.

The target ``steps`` (``envs/targets.py``) is a staircase of three square pockets along
the diagonal, pressed 6, 9 and 12 mm deep and turned 45 degrees.  It was added after the
models were trained and every planner setting was fixed; nothing below is tuned to it.
The models were trained on random single primitives and have never seen any target.

Planners, all with the CEM search of M1/M3 and the flat press:
  oracle      MuJoCo for every candidate (M1)
  hybrid      ensemble of five structured models, best eight re-ranked with MuJoCo at
              every decision (M4 "always", the fallback configuration that paid)
  structured  prior + GNN residual
  unet        height-map U-Net (pure learning)
  gnn         graph network (pure learning)
  prior       swept-volume prior alone (no learning)

Each planner runs ``--seeds`` (planner seed i uses model seed i mod 3, as in M3) and the
results go to results/demo_unseen.json.  The video replays the first seed of each with
the free-space moves simulated, all six side by side, with the live height map and its
error to the target: results/demo_unseen.mp4 and docs/public/video/unseen.mp4.
"""
import argparse
import json
import shutil
import time

import matplotlib
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

from clay_sculptor import RESULTS, ROOT, config
from clay_sculptor.control.mpc import CEMCfg, HybridEval, ModelEval, OracleEval
from clay_sculptor.envs.shape import map_rmse
from clay_sculptor.envs.targets import make_target
from clay_sculptor.eval import render
from clay_sculptor.eval.episodes import episode, mean_std
from clay_sculptor.learn import train as T
from clay_sculptor.models.scene import HOME_Q

PLANNERS = ["oracle", "hybrid", "structured", "unet", "gnn", "prior"]
LABEL = dict(oracle="oracle: MuJoCo for every candidate", hybrid="hybrid: ensemble + MuJoCo re-ranks best 8",
             structured="structured: prior + GNN residual", unet="U-Net on the height map (pure learning)",
             gnn="GNN on the cage (pure learning)", prior="prior only (no learning)")

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="steps", choices=["steps", "smiley", "house"])
ap.add_argument("--seeds", nargs="*", type=int, default=None, help="default: 0 1 2 for steps, 0 for smiley")
ap.add_argument("--planners", nargs="*", default=PLANNERS, choices=PLANNERS)
ap.add_argument("--video-only", action="store_true")
ap.add_argument("--rerun", action="store_true", help="plan again for planners already in the json")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--every", type=int, default=3, help="simulation steps per frame (6 = real time at 30 fps)")
args = ap.parse_args()
torch.set_num_threads(args.threads)

BIG = ["clay.size=[0.245, 0.245, 0.04]", "clay.cells=[21, 21, 3]", "clay.count=[43, 43, 4]",
       "map.size=56", "map.half=0.110", "prim.xy_max=0.10"]
PARK = np.array(HOME_Q, dtype=float)
PARK[0] += 1.3                                  # arm swung aside, clear of the bigger block
SCENES = dict(
    steps=dict(world=[], seeds=[0, 1, 2], cam=(0.36, -45, -45), up_is_plus_x=True, park=np.array(HOME_Q, dtype=float),
               name="demo_unseen", video="unseen.mp4", title="Unseen target: a staircase",
               desc="3 presses: 6, 9, 12 mm deep, turned 45°", block="140 mm block, as in training"),
    smiley=dict(world=BIG, seeds=[0], cam=(0.52, 180, -60), up_is_plus_x=False, park=PARK,
                name="demo_smiley", video="smiley.mp4", title="A smiley on a bigger block",
                desc="eyes 12 mm, smile of 5 presses 10 mm deep", block="245 mm block: 1.75x the training block's side"),
    house=dict(world=BIG, seeds=[0], cam=(0.52, 180, -60), up_is_plus_x=False, park=PARK,
               name="demo_house", video="house.mp4", title="A house, three tools",
               desc="roof: 2 rolls, window: cross stamp, door: press", block="245 mm block; the planner picks the tool each step"),
)
SC = SCENES[args.target]
args.seeds = args.seeds if args.seeds is not None else SC["seeds"]
cfg = config.load(overrides=SC["world"])
env = cfg.env(nsim=1 if args.video_only else 32)      # 32 parallel worlds only for planning
tgt = make_target(env, args.target)


def load_model(kind: str, seed: int):
    """A trained model on this block.  On the training block: ``train.load``.  Otherwise the
    weights are loaded into a model built for this block's geometry (buffers rebuilt)."""
    if not SC["world"]:
        return T.load(kind, seed, env.geom)
    from clay_sculptor.learn.models import DynamicsModel
    if kind == "prior":
        return DynamicsModel("prior", env.geom).eval()
    ck = torch.load(T.MODEL_DIR / f"{kind}_seed{seed}.pt", weights_only=False)
    model = DynamicsModel(kind, env.geom, **ck["info"]["size"])
    params = dict(model.named_parameters())
    missing = set(params) - set(ck["state"])
    assert not missing, missing
    model.load_state_dict({k: v for k, v in ck["state"].items() if k in params}, strict=False)
    return model.eval()


out_json = RESULTS / f"{SC['name']}.json"
out = json.loads(out_json.read_text()) if out_json.exists() else {}
if out.get("target") != args.target:
    out = dict(target=args.target, planners={})
out.update(seeds=args.seeds, initial_mm=1e3 * tgt.initial_cost, demo=[[t, list(map(float, a))] for t, a in tgt.demo])

# ---- plan -----------------------------------------------------------------------------
if not args.video_only:
    cem = CEMCfg()
    ens = None
    for p in args.planners:
        if p in out["planners"] and not args.rerun:
            continue
        eps = []
        for seed in args.seeds:
            if p == "oracle":
                ev = OracleEval(env)
            elif p == "hybrid":
                from clay_sculptor.learn.models import Ensemble
                ens = ens or Ensemble([load_model("structured", s) for s in range(5)])
                ev = HybridEval(ens, env, trigger="always")
            else:
                ev = ModelEval(load_model(p, seed % 3), env)
            t0 = time.time()
            summ, _ = episode(env, tgt, ev, seed, cem)
            summ.pop("info", None)
            summ["seed"], summ["wall_s"] = seed, time.time() - t0
            eps.append(summ)
            print(f"{p:10s} seed {seed}: {summ['initial_mm']:.2f} -> {summ['map_rmse_mm']:.2f} mm ({100 * summ['remaining']:.1f} % left), "
                  f"{summ['n_actions']} actions, {summ['plan_time_per_decision_s']:.2f} s/decision, {summ['plan_calls']} simulations", flush=True)
        agg = {k: mean_std([e[k] for e in eps]) for k in ("map_rmse_mm", "remaining", "chamfer_mm", "emd_mm", "n_actions",
                                                           "plan_calls", "plan_time_per_decision_s")}
        out["planners"][p] = dict(episodes=eps, summary=agg)
        out_json.write_text(json.dumps(out, indent=1))
    print("wrote", out_json, flush=True)

# ---- video ----------------------------------------------------------------------------
CW, CH3, HM = 360, 270, 270                     # cell: 3D view (CW x CH3) beside a height map (HM x HM)
CELL_W, CELL_H, BAR = CW + HM, CH3 + 34, 34
CUR = dict(tool=tgt.tool or "press")            # the tool model being rendered (changes within an episode
                                                 # when the target leaves the tool to the planner)
cam = render.free_camera([0.5, 0.0, 0.03], *SC["cam"])
h0 = env.geom.heightmap(env.nodes(env.state0))
top0 = float(h0.max())
cmap = matplotlib.colormaps["viridis"]
LO, HI = -12.0, 1.0                              # height-map colour range, mm relative to the untouched top


def hmap_img(h: np.ndarray) -> np.ndarray:
    z = np.clip((1e3 * (h - top0) - LO) / (HI - LO), 0, 1)
    rgb = (255 * cmap(z)[..., :3]).astype(np.uint8)
    rgb = rgb[::-1] if SC["up_is_plus_x"] else rgb       # rows are x, columns y; match the camera's "up"
    return np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize((HM, HM), Image.NEAREST))


def view(state=None) -> np.ndarray:
    m, d = env.models[CUR["tool"]], env._data[CUR["tool"]]
    if state is not None:
        env.set_state(d, env.prepare(state, SC["park"]), CUR["tool"])
        mujoco.mj_forward(m, d)
    r = render.renderer(m, CH3, CW)
    r.update_scene(d, camera=cam)
    return r.render().copy()


def cell(img3d, h, title, line, color=(255, 255, 255)) -> np.ndarray:
    body = np.concatenate([img3d, hmap_img(h)], axis=1)
    bar = Image.new("RGB", (CELL_W, BAR), (18, 18, 18))
    dr = ImageDraw.Draw(bar)
    dr.text((8, 3), title, fill=(255, 255, 255), font=render._font(14))
    dr.text((8, 19), line, fill=color, font=render._font(12))
    return np.concatenate([np.asarray(bar), body], axis=0)


def text_cell(lines, sizes=None, colors=None) -> np.ndarray:
    im = Image.new("RGB", (CELL_W, CELL_H), (18, 18, 18))
    dr = ImageDraw.Draw(im)
    y = 12
    for i, s in enumerate(lines):
        sz = (sizes or {}).get(i, 14)
        dr.text((14, y), s, fill=(colors or {}).get(i, (230, 230, 230)), font=render._font(sz))
        y += sz + 8
    return np.asarray(im)


def jpeg(img: np.ndarray) -> bytes:
    """Frames are kept JPEG-compressed in memory (about 30 kB instead of 290 kB each)."""
    import io
    b = io.BytesIO()
    Image.fromarray(img).save(b, format="JPEG", quality=92)
    return b.getvalue()


def unjpeg(b: bytes) -> np.ndarray:
    import io
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def groups(ep):
    """Consecutive actions of one tool: each group is executed without resetting the arm;
    a tool change between groups is instant here (scripts/demo_toolchange.py shows it done by the arm)."""
    tools = ep.get("tools") or [tgt.tool] * len(ep["actions"])
    out_ = []
    for t, a in zip(tools, ep["actions"]):
        if out_ and out_[-1][0] == t:
            out_[-1][1].append(np.asarray(a))
        else:
            out_.append((t, [np.asarray(a)]))
    return out_


shown = [p for p in PLANNERS if p in out["planners"]]
seed0 = out["seeds"][0]
tracks = {}
for p in shown:
    ep = next(e for e in out["planners"][p]["episodes"] if e["seed"] == seed0)
    frames, count = [], [0]
    n_act = len(ep["actions"])

    def on_step(mm, dd, frames=frames):
        count[0] += 1
        if count[0] % args.every == 0:
            h = env.geom.heightmap(env.nodes(env.get_state(dd, CUR["tool"])))
            frames.append((jpeg(view()), h, 1e3 * float(map_rmse(h, tgt.hmap)), CUR["tool"]))

    final = env.state0
    for tool, acts in groups(ep):
        CUR["tool"] = tool
        final = env.run_continuous(final, tool, acts, on_step=on_step)
    hf = env.geom.heightmap(env.nodes(final))
    tracks[p] = dict(frames=frames, final=(view(final), hf, 1e3 * float(map_rmse(hf, tgt.hmap))), n=n_act, ep=ep)
    print(f"replayed {p}: {len(frames)} frames, final {tracks[p]['final'][2]:.2f} mm (episode: {ep['map_rmse_mm']:.2f} mm)", flush=True)

untouched = (view(env.state0), h0, 1e3 * tgt.initial_cost)
target_cell = cell(view(tgt.state), tgt.hmap, "TARGET  (never seen by any model)",
                   f"{SC['desc']}   untouched block: {1e3 * tgt.initial_cost:.2f} mm off")


def score_cell(done: set) -> np.ndarray:
    lines = [f"Final height-map error, mean of {len(out['seeds'])} seed(s)", "planner        error    time/decision   simulations"]
    cols = {0: (255, 255, 255), 1: (160, 160, 160)}
    order = sorted(shown, key=lambda p: out["planners"][p]["summary"]["map_rmse_mm"][0])
    for p in order:
        s = out["planners"][p]["summary"]
        if p in done:
            lines.append(f"{p:11s} {s['map_rmse_mm'][0]:5.2f} mm   {s['plan_time_per_decision_s'][0]:7.2f} s   {s['plan_calls'][0]:8.0f}")
        else:
            lines.append(f"{p:11s}   ...")
    lines.append("")
    lines.append(f"untouched block: {1e3 * tgt.initial_cost:.2f} mm")
    im = Image.new("RGB", (CELL_W, CELL_H), (18, 18, 18))
    dr = ImageDraw.Draw(im)
    y = 12
    for i, s in enumerate(lines):
        f = render._font(15 if i == 0 else 13)
        if i >= 2:
            f = _mono
        dr.text((14, y), s, fill=cols.get(i, (230, 230, 230)), font=f)
        y += 24 if i == 0 else 21
    return np.asarray(im)


try:
    from PIL import ImageFont
    _mono = ImageFont.truetype("DejaVuSansMono.ttf", 14)
except OSError:
    _mono = render._font(13)

title_cell = text_cell([SC["title"], SC["block"],
                        *(["Every planner searches the same way (cross-entropy,",
                           "96 candidates, one press per decision); only the",
                           "model that predicts a press differs."] if tgt.tool else
                          ["Every planner searches the same way (cross-entropy,",
                           "96 candidates per tool, best tool kept, one action",
                           "per decision); only the predicting model differs."]),
                        "",
                        "Left: the arm.  Right: the height map, from above",
                        f"(yellow = untouched top, blue = {-LO:.0f} mm down).",
                        f"Replay of seed {seed0} at {args.every / 6:.1f}x speed; planning pauses cut."],
                       sizes={0: 20}, colors={0: (255, 255, 255)})
n_frames = max(len(t["frames"]) for t in tracks.values())
INTRO, HOLD = 45, 120
import imageio.v2 as imageio  # noqa: E402

path = RESULTS / f"{SC['name']}.mp4"
writer = imageio.get_writer(path, fps=30, codec="libx264", quality=8, macro_block_size=None,
                            ffmpeg_params=["-movflags", "+faststart"])
blank = np.full((CELL_H, CELL_W, 3), 18, np.uint8)
for i in range(-INTRO, n_frames + HOLD):
    cells, done = [], set()
    for p in shown:
        t = tracks[p]
        if i < 0:
            v, h, e = untouched
            line, col = f"untouched: {e:.2f} mm off", (200, 200, 200)
        elif i < len(t["frames"]):
            v, h, e, tl = t["frames"][i]
            v = unjpeg(v)
            line, col = f"error to target now: {e:.2f} mm" + (f"   tool: {tl}" if tgt.tool is None else ""), (255, 220, 120)
        else:
            v, h, e = t["final"]
            done.add(p)
            line, col = f"done, {t['n']} {'actions' if tgt.tool is None else 'presses'}: {e:.2f} mm off ({100 * e / (1e3 * tgt.initial_cost):.0f} % of the start)", (140, 230, 140)
        cells.append(cell(v, h, LABEL[p], line, col))
    cells += [blank] * (6 - len(cells))
    top = [title_cell, target_cell, score_cell(done)]
    grid = np.concatenate([np.concatenate(top, 1), np.concatenate(cells[:3], 1), np.concatenate(cells[3:6], 1)], 0)
    ph, pw = (-grid.shape[0]) % 16, (-grid.shape[1]) % 16      # browsers' hardware decoders show green frames
    grid = np.pad(grid, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)), constant_values=18)  # unless 16 | size
    writer.append_data(grid)
writer.close()
dst = ROOT / "docs" / "public" / "video" / SC["video"]
shutil.copy(path, dst)
render.save_png(RESULTS / f"{SC['name']}.png", grid)
print(f"wrote {path} ({INTRO + n_frames + HOLD} frames, {grid.shape[1]}x{grid.shape[0]}) and {dst}", flush=True)
