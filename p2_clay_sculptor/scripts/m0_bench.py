"""M0: the clay bench.

    env -u PYTHONPATH .venv/bin/python scripts/m0_bench.py

Checks and measurements (results/m0_bench.json, figures, video):

* scripted press, roll and stamp sequences: shapes, repeatability, sensitivity;
* dents are permanent: plastic against elastic clay under the same press;
* state save/restore: bit-for-bit replay with the full state, and what each
  omitted state component costs;
* leaving the free-space moves out of rollouts: difference to continuous execution;
* tool tracking at the bottom of the stroke; simulation speed.
"""
import argparse
import dataclasses
import json
import time

import mujoco
import numpy as np

from clay_sculptor import RESULTS, config
from clay_sculptor.envs.clay_env import ClayEnv
from clay_sculptor.eval import plots, render
from clay_sculptor.models.scene import TOOLS

ap = argparse.ArgumentParser()
ap.add_argument("--no-video", action="store_true")
ap.add_argument("--set", nargs="*", default=[])
args = ap.parse_args()
cfg = config.load(overrides=args.set)
env = cfg.env(nsim=16)
geom = env.geom
out = {"config": dict(clay=dataclasses.asdict(cfg.clay), sim=dataclasses.asdict(cfg.sim), prim=dataclasses.asdict(cfg.prim))}
mm = lambda x: float(1e3 * x)

SEQS = {
    "press": [("press", [-0.030, -0.030, 0.0, 0.012, 0.0]), ("press", [0.0, 0.0, 0.4, 0.012, 0.0]), ("press", [0.030, 0.030, 0.0, 0.012, 0.0])],
    "roll": [("roller", [0.0, y, np.pi / 2 - 1e-6, 0.010, 0.09]) for y in (-0.030, 0.0, 0.030)],
    "stamp": [("stamp", [0.0, 0.0, np.pi / 4 - 1e-6, 0.012, 0.0])],
}


def run(e: ClayEnv, seq, state=None):
    s = e.state0 if state is None else state
    for tool, a in seq:
        s = e.step(s, tool, np.array(a))
    return s


# ---- 1. scripted sequences: shape, repeatability, sensitivity ------------------------
print("scripted sequences", flush=True)
h0 = geom.heightmap(env.nodes(env.state0))
finals, out["sequences"] = {}, {}
env1 = cfg.env(nthread=1, nsim=2)                      # a second, independent environment on one thread
for name, seq in SEQS.items():
    s = run(env, seq)
    finals[name] = s
    again = run(env1, seq)
    # sensitivity: move every action by 0.1 mm in x, y and depth
    seq_p = [(t, (np.array(a) + np.array([1e-4, 1e-4, 0.0, 1e-4, 0.0])).tolist()) for t, a in seq]
    sp = run(env, seq_p)
    h, hp = geom.heightmap(env.nodes(s)), geom.heightmap(env.nodes(sp))
    n = env.nodes(s)
    out["sequences"][name] = dict(
        actions=[[t, list(map(float, a))] for t, a in seq],
        deepest_mm=mm((h0 - h).max()), highest_rise_mm=mm((h - h0).max()),
        side_bulge_mm=mm(np.abs(n[..., :2]).max() - cfg.clay.size[0] / 2),
        volume_ratio=float(plots.volume(geom, n) / plots.volume(geom, env.nodes(env.state0))),
        repeat_bit_identical=bool(np.array_equal(s, again)),
        sensitivity_mm_per_0p1mm=mm(np.abs(h - hp).max()),
    )
    print(" ", name, out["sequences"][name], flush=True)

# ---- 2. permanence: plastic against elastic --------------------------------------------
print("permanence", flush=True)
press = np.array([0.0, 0.0, 0.0, 0.010, 0.0])
hold = 2.0


def centre_trace(e: ClayEnv):
    """Height of the top surface at the press centre during the primitive and ``hold`` s after."""
    g, trace = e.geom, []
    cb = lambda m, d: trace.append(float(g.heightmap(g.nodes_from_qpos(d.qpos))[g.map.size // 2 - 1:g.map.size // 2 + 1,
                                                                             g.map.size // 2 - 1:g.map.size // 2 + 1].mean()))
    s = e.step_reference(e.state0, "press", press, on_step=cb)
    m, d = e.models["press"], e._data["press"]
    for _ in range(int(round(hold / m.opt.timestep))):
        mujoco.mj_step(m, d)
        cb(m, d)
    return np.array(trace), s


env_el = ClayEnv(dataclasses.replace(cfg.clay, yield_strain=0.0), cfg.sim, cfg.prim, cfg.map, nsim=2)
tr_pl, s_press = centre_trace(env)
tr_el, _ = centre_trace(env_el)
dt = cfg.sim.timestep
n_prim = cfg.prim.nsteps("press", dt)
top0 = float(tr_pl[0])
out["permanence"] = dict(
    press=press.tolist(), hold_s=hold,
    plastic=dict(max_depth_mm=mm(top0 - tr_pl.min()), after_primitive_mm=mm(top0 - tr_pl[n_prim - 1]),
                 after_hold_mm=mm(top0 - tr_pl[-1]), drift_during_hold_mm=mm(abs(tr_pl[-1] - tr_pl[n_prim - 1]))),
    elastic=dict(max_depth_mm=mm(float(tr_el[0]) - tr_el.min()), after_primitive_mm=mm(float(tr_el[0]) - tr_el[n_prim - 1]),
                 after_hold_mm=mm(float(tr_el[0]) - tr_el[-1])),
    nplastic=int(env.models["press"].nplastic), nplastic_elastic=int(env_el.models["press"].nplastic),
)
print(" ", out["permanence"], flush=True)
plots.permanence(np.arange(len(tr_pl)) * dt, tr_pl, tr_el, n_prim * dt, cfg, RESULTS / "m0_permanence.png")

# ---- 3. save / restore ---------------------------------------------------------------------
print("save/restore", flush=True)
lay = env.layout
a2 = np.array([-0.020, 0.020, -0.4, 0.008, 0.0])
ref = env.step(s_press, "press", a2)
replay = env.step(s_press.copy(), "press", a2)
single = env.step_reference(s_press, "press", a2)
res = dict(full_state_size=lay.size, components={k: v.stop - v.start for k, v in lay.slices.items() if v.stop > v.start},
           replay_bit_identical=bool(np.array_equal(ref, replay)),
           batch_equals_single_thread=bool(np.array_equal(ref, single)), omitted={})
for comp in ("plastic", "warmstart", "qvel"):
    part = s_press.copy()
    part[lay[comp]] = env.state0[lay[comp]] if comp == "plastic" else 0.0
    alt = env.step(part, "press", a2)
    res["omitted"][comp] = dict(bit_identical=bool(np.array_equal(alt, ref)),
                                max_node_diff_mm=mm(np.abs(env.nodes(alt) - env.nodes(ref)).max()),
                                max_height_diff_mm=mm(np.abs(geom.heightmap(env.nodes(alt)) - geom.heightmap(env.nodes(ref))).max()))
out["restore"] = res
print(" ", res, flush=True)

# ---- 4. continuous execution against rollouts without the free-space moves ----------------
print("continuous", flush=True)
out["continuous"] = {}
for name, seq in SEQS.items():
    tool = seq[0][0]
    s_cont = env.run_continuous(env.state0, tool, [a for _, a in seq])
    d = geom.heightmap(env.nodes(s_cont)) - geom.heightmap(env.nodes(finals[name]))
    out["continuous"][name] = dict(max_height_diff_mm=mm(np.abs(d).max()), rms_height_diff_mm=mm(np.sqrt((d ** 2).mean())))
print(" ", out["continuous"], flush=True)

# ---- 5. tool tracking at the bottom of the stroke ---------------------------------------
print("tracking", flush=True)
out["tracking"] = {}
n_down, n_dwell = int(round(cfg.prim.t_down / dt)), int(round(cfg.prim.t_dwell / dt))
TRACK = [("press", [0.0, 0.0, 0.0, 0.010, 0.0]), ("stamp", [0.0, 0.0, 0.6, 0.010, 0.0]), ("roller", [0.0, 0.0, 0.3, 0.010, 0.08])]


def track(e: ClayEnv, tool: str, a) -> dict:
    q0, ctrl = e.planner.reference(tool, np.array(a))
    log = []

    def cb(m, d):
        mujoco.mj_kinematics(m, d)
        p_ref, _ = e.kin.fk(ctrl[len(log), 0::2])
        log.append(np.concatenate([d.site_xpos[e.ix.tip_site] - p_ref, d.actuator_force[:7], [d.site_xpos[e.ix.tip_site][2]]]))

    e.step_reference(e.state0, tool, np.array(a), on_step=cb)
    log = np.array(log)
    return dict(err_end_of_dwell_mm=[mm(v) for v in log[n_down + n_dwell - 1, :3]], err_max_mm=[mm(v) for v in np.abs(log[:, :3]).max(0)],
                torque_max_nm=[float(v) for v in np.abs(log[:, 3:10]).max(0)], depth_reached_mm=mm(cfg.clay.top - log[:, 10].min()),
                depth_commanded_mm=mm(a[3]))


for tool, a in TRACK:
    out["tracking"][tool] = track(env, tool, a)
    print(" ", tool, out["tracking"][tool], flush=True)
# the same press with the integral action off (plain PD on position and velocity error)
env_pd = ClayEnv(cfg.clay, dataclasses.replace(cfg.sim, ki_time=0.0), cfg.prim, cfg.map, nsim=2)
out["tracking_pd_only"] = {"press": track(env_pd, *TRACK[0])}
print("  PD only, press:", out["tracking_pd_only"]["press"], flush=True)
del env_pd

# ---- 6. speed -------------------------------------------------------------------------------
print("speed", flush=True)
rng = np.random.default_rng(0)
from clay_sculptor.control.primitives import bounds  # noqa: E402

out["speed"] = dict(nv=int(env.models["press"].nv), nvert=int(env.models["press"].nflexvert), steps={t: cfg.prim.nsteps(t, dt) for t in TOOLS})
for nth, n in [(1, 4), (8, 32), (16, 32)]:
    e = cfg.env(nthread=nth, nsim=32)
    for tool in ("press", "roller"):
        lo, hi = bounds(tool, cfg.prim)
        acts = rng.uniform(lo, hi, size=(n, 5))
        t0 = time.time()
        e.rollout(e.state0, tool, acts)
        out["speed"][f"{tool}_s_per_rollout_{nth}thread"] = (time.time() - t0) / n
    del e
print(" ", out["speed"], flush=True)

# ---- figures and video ----------------------------------------------------------------------
np.savez_compressed(RESULTS / "m0_shapes.npz", untouched=env.nodes(env.state0).astype(np.float32),
                    trace_plastic=tr_pl, trace_elastic=tr_el, **{k: env.nodes(v).astype(np.float32) for k, v in finals.items()})
plots.sequence_maps(geom, h0, {k: geom.heightmap(env.nodes(v)) for k, v in finals.items()}, RESULTS / "m0_shapes.png")
shots = [render.label(render.snapshot(env, finals[k], SEQS[k][-1][0], "close"), k) for k in SEQS]
render.save_png(RESULTS / "m0_render.png", render.grid(shots, 3))
if not args.no_video:
    frames = []
    for name, seq in SEQS.items():
        frames += render.record(env, env.state0, seq[0][0], [a for _, a in seq], title=name)
    render.write_mp4(RESULTS / "m0_bench.mp4", frames, fps=30)
    out["video"] = "results/m0_bench.mp4"

(RESULTS / "m0_bench.json").write_text(json.dumps(out, indent=1))
print("wrote results/m0_bench.json", flush=True)
