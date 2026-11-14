"""M0 design check: which flex discretisation can be pressed, released and trusted?

    env -u PYTHONPATH .venv/bin/python scripts/m0_discretisation.py

The same press (flat press, 8 mm edge radius) on the same block with
  * the trilinear cage used everywhere else (12 x 12 x 3 cells),
  * a coarser cage (12 x 12 x 2),
  * a full-vertex flex (every vertex a body, plastic strain per tetrahedron), at two Poisson ratios.
Measured: cost per step, how much of the stroke stays, how far vertices still move
in the 1.5 s after the primitive (creep), and the smallest tetrahedron volume
relative to its reference (<= 0 means an inverted element).
"""
import dataclasses
import json
import time
import warnings

import mujoco
import numpy as np

from clay_sculptor import RESULTS, config
from clay_sculptor.control.arm import Kinematics
from clay_sculptor.control.primitives import Planner
from clay_sculptor.models.scene import Index, build_model

cfg = config.load()
kin = Kinematics(cfg.sim)
planner = Planner(kin, cfg.clay, cfg.prim, cfg.sim.timestep)


def tet_volume(m, x):
    e = m.flex_elem.reshape(-1, 4)
    a, b, c, d = (x[e[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6


def run(label, clay, depth):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_model("press", clay, cfg.sim)
    ix, d = Index.of(m), mujoco.MjData(m)
    a = np.array([0.0, 0.0, 0.0, depth, 0.0])
    q0, ctrl = planner.reference("press", a)
    d.qpos[ix.arm_qpos] = q0
    mujoco.mj_forward(m, d)
    v0 = d.flexvert_xpos.copy()
    vol0 = tet_volume(m, v0)
    top = v0[:, 2] > v0[:, 2].max() - 1e-6
    under = top & (np.abs(v0[:, 0] - cfg.clay.centre[0]) < 0.011) & (np.abs(v0[:, 1]) < 0.011)
    t0, zmin, warn = time.time(), np.inf, 0
    for u in ctrl:
        d.ctrl[:] = u
        mujoco.mj_step(m, d)
        zmin = min(zmin, d.flexvert_xpos[under, 2].mean())
    wall = (time.time() - t0) / len(ctrl)
    v1 = d.flexvert_xpos.copy()
    for _ in range(int(round(1.5 / m.opt.timestep))):
        mujoco.mj_step(m, d)
    v2 = d.flexvert_xpos.copy()
    r1, r2 = tet_volume(m, v1) / vol0, tet_volume(m, v2) / vol0
    stroke = v0[under, 2].mean() - zmin
    res = dict(label=label, depth_cmd_mm=1e3 * depth, dof=int(m.nv - 7), nplastic=int(m.nplastic), ms_per_step=1e3 * wall,
               stroke_mm=1e3 * stroke, dent_mm=1e3 * (v0[under, 2] - v1[under, 2]).mean(),
               retained=float((v0[under, 2] - v1[under, 2]).mean() / stroke),
               creep_mm=1e3 * float(np.abs(v2 - v1).max()), min_volume_ratio=float(r2.min()),
               inverted=int((r2 <= 0).sum()), warnings=int(d.warning.number.sum()))
    print(f"{label:34s} depth {1e3 * depth:4.0f} mm: {res['dof']:5d} dof, {res['ms_per_step']:5.2f} ms/step, dent {res['dent_mm']:5.2f} of {res['stroke_mm']:5.2f} mm, "
          f"creep {res['creep_mm']:7.3f} mm, min volume ratio {res['min_volume_ratio']:6.2f}, inverted {res['inverted']}", flush=True)
    return res


c = cfg.clay
variants = [
    ("trilinear cage 12x12x3 (used)", c),
    ("trilinear cage 12x12x2", dataclasses.replace(c, cells=(12, 12, 2))),
    ("full vertices 15x15x4, nu 0.3", dataclasses.replace(c, dof="full", count=(15, 15, 4))),
    ("full vertices 15x15x4, nu 0", dataclasses.replace(c, dof="full", count=(15, 15, 4), poisson=0.0)),
    # is the spring-back of the cage a material property?  Halve and double the yield strain.
    ("cage 12x12x3, yield strain 0.01", dataclasses.replace(c, yield_strain=0.01)),
    ("cage 12x12x3, yield strain 0.04", dataclasses.replace(c, yield_strain=0.04)),
]
out = [run(label, clay, depth) for label, clay in variants for depth in (0.010, 0.014)]
(RESULTS / "m0_discretisation.json").write_text(json.dumps(out, indent=1))
print("wrote results/m0_discretisation.json")
