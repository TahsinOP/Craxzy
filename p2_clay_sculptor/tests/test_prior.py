"""The geometric prior describes the tool that MuJoCo actually simulates."""
import mujoco
import numpy as np
import pytest
import torch

from clay_sculptor.control.arm import Kinematics
from clay_sculptor.control.primitives import travel_direction
from clay_sculptor.envs.clay_env import quiet_build
from clay_sculptor.learn.prior import envelope
from clay_sculptor.models.scene import HEAD_GROUP, TOOLS, ClayCfg, SimCfg


def lowest_surface(tool: str, tip: np.ndarray, psi: float, pts: np.ndarray) -> np.ndarray:
    """Height of the tool head above table points, by casting rays upwards at the IK pose."""
    m = quiet_build(tool, None, SimCfg())
    d = mujoco.MjData(m)
    kin = Kinematics()
    d.qpos[:7] = kin.ik(tip, psi)
    mujoco.mj_forward(m, d)
    group = np.zeros(6, dtype=np.uint8)
    group[HEAD_GROUP] = 1
    gid = np.zeros(1, dtype=np.int32)
    out = np.full(len(pts), np.inf)
    for i, p in enumerate(pts):
        dist = mujoco.mj_ray(m, d, np.array([p[0], p[1], 1e-4]), np.array([0.0, 0.0, 1.0]), group, 1, -1, gid)
        if dist >= 0:
            out[i] = 1e-4 + dist
    return out


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("psi", [0.0, 0.5, -0.7])
def test_envelope_matches_the_simulated_tool(tool, psi):
    clay = ClayCfg()
    depth = 0.009
    a = np.array([0.012, -0.008, psi, depth, 0.0])
    centre = np.array(clay.centre)
    tip = np.array([*(centre + a[:2]), clay.top - depth])
    rng = np.random.default_rng(0)
    pts_rel = rng.uniform(-0.055, 0.055, size=(400, 2))                 # clay coordinates
    truth = lowest_surface(tool, tip, psi, pts_rel + centre)
    env = envelope(torch.tensor(pts_rel[None]), torch.tensor([TOOLS.index(tool)]), torch.tensor(a[None]), clay.top)[0].numpy()
    hit, pred = np.isfinite(truth), np.isfinite(env)
    # the footprint agrees except within 0.6 mm of its outline (mesh facets, grazing rays)
    both = hit & pred
    assert both.sum() > 40
    assert (hit != pred).mean() < 0.03
    # heights agree where the tool surface is not near-vertical (the rounded rim is a polyhedron in the mesh)
    flat = both & (env < clay.top - depth + 0.004)
    assert np.abs(truth[flat] - env[flat]).max() < 3e-4
    assert np.abs(truth[both] - env[both]).mean() < 2e-4


def test_roller_envelope_is_the_union_over_the_travel():
    clay = ClayCfg()
    psi, depth, length = 0.4, 0.008, 0.07
    a = np.array([0.0, 0.01, psi, depth, length])
    centre = np.array(clay.centre)
    rng = np.random.default_rng(1)
    pts_rel = rng.uniform(-0.06, 0.06, size=(300, 2))
    truth = np.full(len(pts_rel), np.inf)
    for s in np.linspace(-0.5, 0.5, 71):                                 # 1 mm steps along the travel
        tip = np.array([*(centre + a[:2] + s * length * travel_direction(psi)), clay.top - depth])
        truth = np.minimum(truth, lowest_surface("roller", tip, psi, pts_rel + centre))
    env = envelope(torch.tensor(pts_rel[None]), torch.tensor([TOOLS.index("roller")]), torch.tensor(a[None]), clay.top)[0].numpy()
    both = np.isfinite(truth) & np.isfinite(env)
    assert (np.isfinite(truth) != np.isfinite(env)).mean() < 0.03
    assert np.abs(truth[both] - env[both]).mean() < 2e-4
    core = both & (env < clay.top - depth + 0.004)
    assert np.abs(truth[core] - env[core]).max() < 3e-4
