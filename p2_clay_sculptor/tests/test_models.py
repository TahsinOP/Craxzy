"""Learned-model plumbing and the prior's defining properties."""
import numpy as np
import pytest
import torch

from clay_sculptor.envs.clay_env import ClayEnv
from clay_sculptor.learn.models import DynamicsModel
from clay_sculptor.learn.prior import SweptVolumePrior
from clay_sculptor.models.scene import TOOLS


@pytest.fixture(scope="module")
def env():
    return ClayEnv(nsim=4)


def _batch(env, b=3):
    nodes = torch.from_numpy(np.repeat(env.nodes(env.state0)[None], b, 0).astype(np.float32))
    tool = torch.tensor([0, 1, 2])[:b]
    act = torch.tensor([[0.01, -0.02, 0.3, 0.010, 0.0], [0.0, 0.01, -0.5, 0.008, 0.06], [-0.01, 0.0, 0.6, 0.012, 0.0]])[:b]
    return nodes, tool, act


@pytest.mark.parametrize("kind", ["prior", "gnn", "structured", "unet"])
def test_shapes_and_pinned_nodes(env, kind):
    torch.manual_seed(0)
    model = DynamicsModel(kind, env.geom).eval()
    nodes, tool, act = _batch(env)
    x = model.encode(nodes)
    with torch.no_grad():
        y = model(x, tool, act)
    assert y.shape == x.shape
    assert model.heightmap(y).shape == (3, env.geom.map.size, env.geom.map.size)
    if model.state_kind == "nodes":
        assert torch.equal(y[:, :, :, 0], nodes[:, :, :, 0])           # the pinned layer never moves


def test_prior_never_raises_and_only_acts_under_the_tool(env):
    prior = SweptVolumePrior(env.geom)
    nodes, tool, act = _batch(env)
    out = prior(nodes, tool, act)
    verts0, verts1 = env.geom.vertices(nodes), env.geom.vertices(out)
    dz = (verts1 - verts0)[..., 2]
    assert dz.max() < 1e-7                                   # a weighted mean of non-positive displacements: nothing rises
    assert dz.min() < -0.004                                 # it does press
    assert torch.allclose(out[..., :2], nodes[..., :2])      # nothing moves sideways: no bulge in the prior
    # far from the tool nothing changes
    far = (verts0[0, ..., 0] - 0.01).abs() + (verts0[0, ..., 1] + 0.02).abs() > 0.07
    assert dz[0][far].abs().max() < 1e-7


def test_prior_beats_no_change_on_a_real_press(env):
    """One simulated press: the prior's height map is closer to the truth than doing nothing."""
    a = np.array([0.01, -0.02, 0.3, 0.010, 0.0])
    s1 = env.step(env.state0, "press", a)
    n0 = torch.from_numpy(env.nodes(env.state0).astype(np.float32))[None]
    prior = SweptVolumePrior(env.geom)
    pred = prior(n0, torch.tensor([TOOLS.index("press")]), torch.tensor(a[None], dtype=torch.float32))
    h_true = env.geom.heightmap(env.nodes(s1))
    h_prior = env.geom.heightmap(pred[0].double().numpy())
    h_none = env.geom.heightmap(env.nodes(env.state0))
    rm = lambda x: np.sqrt(((x - h_true) ** 2).mean())
    assert rm(h_prior) < rm(h_none)


def test_graph_has_the_grid_edges(env):
    model = DynamicsModel("gnn", env.geom)
    nx, ny, nz = env.geom.nn
    n_undirected = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1)
    assert model.net.send.numel() == 2 * n_undirected
