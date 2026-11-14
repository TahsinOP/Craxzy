"""Planner pieces: bounds, yaw wrapping, circular statistics, CEM on a known optimum."""
import numpy as np
import pytest

from clay_sculptor.control.mpc import CEMCfg, _circular_stats, cem, guided_proposal, wrap_action
from clay_sculptor.control.primitives import PrimCfg, bounds, minjerk, travel_direction, yaw_period
from clay_sculptor.envs.clay_env import ClayEnv


@pytest.fixture(scope="module")
def env():
    return ClayEnv(nsim=4)


def test_minjerk_endpoints_and_symmetry():
    s = minjerk(40)
    assert s[-1] == pytest.approx(1.0) and 0 < s[0] < 1e-3
    assert np.allclose(s[:-1] + s[-2::-1], 1.0)          # s(tau) + s(1 - tau) = 1


def test_yaw_wraps_instead_of_clipping():
    prim = PrimCfg()
    for tool in ("press", "stamp", "roller"):
        lo, hi = bounds(tool, prim)
        per = yaw_period(tool)
        assert hi[2] - lo[2] == pytest.approx(per)
        a = np.array([0.0, 0.0, hi[2] + 0.1, 0.005, lo[4]])
        assert wrap_action(a, tool, prim)[2] == pytest.approx(lo[2] + 0.1)
        a[2] = lo[2] - 0.1
        assert wrap_action(a, tool, prim)[2] == pytest.approx(hi[2] - 0.1)


def test_circular_mean_across_the_wrap():
    per = np.pi / 2
    x = np.array([np.pi / 4 - 0.05, -np.pi / 4 + 0.05, np.pi / 4 - 0.02, -np.pi / 4 + 0.02])   # all near +-pi/4, the same yaw
    mean, std = _circular_stats(x, per)
    assert abs(abs(mean) - np.pi / 4) < 1e-9          # not 0, which the arithmetic mean would give
    assert std < 0.05


def test_travel_direction_is_the_tool_y_axis():
    from clay_sculptor.control.arm import tip_rotation

    for psi in (0.0, 0.7, -1.2):
        assert np.allclose(travel_direction(psi), tip_rotation(psi)[:2, 1])


def test_cem_finds_a_known_optimum(env):
    prim = env.prim
    best = np.array([0.02, -0.03, 0.2, 0.009, 0.0])
    lo, hi = bounds("press", prim)
    span = np.where(hi > lo, hi - lo, 1.0)

    def cost(a):
        d = (a - best) / span
        d[:, 2] = np.sin(np.pi * (a[:, 2] - best[2]) / yaw_period("press")) / np.pi      # periodic in the yaw
        return (d ** 2).sum(axis=1)

    h = np.zeros((32, 32))
    res = cem(cost, "press", np.random.default_rng(0), CEMCfg(pop=64, iters=6, guided=0.0), h, h, env)
    assert res["costs"][0] < 2e-3
    assert np.all(np.diff(res["costs"]) >= 0)


def test_guided_proposal_goes_where_the_clay_is_too_high(env):
    g = env.geom.map
    h_cur = np.full((g.size, g.size), env.clay.top)
    h_tgt = h_cur.copy()
    h_tgt[20:26, 8:12] -= 0.008                       # target pocket
    a = guided_proposal(np.random.default_rng(0), h_cur, h_tgt, "press", 200, env)
    ax = g.axis
    assert abs(a[:, 0].mean() - ax[20:26].mean()) < 0.004 and abs(a[:, 1].mean() - ax[8:12].mean()) < 0.004
    assert abs(a[:, 3].mean() - 0.008) < 0.0005       # tool bottom at the target height
