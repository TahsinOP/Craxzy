"""M0 claims: kinematics, shape maps, state save/restore."""
import mujoco
import numpy as np
import pytest

from clay_sculptor.control.arm import HOME_YAW, Kinematics, tip_rotation, wrap_yaw
from clay_sculptor.control.primitives import PrimCfg, bounds
from clay_sculptor.envs.clay_env import ClayEnv, StateLayout
from clay_sculptor.models.scene import HOME_Q, TOOLS


@pytest.fixture(scope="module")
def env():
    return ClayEnv(nsim=8)


def test_home_pose_points_down():
    kin = Kinematics()
    p, r = kin.fk(HOME_Q)
    assert np.allclose(r[:, 2], [0, 0, -1], atol=1e-3)
    assert np.allclose(r, tip_rotation(HOME_YAW), atol=2e-3)


def test_ik_reaches_whole_workspace():
    kin = Kinematics()
    cfg = PrimCfg()
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, y = rng.uniform(-0.105, 0.105, 2)        # action centre +- half the longest roll
        z = rng.uniform(0.030, 0.060)
        psi = rng.uniform(-np.pi, np.pi)
        p = np.array([0.5 + x, y, z])
        q = kin.ik(p, psi)
        pf, rf = kin.fk(q)
        assert np.abs(pf - p).max() < 1e-6
        assert np.abs(rf - tip_rotation(wrap_yaw(psi))).max() < 1e-5
        assert (q > kin.lo + 0.05).all() and (q < kin.hi - 0.05).all(), "stays clear of the joint limits"


def test_state_layout_matches_mujoco(env):
    m, d = env.models["press"], mujoco.MjData(env.models["press"])
    rng = np.random.default_rng(1)
    d.qpos[:] = rng.normal(size=m.nq) * 1e-3
    d.qvel[:] = rng.normal(size=m.nv)
    d.act[:] = rng.normal(size=m.na)
    d.plastic[:] = rng.normal(size=m.nplastic)
    d.ctrl[:] = rng.normal(size=m.nu)
    s = env.get_state(d)
    lay = env.layout
    for name, field in [("qpos", d.qpos), ("qvel", d.qvel), ("act", d.act), ("plastic", d.plastic), ("ctrl", d.ctrl)]:
        assert np.array_equal(s[lay[name]], field), name
    assert m.nplastic == 6 * np.prod(env.clay.cells)


def test_tool_models_share_one_state_layout(env):
    ref = StateLayout(env.models["press"]).slices
    for t in TOOLS:
        assert StateLayout(env.models[t]).slices == ref
        assert env.models[t].nq == env.models["press"].nq


def test_vertices_are_trilinear_in_the_nodes(env):
    m, d = env.models["press"], env._data["press"]
    s = env.step(env.state0, "press", np.array([0.01, -0.02, 0.3, 0.010, 0.0]))
    env.set_state(d, s)
    mujoco.mj_forward(m, d)
    v = env.geom.vertices(env.nodes(s)).reshape(-1, 3) + env.geom.origin
    assert np.abs(v - d.flexvert_xpos).max() < 1e-12
    # and the inverse map qpos <- nodes
    q = env.geom.qpos_from_nodes(env.nodes(s), s[env.layout["qpos"]])
    assert np.abs(q - s[env.layout["qpos"]]).max() < 1e-15      # (node0 + d) - node0 is exact to round-off


def test_heightmap_exact_on_a_sheared_plane(env):
    g = env.geom
    n = g.node0.copy()
    # shear the surface sideways by a smooth field, then tilt it: z = c + a x + b y in the
    # *deformed* coordinates.  Bilinear patches keep a linear relation between x, y and z
    # exactly, so the recovered height must be the plane whatever the shear.
    frac = n[..., 2] / n[..., 2].max()
    n[..., 0] += 0.004 * np.sin(20 * n[..., 1]) * frac
    n[..., 2] += frac * (0.05 * n[..., 0] - 0.03 * n[..., 1])
    h = g.heightmap(n)
    ax = g.map.axis
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    top0 = g.node0[..., 2].max()
    assert np.abs(h - (top0 + 0.05 * X - 0.03 * Y + g.radius)).max() < 1e-9


def test_batch_equals_single_thread_bit_for_bit(env):
    a = np.array([0.01, -0.02, 0.3, 0.010, 0.0])
    assert np.array_equal(env.step_reference(env.state0, "press", a), env.step(env.state0, "press", a))


def test_save_restore_replays_bit_for_bit(env):
    s1 = env.step(env.state0, "press", np.array([0.01, -0.02, 0.3, 0.010, 0.0]))
    saved = s1.copy()
    a2 = np.array([-0.02, 0.02, -0.4, 0.008, 0.0])
    s2 = env.step(s1, "press", a2)
    env.step(env.state0, "stamp", np.array([0.0, 0.0, 0.7, 0.012, 0.0]))     # unrelated work in between
    assert np.array_equal(env.step(saved, "press", a2), s2)


def test_restore_without_plastic_state_diverges(env):
    lay = env.layout
    s1 = env.step(env.state0, "press", np.array([0.01, -0.02, 0.3, 0.010, 0.0]))
    a2 = np.array([-0.02, 0.02, -0.4, 0.008, 0.0])
    s2 = env.step(s1, "press", a2)
    partial = s1.copy()
    partial[lay["plastic"]] = env.state0[lay["plastic"]]
    s2p = env.step(partial, "press", a2)
    assert np.abs(env.nodes(s2p) - env.nodes(s2)).max() > 1e-3     # millimetres, not round-off


def test_dent_is_permanent(env):
    s = env.step(env.state0, "press", np.array([0.0, 0.0, 0.0, 0.010, 0.0]))
    h0, h1 = env.geom.heightmap(env.nodes(env.state0)), env.geom.heightmap(env.nodes(s))
    assert (h0 - h1).max() > 0.005
    # hold for one more second with the tool away: the dent does not recover
    m, d = env.models["press"], env._data["press"]
    env.set_state(d, s)
    for _ in range(200):
        mujoco.mj_step(m, d)
    h2 = env.geom.heightmap(env.nodes(env.get_state(d)))
    assert np.abs(h2 - h1).max() < 5e-5
