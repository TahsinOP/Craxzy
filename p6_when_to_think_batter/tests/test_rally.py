"""The rally: scene, mirroring, the bounce-aware flight model."""
import mujoco
import numpy as np
import pytest

from batter import config, rally
from batter.control.arm import Arm
from batter.estim.flight import FlightModel
from batter.rally.flight import BounceModel, TableFlight, mirrored
from batter.rally.scene import RallyIndex, build_arm_model, build_rally_model, to_arm_frame


@pytest.fixture(scope="module")
def world():
    cfg = rally.load(config.load())
    m = build_rally_model(cfg)
    fm = FlightModel.identify(m)
    return cfg, m, fm, BounceModel.identify(m, fm, cfg["table"])


def _simulate(m, x0, n):
    ix = RallyIndex.of(m)
    d = mujoco.MjData(m)
    a, dd = ix.ball_qadr, ix.ball_dadr
    d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = x0[:3], [1, 0, 0, 0]
    d.qvel[dd:dd + 3], d.qvel[dd + 3:dd + 6] = x0[3:6], x0[6:9]
    out = [x0.copy()]
    for _ in range(n):
        mujoco.mj_step(m, d)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[a + 3:a + 7])
        out.append(np.concatenate([d.qpos[a:a + 3], d.qvel[dd:dd + 3], R.reshape(3, 3) @ d.qvel[dd + 3:dd + 6]]))
    return np.array(out)


def test_bounce_model_is_the_pairs_restitution(world):
    cfg, m, fm, bm = world
    assert abs(bm.e - cfg["table"]["restitution"]) < 0.005
    assert abs(bm.b) < 0.005


def test_flight_through_a_bounce_matches_the_simulator(world):
    """Balls with spin on arm A's side of the net are predicted to 0.05 mm until they reach arm A,
    through their bounce on the table (the scatter of the restitution offset, about 0.1 mm/s,
    accounts for the 1e-5 m that a bounce leaves)."""
    cfg, m, fm, bm = world
    tf = TableFlight(fm, bm)
    rng = np.random.default_rng(0)
    n_bounced = 0
    for _ in range(5):
        x_net = cfg["table"]["gap"] + cfg["table"]["length"] / 2
        x0 = np.array([x_net - 0.05, rng.uniform(-0.2, 0.2), rng.uniform(0.95, 1.10), -rng.uniform(3.0, 4.5), rng.uniform(-0.3, 0.3),
                       rng.uniform(-1.5, 0.0), *rng.normal(0, 30, 3)])
        sim = _simulate(m, x0, 60)
        pred = tf.rollout(x0, 60)
        keep = sim[:, 0] > 0.9                              # until the ball reaches arm A (it hits nothing else)
        n_bounced += int((np.diff(np.sign(sim[keep, 5])) > 0).sum() >= 1)
        assert np.abs(sim[keep, :3] - pred[keep, :3]).max() < 5e-5
    assert n_bounced >= 3


def test_jacobian_through_a_bounce(world):
    cfg, m, fm, bm = world
    tf = TableFlight(fm, bm)
    x = np.array([1.5, 0.1, bm.z_surface + 0.01, -4.0, 0.2, -3.0, 10.0, -40.0, 5.0])
    assert tf.bounces(x, fm.step(x))
    F = tf.jacobian(x)
    num = np.zeros((9, 9))
    for i in range(9):
        e = np.zeros(9)
        e[i] = 1e-7
        num[:, i] = (tf.step(x + e) - tf.step(x - e)) / 2e-7
    assert np.abs(F - num).max() < 1e-5


def test_arm_b_is_arm_a_in_its_own_frame(world):
    cfg, m, fm, bm = world
    ix = RallyIndex.of(m)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(1)
    arm = Arm(build_arm_model(cfg))
    for _ in range(3):
        q = rng.uniform(arm.q_min + 0.2, arm.q_max - 0.2)
        d.qpos[ix.qadr[1]] = q
        mujoco.mj_forward(m, d)
        world_pos = d.site_xpos[ix.sweet[1]]
        assert np.allclose(to_arm_frame(world_pos, 1, cfg["table"]["L"]), arm.fk(q)[0], atol=1e-9)
    bmb = mirrored(bm, cfg["table"]["L"])
    assert np.allclose(bmb.x_range, bm.x_range) and np.allclose(bmb.y_range, bm.y_range)   # the table is symmetric


def test_serves_reach_arm_a(world):
    """Every serve clears the net, bounces once on A's half and crosses A's strike plane inside its zone."""
    from batter.rally import serve

    cfg, m, fm, bm = world
    tf = TableFlight(fm, bm)
    S = serve.generate(tf, cfg, 12, 5)
    t, rc = cfg["table"], cfg["rally"]
    x_net = t["gap"] + t["length"] / 2
    for x0 in S:
        path = tf.rollout(x0, 150)
        k_net = int(np.argmax(path[:, 0] <= x_net))
        assert path[k_net, 2] > t["height"] + t["net_height"] + fm.radius
        k = int(np.argmax(path[:, 0] <= np.mean(rc["strike_x"])))
        assert rc["zone_z"][0] <= path[k, 2] <= rc["zone_z"][1]
        assert (np.diff(np.sign(path[:k, 5])) > 0).sum() == 1


def test_rally_with_the_true_ball_state(world):
    """Two arms keep two rallies going for at least four returns each when they know the ball's state."""
    from batter.rally import serve
    from batter.rally.env import RallyEnv, run_rally

    cfg = world[0]
    c = {**cfg, "rally": {**cfg["rally"], "max_hits": 4}}
    env = RallyEnv(c, 2, oracle=True, nthread=2)
    out = run_rally(env, serve.generate(env.tf, c, 2, 0))
    assert (out["n_hits"] >= 4).all(), (out["n_hits"], out["reason"])
