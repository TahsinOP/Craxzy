"""Aim solver, swing QP, strike kinematics."""
import numpy as np
import pytest

from batter import config
from batter.control.arm import NOMINAL_STRIKE, READY, Arm
from batter.control.planner import Aimer, SwingPlanner, planner_params
from batter.control.traj import SwingQP
from batter.estim.flight import FlightModel
from batter.learn import wm
from batter.learn.wm_planner import SwingDecoder
from batter.models.pitch import generate
from batter.models.scene import QD_MAX, build_cage_model


@pytest.fixture(scope="module")
def world():
    cfg = config.load()
    m = build_cage_model(cfg)
    return cfg, Arm(m), FlightModel.identify(m)


def test_aimer_lands_on_the_target(world):
    cfg, arm, fm = world
    rng = np.random.default_rng(0)
    n = 100
    p0 = np.stack([rng.uniform(0.5, 0.7, n), rng.uniform(-0.2, 0.2, n), rng.uniform(0.6, 1.3, n)], 1)
    w = rng.normal(0, 80, (n, 3))
    r, az = rng.uniform(3.5, 6.0, n), rng.uniform(-0.35, 0.35, n)
    tgt = np.stack([r * np.cos(az), r * np.sin(az)], 1)
    for T in (0.75, 0.95, 1.15):
        v = Aimer(fm).solve(p0, w, tgt, np.full(n, T))
        land, _ = fm.landing(np.concatenate([p0, v, w], 1))        # the exact model, 10 ms steps
        assert np.linalg.norm(land - tgt, axis=1).max() < 0.05


def test_swing_reaches_the_strike_and_respects_limits(world):
    cfg, arm, fm = world
    qp = SwingQP(arm.h, arm.q_min, arm.q_max)
    n = np.array([np.cos(0.4), 0.0, np.sin(0.4)])
    v_imp = arm.strike_velocity(NOMINAL_STRIKE, n, 1.5)
    plan, miss_q, miss_v, _ = qp.swing(READY, np.zeros(7), NOMINAL_STRIKE, v_imp, 45.3, k0=10)
    assert np.abs(miss_q).max() < 1e-6 and np.abs(miss_v).max() < 1e-6
    # at the (fractional) impact step the reference is at the strike, moving at the strike velocity
    q44, v44, _ = plan.ref(10 + 45)
    assert np.allclose(q44 - (45 - 45.3) * arm.h * v_imp, NOMINAL_STRIKE, atol=1e-9)
    assert np.allclose(v44, v_imp, atol=1e-9)
    # the reference obeys the simulator's integrator exactly, and the limits
    assert np.allclose(plan.q[1:], plan.q[:-1] + arm.h * plan.v[1:], atol=1e-12)
    assert np.allclose(plan.v[1:], plan.v[:-1] + arm.h * plan.a, atol=1e-12)
    assert (np.abs(plan.v) <= qp.v_max + 1e-6).all() and (np.abs(plan.a) <= qp.a_max + 1e-6).all()
    assert np.allclose(plan.v[-1], 0.0, atol=1e-9)                 # it brakes to rest


def test_soft_swing_gets_as_close_as_the_limits_allow(world):
    cfg, arm, fm = world
    qp = SwingQP(arm.h, arm.q_min, arm.q_max)
    n = np.array([np.cos(0.4), 0.0, np.sin(0.4)])
    v_imp = arm.strike_velocity(NOMINAL_STRIKE, n, 1.5)
    plan, miss_q, miss_v, _ = qp.swing(READY, np.zeros(7), NOMINAL_STRIKE, v_imp, 9.0)     # 90 ms: not reachable
    assert np.abs(miss_q).max() > 1e-3
    assert (np.abs(plan.a) <= qp.a_max + 1e-6).all() and (np.abs(plan.v) <= qp.v_max + 1e-4).all()
    dist0 = np.abs(NOMINAL_STRIKE - READY).max()
    assert np.abs(miss_q).max() < dist0                            # but it moved towards the strike


def test_strike_velocity_is_along_the_normal(world):
    cfg, arm, fm = world
    Jv, _, _, R = arm.jacobian(NOMINAL_STRIKE)
    n = R[:, 0]
    qd = arm.strike_velocity(NOMINAL_STRIKE, n, 1.2)
    assert abs(n @ (Jv @ qd) - 1.2) < 1e-9
    assert np.max(np.abs(qd) / QD_MAX) < 1.0                       # 1.2 m/s is within the joint speed limits


def test_planner_strike_geometry(world):
    """The planned bat pose puts the blade's face at the ball, facing where the impulse must go."""
    cfg, arm, fm = world
    pl = SwingPlanner(arm, fm, cfg["bat"]["thickness"], planner_params(cfg))
    P = generate(fm, cfg, 5, 3)
    for i in range(5):
        traj = fm.rollout(P.x0[i], 95)
        plan = pl.plan(traj[8:], 8, READY, np.zeros(7), P.target[i])
        assert plan is not None
        info = plan.info
        pos, R = arm.fk(info["q_imp"])
        assert np.linalg.norm(pos + pl.offset * info["n"] - info["p_b"]) < 5e-3
        assert np.arccos(np.clip(R[:, 0] @ info["n"], -1, 1)) < 3e-2
        # impact model: the commanded speed turns v_b into v_out
        e = pl.prm.restitution
        v_out = info["v_b"] - (1 + e) * ((info["v_b"] - info["speed"] * info["n"]) @ info["n"]) * info["n"]
        assert np.allclose(v_out, info["v_out"], atol=1e-9)


def test_swing_parameterisation_round_trip(world):
    cfg, arm, fm = world
    u = np.array([0.4, 0.05, 0.95, 0.1, 0.45, 1.0])
    assert np.allclose(wm.denormalize_u(wm.normalize_u(u)), u)
    n = wm.normal_from_angles(u[3], u[4])
    assert np.allclose(wm.angles_from_normal(n), (u[3], u[4]))
    dec = SwingDecoder(arm, fm.radius + cfg["bat"]["thickness"] / 2)
    plan = dec.decode(u, 0, READY, np.zeros(7), None)
    pos, R = arm.fk(plan.info["q_imp"])
    assert np.linalg.norm(pos + dec.offset * n - np.array([wm.STRIKE_X, u[1], u[2]])) < 5e-3
