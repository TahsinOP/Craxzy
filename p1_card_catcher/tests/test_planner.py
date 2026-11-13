"""Trajectory QP: minimum-time formula, closed-form fast path, limits, exact trackability."""
import mujoco
import numpy as np
import pytest
from scipy.optimize import linprog

from card_catcher.control.arm import Arm
from card_catcher.control.planner import TrajectoryQP, min_transfer_time
from card_catcher.models.scene import PARK_POS, build_catch_model


@pytest.fixture(scope="module")
def arm():
    return Arm(build_catch_model("ball"))


def _random_case(rng, qp):
    j = int(rng.integers(7))
    mid = 0.5 * (qp.q_min[j] + qp.q_max[j])
    q0 = mid + rng.uniform(-0.4, 0.4)
    v0 = rng.uniform(-1, 1) * qp.v_max[j]
    qg = q0 + rng.uniform(-0.8, 0.8)
    return j, q0, v0, qg


def test_min_transfer_time_matches_lp_feasibility(arm):
    """Discrete problem is feasible just above the continuous minimum time and not below it."""
    qp = TrajectoryQP(arm.h, arm.q_min, arm.q_max)
    rng = np.random.default_rng(0)
    h = arm.h
    feas_above, feas_below, n_cases = 0, 0, 60
    for _ in range(n_cases):
        j, q0, v0, qg = _random_case(rng, qp)
        n = int(np.ceil(float(min_transfer_time(qg - q0, v0, qp.v_max[j], qp.a_max[j])) / h))
        for dn in (-4, 3):
            N = max(n + dn, 2)
            k = np.arange(1, N + 1)
            drift = q0 + k * h * v0
            L = np.tril(np.ones((N, N)))
            G, E = np.vstack([h * L, h * h * np.cumsum(L, axis=0)]), qp.terminal_rows(N)
            ub = np.r_[np.full(N, qp.v_max[j] - v0), qp.q_max[j] - drift, np.full(N, qp.v_max[j] + v0), drift - qp.q_min[j]]
            r = linprog(np.zeros(N), A_ub=np.vstack([G, -G]), b_ub=ub, A_eq=E, b_eq=[-v0, qg - drift[-1]],
                        bounds=[(-qp.a_max[j], qp.a_max[j])] * N, method="highs")
            if dn > 0:
                feas_above += r.status == 0
            else:
                feas_below += r.status == 0
    assert feas_above == n_cases
    assert feas_below <= 0.1 * n_cases   # only near the braking curve, where the discrete scheme stops shorter


def test_closed_form_is_the_qp_optimum_when_limits_are_inactive(arm):
    """Closed form == OSQP on the full problem for gentle moves, and it is a true minimiser:
    P a - E^T lam = 0 with E a = b (stationarity of the equality-constrained problem)."""
    qp = TrajectoryQP(arm.h, arm.q_min, arm.q_max, osqp_max_iter=4000)
    rng = np.random.default_rng(1)
    for _ in range(20):
        j, q0, _, _ = _random_case(rng, qp)
        v0, qg, N = rng.uniform(-0.2, 0.2), q0 + rng.uniform(-0.15, 0.15), int(rng.integers(40, 80))
        a = qp.closed_form(q0, v0, qg, N)
        assert qp.within_limits(j, a, q0, v0), "expected a gentle move"
        E, P = qp.terminal_rows(N), qp.cost_matrix(N)
        assert np.allclose(E @ a, [-v0, qg - q0 - N * arm.h * v0], atol=1e-10)
        lam = np.linalg.lstsq(E.T, P @ a, rcond=None)[0]
        assert np.abs(P @ a - E.T @ lam).max() < 1e-8
        x = qp.solve_osqp(j, q0, v0, qg, N)
        assert x is not None and np.abs(x - a).max() < 1e-4 * max(1.0, np.abs(a).max())


def test_osqp_handles_active_limits(arm):
    """A move that needs the limits: OSQP solution reaches the goal on time inside the limits."""
    qp = TrajectoryQP(arm.h, arm.q_min, arm.q_max, osqp_max_iter=4000)
    j, q0, v0 = 0, 0.0, 0.0
    N = 60
    qg = 0.75 * qp.a_max[j] * (N * arm.h) ** 2 / 4          # 75 % of the bang-bang reach
    assert not qp.within_limits(j, qp.closed_form(q0, v0, qg, N), q0, v0)
    a = qp.solve_osqp(j, q0, v0, qg, N)
    assert a is not None and qp.within_limits(j, a, q0, v0, tol=1e-4)
    v = v0 + arm.h * np.cumsum(a)
    assert abs(q0 + arm.h * v.sum() - qg) < 1e-5 and abs(v[-1]) < 1e-4


def test_plan_respects_limits_and_reaches_goal(arm):
    qp = TrajectoryQP(arm.h, arm.q_min, arm.q_max)
    rng = np.random.default_rng(2)
    q_mid = 0.5 * (arm.q_min + arm.q_max)
    n_late = 0
    for _ in range(25):
        q0 = q_mid + rng.uniform(-0.3, 0.3, 7)
        v0 = rng.uniform(-0.5, 0.5, 7)
        qg = q0 + rng.uniform(-0.6, 0.6, 7)
        N = int(rng.integers(20, 80))
        plan = qp.solve(q0, v0, qg, N)
        assert np.abs(plan.a).max(axis=0).__le__(qp.a_max + 1e-4).all()
        assert np.abs(plan.v[1:]).max(axis=0).__le__(qp.v_max + 1e-4).all()
        assert (plan.q[1:] >= qp.q_min - 1e-4).all() and (plan.q[1:] <= qp.q_max + 1e-4).all()
        # the plan obeys the simulator's integrator exactly
        assert np.allclose(np.diff(plan.v, axis=0), arm.h * plan.a, atol=1e-12)
        assert np.allclose(np.diff(plan.q, axis=0), arm.h * plan.v[1:], atol=1e-12)
        # every joint ends at rest at the goal; on-time joints are there at the interception
        assert np.abs(plan.q[-1] - plan.q_goal).max() < 1e-4 and np.abs(plan.v[-1]).max() < 1e-3
        ot = plan.on_time
        assert np.abs(plan.q_at_intercept[ot] - plan.q_goal[ot]).max(initial=0.0) < 1e-4
        n_late += (~ot).any()
    assert n_late > 0, "test should exercise the late (intercept) branch"


def test_plan_is_tracked_exactly_by_computed_torque(arm):
    m, ix = arm.m, arm.ix
    qp = TrajectoryQP(arm.h, arm.q_min, arm.q_max)
    d = mujoco.MjData(m)
    q0 = np.array([-0.34, -0.49, 0.32, -2.29, 0.44, 2.86, 2.03])
    d.qpos[ix.arm_qadr] = q0
    d.qpos[ix.finger_qadr] = 0.04
    d.ctrl[ix.grip_act] = 255
    d.qpos[ix.obj_qadr:ix.obj_qadr + 3] = PARK_POS
    plan = qp.solve(q0, np.zeros(7), q0 + np.array([0.3, -0.2, 0.2, 0.3, -0.4, -0.3, 0.5]), 60)
    worst = 0.0
    for k in range(plan.a.shape[0] + 5):
        mujoco.mj_step1(m, d)
        qr, vr, ar = plan.ref(k)
        tau, sat = arm.torque(d.M, d.qfrc_bias, d.qfrc_passive, d.qpos[ix.arm_qadr], d.qvel[ix.arm_dadr], qr, vr, ar)
        assert not sat.any()
        worst = max(worst, np.abs(qr - d.qpos[ix.arm_qadr]).max())
        d.ctrl[:7] = tau
        mujoco.mj_step2(m, d)
    assert worst < 1e-5
