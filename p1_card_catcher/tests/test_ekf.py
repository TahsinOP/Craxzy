"""EKF: Jacobian, exactness of the discrete model, filter consistency, forecast covariance."""
import mujoco
import numpy as np

from card_catcher import config
from card_catcher.envs.flight import generate
from card_catcher.estim import physics
from card_catcher.estim.ekf import EKFParams, _jacobian, _step_mean, forecast
from card_catcher.estim.runner import forecast_errors, run_filter, simulate_measurements
from card_catcher.models.objects import OBJECTS
from card_catcher.models.scene import build_flight_model

CFG = config.load()
PLANE = config.Plane.from_cfg(CFG)
H = CFG["sim"]["timestep"]


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=(5, 3)), rng.normal(0, 3, (5, 3)), rng.uniform(0.02, 1.5, (5, 1))], axis=1)
    F = _jacobian(x[:, 3:6], x[:, 6], H)
    eps = 1e-6
    for i in range(7):
        dx = np.zeros(7)
        dx[i] = eps
        num = (_step_mean(x + dx, H, None) - _step_mean(x - dx, H, None)) / (2 * eps)
        assert np.abs(num - F[:, :, i]).max() < 1e-7


def _sim_ball(sim, nstep=100):
    m = build_flight_model("ball", sim)
    d = mujoco.MjData(m)
    p0, v0 = np.array([2.8, 0.1, 1.2]), np.array([-3.5, 0.2, 3.0])
    d.qpos[:3], d.qvel[:3] = p0, v0
    P = [p0.copy()]
    for _ in range(nstep):
        mujoco.mj_step(m, d)
        P.append(d.qpos[:3].copy())
    return p0, v0, np.array(P)


def test_discrete_model_matches_mujoco_for_a_nonspinning_sphere():
    """The physics predictor *is* the simulator for a drag-only sphere."""
    kd = OBJECTS["ball"].kd_axis(np.array([1.0, 0, 0]), CFG["sim"]["density"])
    # inviscid air: only quadratic drag acts -> agreement to round-off over 0.8 s
    p0, v0, P = _sim_ball(dict(CFG["sim"], viscosity=0.0))
    Pn, _ = physics.rollout_nominal(p0, v0, kd, H, 100)
    assert np.abs(Pn - P).max() < 1e-9
    # real air: the remainder is the Stokes term (sub-millimetre)
    p0, v0, P = _sim_ball(CFG["sim"])
    Pn, _ = physics.rollout_nominal(p0, v0, kd, H, 100)
    assert np.abs(Pn - P).max() < 5e-4


def test_model_accel_tends_to_continuous_drag():
    v = np.array([[-3.0, 0.5, 2.0]])
    assert np.abs(physics.model_accel(v, 0.7, 1e-9) - (physics.GRAVITY + physics.drag_accel(v, np.array([0.7])))).max() < 1e-6


def test_residual_reproduces_trajectory_exactly():
    D = generate("card", 4, CFG, seed=5)
    kd = 1.0
    r = physics.residual_accel(D.v, kd, D.dt)
    P, V = physics.rollout_nominal(D.p[:, 0], D.v[:, 0], kd, D.dt, D.p.shape[1] - 1, r)
    assert np.abs(P - D.p).max() < 1e-9 and np.abs(V - D.v).max() < 1e-9


def _ball_setup(n=300, seed=7):
    D = generate("ball", n, CFG, seed=seed)
    kd = OBJECTS["ball"].kd_axis(np.array([1.0, 0, 0]), CFG["sim"]["density"])
    prm = EKFParams(sigma_a=0.1, kd0=kd, kd0_std=0.3 * kd, sigma_meas=CFG["measurement"]["sigma"])
    z, valid = simulate_measurements(D.p, CFG["measurement"], np.random.default_rng(seed))
    return D, prm, z, valid, run_filter(z, valid, D.dt, prm)


def test_filter_is_consistent_on_the_ball():
    """Normalised estimation error squared of (p, v) is chi^2_6-like: mean close to 6."""
    D, prm, z, valid, res = _ball_setup()
    k = 50
    e = np.concatenate([res["X"][:, k, :3] - D.p[:, k], res["X"][:, k, 3:6] - D.v[:, k]], axis=1)
    Pk = res["P"][:, k, :6, :6]
    nees = np.einsum("bi,bi->b", e, np.linalg.solve(Pk, e[:, :, None])[:, :, 0])
    assert 4.0 < nees.mean() < 9.0, nees.mean()


def test_forecast_covariance_matches_monte_carlo():
    """Catch-point covariance from the linearised forecast vs sampling the state distribution."""
    rng = np.random.default_rng(0)
    x = np.array([[2.0, 0.1, 1.3, -3.0, -0.1, 2.0, 0.05]])
    Lc = np.diag([0.004, 0.004, 0.004, 0.05, 0.05, 0.05, 0.005])
    P = (Lc @ Lc.T)[None]
    sig = 0.5
    fc = forecast(x, P, H, PLANE, 200, sig)
    assert fc["found"][0]
    n = 4000
    xs = x + rng.normal(size=(n, 7)) @ Lc.T
    nstep = int(fc["k"][0]) + 30
    r = rng.normal(0, sig, (n, nstep, 3))
    Pm, Vm = physics.rollout_nominal(xs[:, :3], xs[:, 3:6], xs[:, 6], H, nstep, r)
    c = physics.plane_crossing(Pm, PLANE.c0, PLANE.n, H)
    assert c["found"].all()
    uw = PLANE.coords(c["p"])
    emp = np.cov(uw.T)
    assert np.abs(emp - fc["cov_uw"][0]).max() < 0.12 * np.abs(fc["cov_uw"][0]).max()
    assert abs(c["t"].var() - fc["var_t"][0]) < 0.12 * fc["var_t"][0]
    assert np.linalg.norm(uw.mean(0) - PLANE.coords(fc["p"][0])) < 2e-3


def test_forecast_covariance_with_correlated_noise_matches_monte_carlo():
    rng = np.random.default_rng(1)
    x = np.array([[1.5, 0.0, 1.4, -2.5, 0.0, 0.5, 1.0]])
    P = np.zeros((1, 7, 7))
    sig, phi = 3.0, 0.9
    fc = forecast(x, P, H, PLANE, 200, sig, phi=phi)
    assert fc["found"][0]
    n, nstep = 4000, int(fc["k"][0]) + 40
    eta = np.zeros((n, nstep, 3))
    eta[:, 0] = rng.normal(size=(n, 3))
    for j in range(1, nstep):
        eta[:, j] = phi * eta[:, j - 1] + np.sqrt(1 - phi**2) * rng.normal(size=(n, 3))
    Pm, _ = physics.rollout_nominal(np.repeat(x[:, :3], n, 0), np.repeat(x[:, 3:6], n, 0), x[0, 6], H, nstep, sig * eta)
    c = physics.plane_crossing(Pm, PLANE.c0, PLANE.n, H)
    emp = np.cov(PLANE.coords(c["p"][c["found"]]).T)
    assert c["found"].mean() > 0.99
    assert np.abs(emp - fc["cov_uw"][0]).max() < 0.15 * np.abs(fc["cov_uw"][0]).max()


def test_ball_meets_m1_acceptance():
    D, prm, z, valid, res = _ball_setup()
    fe = forecast_errors(res, D.cross_k, D.cross_t, D.cross_p, D.dt, PLANE, prm.sigma_a, valid=valid)
    sel = np.abs(fe["tau"] - 0.2) < 0.004
    assert sel.sum() > 100
    assert np.sqrt(np.mean(fe["err"][sel] ** 2)) < 0.01
