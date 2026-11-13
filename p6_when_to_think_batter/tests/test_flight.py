"""The closed-form flight model against the simulator."""
import mujoco
import numpy as np

from batter import config
from batter.estim.flight import FlightModel
from batter.models.scene import build_cage_model, build_flight_model


def _simulate(m, x0, n):
    d = mujoco.MjData(m)
    adr, qadr = int(m.jnt_dofadr[m.joint("ball").id]), int(m.jnt_qposadr[m.joint("ball").id])
    d.qpos[qadr:qadr + 3] = x0[:3]
    d.qvel[adr:adr + 3] = x0[3:6]
    d.qvel[adr + 3:adr + 6] = x0[6:9]          # identity orientation: local = world
    out = [x0.copy()]
    for _ in range(n):
        mujoco.mj_step(m, d)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[qadr + 3:qadr + 7])
        out.append(np.concatenate([d.qpos[qadr:qadr + 3], d.qvel[adr:adr + 3], R.reshape(3, 3) @ d.qvel[adr + 3:adr + 6]]))
    return np.array(out)


X0 = np.array([5.0, 0.05, 1.3, -10.5, 0.3, 1.2, 40.0, 180.0, -60.0])


def test_step_matches_mujoco_flight_model():
    cfg = config.load()
    m = build_flight_model(cfg["sim"], cfg["ball"])
    fm = FlightModel.identify(m)
    sim = _simulate(m, X0, 45)
    mod = fm.rollout(X0, 45)
    assert np.abs(sim[:, :3] - mod[:, :3]).max() < 1e-9
    assert np.abs(sim[:, 3:6] - mod[:, 3:6]).max() < 1e-8
    assert np.abs(sim[:, 6:9] - mod[:, 6:9]).max() < 1e-7


def test_same_flight_in_the_cage_model():
    """The ball flies identically in the full scene (arm present) until it touches something."""
    cfg = config.load()
    m = build_cage_model(cfg)
    fm = FlightModel.identify(m)
    sim = _simulate(m, X0, 30)
    assert np.abs(sim[:, :3] - fm.rollout(X0, 30)[:, :3]).max() < 1e-9


def test_spin_matters():
    """Lift from 200 rad/s of backspin moves the crossing point by well over a bat thickness."""
    cfg = config.load()
    fm = FlightModel.identify(build_flight_model(cfg["sim"], cfg["ball"]))
    a = fm.rollout(np.array([5.0, 0, 1.3, -10.0, 0, 1.0, 0, 0, 0]), 40)[-1]
    b = fm.rollout(np.array([5.0, 0, 1.3, -10.0, 0, 1.0, 0, 200.0, 0]), 40)[-1]
    assert b[2] - a[2] > 0.10


def test_jacobian_is_the_derivative_of_step():
    cfg = config.load()
    fm = FlightModel.identify(build_flight_model(cfg["sim"], cfg["ball"]))
    F = fm.jacobian(X0)
    num = np.zeros((9, 9))
    for i in range(9):
        e = np.zeros(9)
        e[i] = 1e-6
        num[:, i] = (fm.step(X0 + e) - fm.step(X0 - e)) / 2e-6
    assert np.abs(F - num).max() < 1e-7


def test_landing_matches_simulation():
    cfg = config.load()
    m = build_flight_model(cfg["sim"], cfg["ball"])
    fm = FlightModel.identify(m)
    x0 = np.array([0.6, 0.0, 0.9, 7.0, 1.0, 4.0, 0.0, 120.0, 30.0])
    sim = _simulate(m, x0, 200)
    k = int(np.argmax(sim[:, 2] <= fm.radius))
    xy, t = fm.landing(x0)
    assert (k - 1) * fm.h <= t <= k * fm.h
    assert np.linalg.norm(xy - sim[k, :2]) < np.linalg.norm(sim[k, :2] - sim[k - 1, :2]) + 1e-9
