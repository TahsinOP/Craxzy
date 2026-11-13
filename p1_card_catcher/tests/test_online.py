"""Online (batched, streaming) components agree with the offline pipeline; trigger logic; env smoke test."""
import numpy as np
import torch

from card_catcher import config
from card_catcher.control.triggers import Drift, Periodic, Variance
from card_catcher.envs.catch import DEFAULT_PARAMS, CatchEnv, run_fixed_rate
from card_catcher.envs.flight import generate
from card_catcher.estim.ekf import EKFParams
from card_catcher.estim.predictor import HybridPredictor, OraclePredictor, PhysicsPredictor
from card_catcher.estim.runner import forecast_errors, run_filter
from card_catcher.learn import data as ld
from card_catcher.learn.models import SeqPredictor

CFG = config.load()
PLANE = config.Plane.from_cfg(CFG)
H = CFG["sim"]["timestep"]
PRM = EKFParams(sigma_a=8.0, kd0=1.46, kd0_std=0.07, sigma_meas=CFG["measurement"]["sigma"])


def test_hybrid_predictor_streaming_equals_offline():
    """Stepping the HybridPredictor frame by frame reproduces the offline M2 evaluation pipeline."""
    torch.manual_seed(0)
    model = SeqPredictor(ld.N_FEAT_HYBRID, 112, 32, "residual")       # untrained weights are enough for this check
    model.scale[:] = 5.0
    D = generate("card", 6, CFG, seed=11)
    sd = ld.build(D, PRM, CFG["measurement"], np.random.default_rng(0), "residual")
    with torch.no_grad():
        Hs, _ = model.encode(torch.from_numpy(sd.feats))
        mu, sig = model.decode_physical(Hs)
    mu, sig = mu.numpy(), sig.numpy()
    prm_h = EKFParams(sigma_a=PRM.sigma_a, kd0=PRM.kd0, kd0_std=0.0, sigma_meas=PRM.sigma_meas, q_kd=0.0)
    res = run_filter(sd.z, sd.valid, D.dt, prm_h, r=mu[:, :-1, 0], r_var=sig[:, :-1, 0] ** 2)

    pred = HybridPredictor(6, H, PRM, PLANE, model, phi=0.9, sigma_floor=0.0)
    for k in range(40):
        pred.step(sd.z[:, k], sd.valid[:, k])
        assert np.abs(pred.ekf.x - res["X"][:, k]).max() < 1e-4, k
    assert np.abs(pred.mu - mu[:, 39]).max() < 1e-3 and np.abs(pred.sig - sig[:, 39]).max() < 1e-3
    fc = pred.forecast()
    from card_catcher.estim.ekf import forecast

    pad = lambda a: np.concatenate([a, np.repeat(a[:, -1:], 200 - a.shape[1], axis=1)], axis=1)
    ref = forecast(res["X"][:, 39], res["P"][:, 39], H, PLANE, 200, pad(sig[:, 39]), pad(mu[:, 39]), 0.9)
    ok = ref["found"] & fc["found"]
    assert ok.any() and np.abs(fc["p"][ok] - ref["p"][ok]).max() < 1e-4


def test_physics_predictor_equals_offline():
    D = generate("ball", 5, CFG, seed=3)
    prm = EKFParams(sigma_a=0.2, kd0=0.0295, kd0_std=0.002, sigma_meas=CFG["measurement"]["sigma"])
    sd = ld.build(D, prm, CFG["measurement"], np.random.default_rng(1), "residual")
    res = run_filter(sd.z, sd.valid, D.dt, prm)
    pred = PhysicsPredictor(5, H, prm, PLANE)
    for k in range(30):
        pred.step(sd.z[:, k], sd.valid[:, k])
    assert np.abs(pred.ekf.x - res["X"][:, 29]).max() < 1e-12


def test_periodic_trigger_averages_its_rate_on_the_simulation_clock():
    trig = Periodic(50.0)
    trig.reset(1)
    fires = 0
    one, zero = np.ones(1, bool), np.zeros(1, bool)
    for k in range(126):
        obs = {"k": k, "h": H, "need_first": one if k == 0 else zero, "has_plan": one if k > 0 else zero, "active": one}
        fires += int(trig(obs)[0])
    assert fires == 50          # 1 s at 125 Hz: 50 replans, alternating 2- and 3-step gaps


def test_event_triggers_fire_on_their_statistic():
    one = np.ones(1, bool)
    base = {"active": one, "has_plan": one, "speed": np.array([3.0]), "time_drift": np.array([0.0])}
    assert not Variance(0.01)({**base, "tr_plan": np.array([1.0e-4]), "tr_now": np.array([0.5e-4])})[0]      # sqrt(0.5e-4) = 7 mm
    assert Variance(0.005)({**base, "tr_plan": np.array([1.0e-4]), "tr_now": np.array([0.5e-4])})[0]
    assert not Drift(0.01)({**base, "drift": np.array([0.008])})[0]
    assert Drift(0.01)({**base, "drift": np.array([0.008]), "time_drift": np.array([0.003])})[0]           # 3 ms * 3 m/s = 9 mm more


def test_oracle_ball_catch_smoke():
    D = generate("ball", 8, CFG, seed=50)
    env = CatchEnv("ball", CFG, 8, OraclePredictor(8, H, PLANE), DEFAULT_PARAMS["ball"], seed=0)
    out = run_fixed_rate(env, D, np.arange(8))
    assert out["caught"].mean() >= 0.75
    assert np.nanmedian(out["tcp_err_at_cross"]) < 2e-3
    assert out["sat_steps"].sum() == 0
