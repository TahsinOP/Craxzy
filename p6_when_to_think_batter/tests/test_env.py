"""The batting cage: sensor delay, speculative contacts, landing extrapolation, triggers, PPO plumbing."""
import mujoco
import numpy as np
import pytest
import torch

from batter import config
from batter.control.arm import NOMINAL_STRIKE, Arm
from batter.control.triggers import Innovation, Periodic, Variance
from batter.envs.cage import BatEnv, EnvParams, run_episode
from batter.learn import ppo
from batter.models.pitch import generate
from batter.models.scene import build_cage_model


@pytest.fixture(scope="module")
def cfg():
    return config.load()


def test_camera_sensor_is_delayed(cfg):
    """MuJoCo 4.0 sensor delay: the camera shows the ball where it was 30 ms (3 steps) ago."""
    env = BatEnv(cfg, 2, EnvParams(swing=False), nthread=1)
    env.reset(generate(env.fm, cfg, 2, 0))
    hist = [env.ball_state()[:, :3].copy()]
    for _ in range(12):
        env.step()
        hist.append(env.ball_state()[:, :3].copy())
        if env.k >= env.nd:
            assert np.allclose(env.sens[:, env._cam], hist[env.k - env.nd], atol=1e-12)
    assert env.nd == 3


def _static(cfg, speculative, n=24):
    arm = Arm(build_cage_model(cfg))
    sweet, R = arm.fk(NOMINAL_STRIKE)
    env = BatEnv(cfg, n, EnvParams(swing=False, q_start=tuple(NOMINAL_STRIKE)), nthread=2, sim_override={"speculative": speculative})
    aim = np.tile(sweet + (cfg["ball"]["radius"] + cfg["bat"]["thickness"] / 2) * R[:, 0], (n, 1))
    out = run_episode(env, generate(env.fm, cfg, n, 1, aim=aim), Periodic(0.0))
    return (out["hit"] & (out["v_post"][:, 0] > 0.5)).mean()


def test_speculative_contacts_stop_the_ball_tunnelling(cfg):
    assert _static(cfg, True) == 1.0          # every pitch bounces off the 1.2 cm bat
    assert _static(cfg, False) < 0.8          # without the flag a good share passes through it


def test_batter_hits_and_landing_extrapolation_is_exact(cfg):
    """100 Hz batter with the true ball state: every pitch is hit and lands near the target.  The landing
    point is extrapolated with the flight model from the ball's state a few steps after the impact; here
    the simulation is continued and must follow that extrapolation exactly for the whole flight, down to
    5 cm above the floor.  (Lower than that the floor's speculative contact starts to brake the ball a
    step before it touches, and it then bounces and rolls: a landing point read off the continued
    simulation is not the point where the free flight meets the ground, and differs by up to 15 cm.)"""
    n = 6
    env = BatEnv(cfg, n, EnvParams(oracle=True), nthread=2)
    P = generate(env.fm, cfg, n, 5)
    out = run_episode(env, P, Periodic(1000.0))
    assert out["hit"].all()
    assert np.median(out["dist"]) < 0.3
    assert (out["n_plans"] > 30).all()
    pred = env.fm.rollout(env.x_after, 160)                      # (161, n, 9) from each world's finishing step
    checked = 0
    for _ in range(160):
        env.step()
        x = env.ball_state()
        m = env.k - env.finish_k
        for i in range(n):
            if 0 < m[i] <= 160 and x[i, 2] > env.fm.radius + 0.05 and pred[: m[i] + 1, i, 2].min() > env.fm.radius + 0.05:
                assert np.abs(x[i, :6] - pred[m[i], i, :6]).max() < 1e-8
                checked += 1
    assert checked > 300


def test_triggers(cfg):
    n = 16
    env = BatEnv(cfg, n, nthread=2)
    P = generate(env.fm, cfg, n, 7)
    once = run_episode(env, P, Periodic(0.0))
    assert once["n_plans"].mean() < 1.3            # one call (retried only if the first forecast allowed no strike)
    ten = run_episode(env, P, Periodic(10.0))
    assert 3 <= ten["n_plans"].mean() <= 8
    assert ten["quality"].mean() > once["quality"].mean() + 0.3      # replanning is what makes it hit
    # the innovation trigger looks at data: it spends more calls when a gust hits; the variance trigger cannot
    Pg = generate(env.fm, cfg, n, 7, surprise="gust")
    inn_plain, inn_gust = run_episode(env, P, Innovation(6.5))["n_plans"].mean(), run_episode(env, Pg, Innovation(6.5))["n_plans"].mean()
    assert inn_gust > inn_plain + 0.3
    var_plain, var_gust = run_episode(env, P, Variance(0.03))["n_plans"].mean(), run_episode(env, Pg, Variance(0.03))["n_plans"].mean()
    assert abs(var_gust - var_plain) < 0.5


def test_ppo_rollout_and_update(cfg):
    torch.manual_seed(0)
    env = BatEnv(cfg, 8, nthread=2)
    policy = ppo.TriggerPolicy()
    rng = np.random.default_rng(0)
    ro = ppo.collect(env, generate(env.fm, cfg, 8, 9), policy, rng)
    assert ro.feats.shape[1] == ppo.N_OBS and len(ro.feats) == len(ro.actions) == len(ro.adv) == len(ro.ret)
    assert np.isfinite(ro.feats).all() and np.isfinite(ro.adv).all()
    before = [p.clone() for p in policy.parameters()]
    info = ppo.update(policy, torch.optim.Adam(policy.parameters(), lr=3e-4), ro, rng)
    assert np.isfinite(info["pg"]) and any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))


def test_filament_look_model_matches_the_simulated_one(cfg):
    """The Filament videos copy a world's state into a look model with materials and lighting flags;
    that is only valid if the two have the same state layout and the same geometry."""
    from batter.eval.filament import look_model
    from batter.models.scene import build_cage_model

    sim, look = build_cage_model(cfg), look_model(cfg)
    for field in ("nq", "nv", "nu", "nbody", "ngeom", "nmocap", "nsensordata"):
        assert getattr(sim, field) == getattr(look, field), field
    assert np.allclose(sim.geom_size, look.geom_size) and np.allclose(sim.body_pos, look.body_pos)
    assert look.nmat > sim.nmat and look.ntex > sim.ntex
