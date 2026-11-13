"""Compute triggers: when is it worth running the expensive planner again?

Every trigger sees the same per-step observation from the environment and returns a
boolean per world.  The first plan of a pitch is made by the environment and counts as a
planner call for every trigger.

Notation: t_p is the time of the last plan; the plan was built on a forecast of the ball,
p_hat(t | t_p) with covariance P(t | t_p), and on the crossing point c_hat(t_p) of the
hitting plane with covariance C(t_p).

* ``Periodic(rate)``    replan every 1/rate seconds (baseline: 100 Hz = every step).
* ``Innovation(gate)``  replan when the camera contradicts the forecast the plan was built
      on.  Each fix z(t) gives nu = z - p_hat(t | t_p) with S = P(t | t_p) + R; under the
      model nu' S^-1 nu is chi-square with 3 dof (mean 3).  The trigger fires when the mean
      over the last few fixes exceeds ``gate`` (a chi-square gate).  It looks at data.
* ``Variance(delta)``   replan when  tr C(t_p) - tr C(t) > delta^2  (Trimpe & D'Andrea):
      covariances only, never the realised measurements, so it could be scheduled in
      advance -- and it cannot see a surprise.  For a consistent estimator the left side
      is the expected squared distance between the crossing point the plan aims at and the
      one we would aim at now.
* ``Drift(delta)``      replan when the realised crossing-point estimate has moved by more
      than delta (send-on-delta; a timing change is converted to a distance).
* ``Learned(policy)``   PPO policy on (innovation, covariance, time to impact, ...) with
      reward = hit quality - lam * planner calls; see batter.learn.ppo.
"""
from __future__ import annotations

import numpy as np


class Periodic:
    def __init__(self, rate_hz: float):
        self.rate = rate_hz
        self.period = np.inf if rate_hz <= 0 else 1.0 / rate_hz

    def reset(self, B: int) -> None:
        self.next_t = np.full(B, np.inf)

    def __call__(self, obs: dict) -> np.ndarray:
        t = obs["k"] * obs["h"]
        first = obs["need_first"]
        self.next_t = np.where(first, t + self.period, self.next_t)
        fire = obs["active"] & obs["has_plan"] & (t >= self.next_t - 1e-9)
        self.next_t = np.where(fire, np.maximum(self.next_t + self.period, t + 1e-9), self.next_t)   # fixed grid
        return fire


class Innovation:
    def __init__(self, gate: float, min_fixes: int = 2):
        self.gate, self.min_fixes = gate, min_fixes

    def reset(self, B: int) -> None:
        pass

    def __call__(self, obs: dict) -> np.ndarray:
        return obs["active"] & obs["has_plan"] & (obs["chi2_n"] >= self.min_fixes) & (obs["chi2"] > self.gate)


class Variance:
    def __init__(self, delta: float):
        self.delta = delta

    def reset(self, B: int) -> None:
        pass

    def __call__(self, obs: dict) -> np.ndarray:
        return obs["active"] & obs["has_plan"] & (obs["tr_plan"] - obs["tr_now"] > self.delta**2)


class Drift:
    def __init__(self, delta: float):
        self.delta = delta

    def reset(self, B: int) -> None:
        pass

    def __call__(self, obs: dict) -> np.ndarray:
        d2 = obs["drift"] ** 2 + (obs["speed"] * obs["time_drift"]) ** 2
        return obs["active"] & obs["has_plan"] & (d2 > self.delta**2)


class Learned:
    def __init__(self, policy, lam: float, rng: np.random.Generator, greedy: bool = False):
        self.policy, self.lam, self.rng, self.greedy = policy, lam, rng, greedy

    def reset(self, B: int) -> None:
        self.lams = np.full(B, self.lam)

    def __call__(self, obs: dict) -> np.ndarray:
        from batter.learn.ppo import act

        a, _, _ = act(self.policy, obs, self.lams, self.rng, self.greedy)
        return obs["active"] & obs["has_plan"] & a
