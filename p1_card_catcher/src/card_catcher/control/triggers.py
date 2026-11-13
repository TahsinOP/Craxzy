"""Replanning triggers (M4): when is it worth calling the planner again?

Every trigger sees the same per-step observation from the environment and returns
a boolean per world.  The first plan of a throw is always made by the environment
(as soon as a forecast exists) and counts as a planner call for every trigger.

Notation: c_hat(t) is the current catch-point estimate with in-plane covariance
P(t); t_p is the time of the last plan, which was built for c_hat(t_p).

* ``Periodic(rate)``      replan every 1/rate seconds (baseline: 50 Hz).
* ``Variance(delta)``     replan when  tr P(t_p) - tr P(t) > delta^2.
      For a (near-)optimal estimator the estimate is a martingale: the update
      c_hat(t) - c_hat(t_p) is uncorrelated with c_hat(t_p), so
      E|c_hat(t) - c_hat(t_p)|^2 = tr P(t_p) - tr P(t).  The left side is the mean
      squared distance between where the plan aims and where we would aim now;
      the trigger fires when its *expected* value exceeds delta^2.  Like the
      variance-based triggers of Trimpe & D'Andrea (2014) it uses covariances
      only, never the realised estimate, so it could be scheduled ahead of time.
* ``Drift(delta)``        replan when the realised displacement exceeds delta:
      |c_hat(t) - c_hat(t_p)|^2 + (speed * |t_c(t) - t_c(t_p)|)^2 > delta^2
      (send-on-delta; the second term converts a timing change into a distance).
* ``Learned(policy, lam)`` a small stochastic policy trained with PPO on
      reward = catch - lam * (number of replans); see card_catcher.learn.ppo.
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
        t = obs["k"] * self._h(obs)
        first = obs["need_first"]
        self.next_t = np.where(first, t + self.period, self.next_t)
        fire = obs["active"] & obs["has_plan"] & (t >= self.next_t - 1e-9)
        # keep the schedule on a fixed grid (50 Hz on a 125 Hz clock alternates 2- and 3-step gaps)
        self.next_t = np.where(fire, np.maximum(self.next_t + self.period, t + 1e-9), self.next_t)
        return fire

    @staticmethod
    def _h(obs: dict) -> float:
        return obs["h"]


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
        from card_catcher.learn.ppo import act

        a, _, _ = act(self.policy, obs, self.lams, self.rng, self.greedy)
        return obs["active"] & obs["has_plan"] & a
