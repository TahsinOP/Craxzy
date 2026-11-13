---
title: "Five Ways to Decide"
description: "Fixed rate, innovation, variance, drift and learned triggers: what each one sees and when it fires."
sidebar:
  order: 1
---

:::note[Guiding question]
The arm is executing a cached swing. What should make it think again?
:::

Every trigger sees the same observation at every step and returns yes or no. The first plan of a pitch (after five camera fixes) is made for every trigger and counts as a call. Notation: $t_p$ is when the cached plan was made; it was built on a forecast $\hat p(t \mid t_p)$ of the ball with covariance $P(t \mid t_p)$, and on a crossing point $\hat c(t_p)$ with in-plane covariance $C(t_p)$.

## Fixed rate

Replan every $1/f$ seconds, on a fixed grid. The rate is swept from every step (100 Hz) down to "plan once". This is the classical baseline, and the one to beat.

## Innovation: does the camera contradict the plan?

Each new fix $z(t)$ is compared with what the plan's forecast predicted for that instant:

$$
\nu = z(t) - \hat p(t \mid t_p), \qquad S = P(t \mid t_p) + R(t), \qquad \chi^2 = \nu^\top S^{-1} \nu .
$$

Under the model, $\chi^2$ follows a χ² distribution with 3 degrees of freedom (mean 3). The trigger fires when its mean over the last four fixes exceeds a gate. It looks at data: the χ²-gate idea from the brief.

## Variance: has the uncertainty fallen enough? (Trimpe & D'Andrea)

$$
\operatorname{tr} C(t_p) - \operatorname{tr} C(t) > \delta^2 .
$$

For a consistent estimator the left side is the expected squared distance between the crossing point the plan aims at and the one we would aim at now. It uses covariances only, never the measurements, so its schedule could be computed in advance, and it cannot notice a surprise.

## Drift: has the aim point moved? (send-on-delta)

Fire when the realised crossing-point estimate has moved since the plan by more than $\delta$, counting a change in arrival time as a distance (speed × time). Added next to the required triggers as the realised counterpart of the variance trigger.

## Learned

A small policy (two layers of 64) trained with PPO. It observes the time to the plate, the time since the last plan, the innovation statistic and how many fixes it averages, how far the crossing point has drifted, the crossing-point uncertainty now and at plan time, the variance trigger's statistic, and $\log \lambda$. Its reward is the brief's:

$$
R = \text{hit quality} - \lambda \cdot \text{planner calls}.
$$

See [The learned trigger](/when-to-think/learned-trigger/).

:::tip[Where in the code]
`control/triggers.py`; the statistics are computed in `envs/cage.py` (`_sense`, `_obs`).
:::
