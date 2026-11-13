---
title: "Quality Against Compute"
description: "The Pareto curves of hit quality against planner calls, and why the classical event triggers lose to a fixed rate."
sidebar:
  order: 3
---

:::note[Guiding question]
For a given number of planner calls, which trigger hits best, and why?
:::

<figure class="plot">
<img class="fig-light" src="/fig/pareto-light.svg" alt="Hit quality against planner calls per pitch and against planning time, for every trigger">
<img class="fig-dark" src="/fig/pareto-dark.svg" alt="Hit quality against planner calls per pitch and against planning time, for every trigger">
<figcaption>3 seeds of 200 pitches per point. Left: against planner calls. Right: against the planning time actually measured (inside the loaded process pool, so about twice the single-core figure); the ordering is the same. Orange is the pure-learning baseline: the world-model planner at fixed rates.</figcaption>
</figure>

## Best quality within a budget

Hit quality of the best swept setting using at most the given number of planner calls per pitch (calls actually used in brackets):

| trigger | ≤ 2 | ≤ 3 | ≤ 4.5 | ≤ 7 | ≤ 13 |
|---|---|---|---|---|---|
| fixed rate | 0.13 (1.9) | 0.58 (2.9) | 0.91 (4.4) | 0.95 (6.5) | 0.97 (12.4) |
| innovation | 0.24 (1.9) | 0.24 (1.9) | 0.45 (4.0) | 0.45 (4.0) | 0.74 (10.9) |
| variance | – | 0.29 (3.0) | 0.29 (3.0) | 0.64 (6.6) | 0.88 (12.5) |
| drift | – | – | 0.44 (3.4) | 0.76 (6.5) | 0.95 (11.4) |
| **learned** | – | **0.81 (2.9)** | **0.94 (4.0)** | **0.97 (6.6)** | 0.97 (10.4) |
| world-model planner | 0.38 (1.8) | 0.57 (2.9) | 0.57 (2.9) | 0.69 (5.3) | 0.79 (12.4) |

Replanning at every step gives 0.981 with 48 calls. To stay within 0.02 of that, a fixed rate needs 12.4 calls (25 Hz), the learned trigger 6.6.

## Why the classical triggers lose

<figure class="plot">
<img class="fig-light" src="/fig/when-light.svg" alt="Distribution of replanning times over the flight for each trigger at about four calls per pitch">
<img class="fig-dark" src="/fig/when-dark.svg" alt="Distribution of replanning times over the flight for each trigger at about four calls per pitch">
<figcaption>When in the flight each trigger calls the planner, at settings that cost about 4 calls per pitch. 300 pitches.</figcaption>
</figure>

| trigger, at ~4 calls | hit quality | median time to the plate at a replan |
|---|---|---|
| variance | 0.29 | 499 ms |
| drift | 0.44 | 487 ms |
| innovation | 0.45 | 393 ms |
| learned | 0.94 | 302 ms |
| fixed rate | 0.91 | 254 ms |

The [forecast](/estimation/ekf/) improves by centimetres per frame early in the flight and by millimetres per frame late. Variance and drift thresholds are in centimetres, so they fire repeatedly early, while the estimate is moving fast, and stop firing in the last 300 ms, when the estimate barely moves but those millimetres decide whether the blade meets the ball. A threshold small enough to keep firing late fires constantly early and costs as much as a high fixed rate.

The innovation trigger has a different failure: on an ordinary pitch the camera never *contradicts* the forecast, it only sharpens it. Large innovations come from noise, early, when the filter's covariance is still wide; low gates fire on noise all the time.

The learned trigger saw the outcome of each schedule and learned the useful one: few calls early, more as the plate approaches, and a call in the last 100 ms. It is the only trigger that beats the fixed rate at small budgets.

The **world-model planner** (pure learning) is flatter: cheaper at the lowest budgets because its swings degrade more gracefully, but it never gets near the analytic planner's quality.

:::tip[Where in the code]
`scripts/m3_triggers.py`, `eval/sweeps.py`; every setting in `results/M3.md`.
:::
