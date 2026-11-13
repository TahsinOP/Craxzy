---
title: "The Model-Based Hitter"
description: "Hit rate and target accuracy against pitch speed, with replanning at every step, with a perfect estimator, and with a single plan."
sidebar:
  order: 3
---

:::note[Guiding question]
How good is the batter when compute is free, and how much of that comes from replanning?
:::

## Three versions of the same planner

- **Every step:** the planner runs on the newest forecast at every 10 ms step (48 calls per pitch). This is the M1 controller and the ceiling the triggers are measured against.
- **True ball state:** the same, with the EKF replaced by the simulator's ball state. This separates what estimation costs from what the swing and the impact model cost.
- **Called once:** the planner runs on the forecast after the first five fixes and never again.

<figure class="plot">
<img class="fig-light" src="/fig/speed-light.svg" alt="Hit rate and on-target rate against pitch speed for the analytic planner variants and the world-model planner">
<img class="fig-dark" src="/fig/speed-dark.svg" alt="Hit rate and on-target rate against pitch speed for the analytic planner variants and the world-model planner">
<figcaption>3 seeds of 200 pitches at each speed (128 for the world-model planner, from Part V). Hit rate is near 100 % for every replanning variant; target accuracy is where they differ.</figcaption>
</figure>

| release speed | every step: hit / on target | true ball state: on target | called once: hit |
|---|---|---|---|
| 9 m/s | 100.0 % / 99.7 % | 100.0 % | 9.3 % |
| 10 m/s | 99.8 % / 99.7 % | 100.0 % | 10.5 % |
| 11 m/s | 100.0 % / 98.5 % | 100.0 % | 12.2 % |
| 12 m/s | 100.0 % / 96.0 % | 99.5 % | 13.5 % |
| 13 m/s | 100.0 % / 89.5 % | 96.7 % | 16.3 % |
| **all pitches** | **100.0 ± 0.0 % / 97.8 ± 0.2 %** | **99.7 ± 0.5 %** | **11.8 ± 1.8 %** |

## Reading it

- **Replanning is not a refinement here, it is what makes the batter work.** Planned once, on a forecast that is about 15 cm off, the swing meets the ball in one pitch in nine.
- **Target accuracy falls with speed.** A fast pitch leaves less time between the last accurate forecasts and the impact for the arm to act on them.
- **Estimation, not the swing, is what is lost.** With the true ball state the same planner is on target 99.7 % of the time.

## Compute

Measured alone on one core, a planner call takes 9.1 ms at the median, 13.1 ms on average and 21.5 ms at the 90th percentile (the slow calls are the ones where OSQP has to step in). The control period is 10 ms, so this Python planner is not real-time at every step. The simulation does not charge for planning time: a plan is applied in the step it was asked for. That makes the comparison of triggers a comparison of *how many* calls are needed, and Part VI also plots the measured wall-clock.

:::tip[Where in the code]
`scripts/m1_hitter.py`, `envs/cage.py`, `eval/runner.py` (`time_planner`).
:::
