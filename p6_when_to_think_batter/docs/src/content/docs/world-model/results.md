---
title: "Learned Against Analytic"
description: "The world-model planner hits but does not aim; it is no faster; and it needs all the data it was given."
sidebar:
  order: 2
---

:::note[Guiding question]
What does the learned planner get right, and where does it fall short?
:::

## Closed loop

Both planners called at every step, 3 seeds (a separately trained model per seed), 128 pitches each. Each cell: hit rate / on target / median distance from the target centre.

| release speed | analytic (classical) | world model + MPPI (pure learning) | world-model prior alone |
|---|---|---|---|
| 9 m/s | 100.0 % / 100.0 % / 0.07 m | 96.1 % / 45.1 % / 0.82 m | 95.3 % / 48.4 % / 0.73 m |
| 11 m/s | 100.0 % / 100.0 % / 0.14 m | 95.8 % / 63.0 % / 0.54 m | 94.0 % / 46.4 % / 0.77 m |
| 13 m/s | 100.0 % / 91.4 % / 0.29 m | 94.8 % / 36.2 % / 0.99 m | 96.9 % / 20.8 % / 1.40 m |
| **all pitches** | **99.7 % / 96.9 % / quality 0.98** | **96.9 % / 54.9 % / quality 0.75** | **95.3 % / 45.8 % / quality 0.71** |

The learned planner **connects with almost every pitch but aims worse**: its median miss distance is several times the analytic planner's. Meeting the ball needs the 9 × 38 cm blade at the ball; landing on a 0.75 m disc five metres out needs the face angle and the bat speed right to a few per cent. On held-out data the outcome model predicts hit or miss correctly in 91.8 % of cases but the landing point only to 0.96 m (median), about the size of the target. MPPI on the model helps over its prior alone (0.75 against 0.71).

## Planning time

Measured alone on one core:

<figure class="plot">
<img class="fig-light" src="/fig/wm-light.svg" alt="Planning time per call for the three planners, and the learned planner's rates against the amount of data">
<img class="fig-dark" src="/fig/wm-dark.svg" alt="Planning time per call for the three planners, and the learned planner's rates against the amount of data">
<figcaption>Left: median time per call, with a line to the 90th percentile. Right: the learned planner trained on 1/16, 1/4 and all of the batting data (one model each, evaluated over 3 seeds).</figcaption>
</figure>

| planner | median | mean | 90th percentile |
|---|---|---|---|
| analytic | 9.3 ms | 13.3 ms | 21.9 ms |
| world model + MPPI | 8.7 ms | 12.1 ms | 14.7 ms |
| world-model prior only | 1.2 ms | 1.7 ms | 2.7 ms |

The learned planner is no faster per call at the median, but its tail is shorter (the analytic planner's slow calls are OSQP solves). The prior alone is eight times faster than either, at a lower quality.

## Data

With 3,000 pitches of batting data the learned planner hits 65 % and lands 15 % on target; with 12,000, 75 % and 21 %; with 48,000, 99 % and 63 %. It was still improving at the largest budget this project could afford on a CPU.

:::tip[Where in the code]
`scripts/m2_worldmodel.py`; full tables in `results/M2.md`.
:::
