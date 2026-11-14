---
title: "The Oracle"
description: "Milestone 1: the planner with MuJoCo itself as the rollout model. Three targets, what was reached, what it cost, and what the height-map cost does not see."
sidebar:
  order: 2
---

:::note[Guiding question]
If the model is perfect, what is left? How close does a small greedy search get to a shape that is known to be reachable, and what does each decision cost?
:::

## Targets that are known to be reachable

Each target is produced by a **hidden demonstration**: a short scripted sequence run in the simulator from the untouched block. The planner receives the resulting height map and the tool. It never sees the actions.

| target | tool | demonstration | cap on actions | height-map RMSE of the untouched block |
|---|---|---|---|---|
| **slab** | roller | four parallel passes, 8 mm deep, 100 mm long, 30 mm apart | 6 | 6.97 mm |
| **dent** | flat press | one press, 11 mm deep, off-centre and turned 20° | 3 | 2.30 mm |
| **X** | cross stamp | one stamp, 12 mm deep, turned 45° | 3 | 4.39 mm |

Because the demonstration exists, zero error is attainable, and an error can be read on an absolute scale: the last column is the error of doing nothing.

Targets drawn by hand (a pocket with vertical walls, a slab of exactly uniform height) would not have that property. This clay springs back by a third and cannot hold a sharp edge ([The Clay](/bench/clay/)), so the best reachable error for such a target would be an unknown positive number, and every planner would be compared against a floor nobody can compute.

## The oracle

Here the rollout model of [The Planner](/planning/cem/) is MuJoCo. For each of the 96 candidate actions of a decision, the saved state is loaded into a world of `mujoco.batch`, the primitive is simulated, and the height map of the resulting shape is compared with the target. Nothing is approximated: the model *is* the environment, down to the last bit ([State, Save and Restore](/bench/state/)).

This is the most expensive planner in the project and the upper bound for the others: same search, same budget, exact predictions.

## Results

Ten seeds of the search per target; mean ± standard deviation.

| target | reached: height-map RMSE | share of the initial error left | Chamfer | EMD | actions | simulated primitives for planning | planning time |
|---|---|---|---|---|---|---|---|
| slab | 1.05 ± 0.13 mm | 15.1 ± 1.9 % | 3.10 ± 0.51 mm | 7.66 ± 2.65 mm | 5.7 ± 0.6 | 566 ± 29 | 213 ± 11 s |
| dent | 0.27 ± 0.05 mm | 11.8 ± 2.4 % | 0.29 ± 0.09 mm | 0.34 ± 0.11 mm | 2.6 ± 0.7 | 278 ± 29 | 54 ± 6 s |
| X | 0.65 ± 0.20 mm | 14.7 ± 4.6 % | 0.65 ± 0.17 mm | 0.74 ± 0.21 mm | 2.4 ± 0.8 | 269 ± 38 | 52 ± 7 s |

For scale, the untouched block against the three targets has Chamfer distances of 6.91, 1.38 and 3.26 mm and earth mover's distances of 11.57, 1.58 and 4.02 mm.

Averaged over targets, 13.9 ± 1.9 % of the initial height-map error is left, at 371 simulated primitives per episode. One decision takes 18 s with the press or the stamp and 36 s with the roller, whose primitive is almost twice as long.

<figure class="plot">
<img class="fig-light" src="/fig/m1-targets-light.svg" alt="Target height maps, the height maps reached by oracle MPC, and their signed difference, for the slab, the dent and the X">
<img class="fig-dark" src="/fig/m1-targets-dark.svg" alt="Target height maps, the height maps reached by oracle MPC, and their signed difference, for the slab, the dent and the X">
<figcaption>First seed of each target. Left and centre share the depth scale; the right column is reached minus target, red where the clay is still too high. Rows of the image run along x, away from the robot; columns along y.</figcaption>
</figure>

<figure class="plot">
<img class="fig-light" src="/fig/m1-curves-light.svg" alt="Height-map RMSE after each executed action for the three targets, one line per seed">
<img class="fig-dark" src="/fig/m1-curves-dark.svg" alt="Height-map RMSE after each executed action for the three targets, one line per seed">
<figcaption>True error after each executed action, one line per seed. The first action does most of the work in every case.</figcaption>
</figure>

### Why an exact model does not reach zero

The demonstration of the dent is a single press 11 mm deep, and the oracle finds it once in ten seeds: seed 9 pressed 11.1 mm deep, 1 mm from the demonstrated position, reached 0.19 mm and correctly stopped. The other nine first presses are 8.7 to 10.2 mm deep (one at 11.0), a few millimetres off in position and often turned. They leave 0.27 to 0.60 mm, and one or two further presses bring the result to 0.22 to 0.35 mm.

The model is exact, so this is entirely the search. With 96 evaluations over four parameters, the best sample is near the optimum and not at it. Then the greedy loop continues from a shape the demonstration never visited, where the remaining error can no longer be removed by one clean press.

The X shows a *systematic* part of the search error. All ten first stamps are too shallow: 9.3 to 11.7 mm, against 12 mm in the demonstration. The reason is the guided proposal of [The Planner](/planning/cem/#a-proposal-that-looks-at-the-target). It sets the depth of half the first population so that the tool bottom lands on the *target surface*, which would be right for a material that keeps the whole stroke. This clay gives back about a third ([The Clay](/bench/clay/)), so a dent that ends 8 mm deep needs a 12 mm stroke, and three iterations of the search only move part of the way from where the proposal started. A stamp that is too shallow cannot be fixed by stamping again in the same place without reproducing the error elsewhere, so the X ends at 0.65 mm, worse than the dent, although it is also a single action.

Scaling the proposed depth by the spring-back, a factor of about 1.5, would remove most of this bias. It was not done, because by the time it was found every planner in [parts VI](/learned-mpc/results/) and [VII](/fallback/results/) used this proposal, and they are compared with each other at a fixed search.

The slab adds a third effect, visible as the blue centre of its difference map: the block is too *low* there. The first greedy pass goes as deep as the tool is allowed, or nearly (10.9 to 14 mm, against 8 mm in the demonstration), because that removes the most error in one action, and clay that has been pressed too far cannot be raised again. A planner that looked two actions ahead would not make that trade. [Planning with Learned Models](/learned-mpc/results/#the-slab-a-wrong-model-that-helps) shows a model that avoids it by being wrong.

A larger population would close most of the random part of this gap at a proportional cost. The budget was kept at 96 because the point of the oracle here is to be a fixed reference for the learned planners, which are given the same 96.

### What the height map does not see

For the dent and the X, Chamfer and EMD tell the same story as the height map. For the slab they do not. The height-map error falls to 15 % of its initial value, but the earth mover's distance only falls from 11.6 mm to 7.7 mm, with a spread across seeds of 2.7 mm (from 2.4 to 10.8 mm).

The slab is the target where material moves sideways by more than a centimetre ([The Clay](/bench/clay/)). The demonstration rolls four parallel passes in one direction. The planner rolls five or six passes in assorted directions: its first pass is nearly always a deep, long roll across the middle of the block, at a different yaw in every seed. The top surface ends up at nearly the right heights, with its material in different places and its side walls bulged differently. The height map cannot tell, and the planner was told to minimise the height map.

This is a property of the cost, not of the oracle, and it is inherited by every planner in the following parts. It is one of the limits listed in [Limits](/evaluation/limits/).

<figure class="sim">
<video src="/video/oracle.mp4" loop muted playsinline controls preload="metadata"></video>
<figcaption>The first-seed episodes of the table, replayed in real time with the moves between primitives included. The pauses of 18 to 36 s in which the planner thinks are not shown.</figcaption>
</figure>

:::tip[Where in the code]
`envs/targets.py`: `DEMOS`, `make_target`, `shape_errors`. `control/mpc.py`: `OracleEval`, `run_episode`. `scripts/m1_oracle.py` → `results/m1_oracle.json`, `results/M1.md`.
:::
