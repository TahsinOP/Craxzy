---
title: "The World Model"
description: "A small latent model in the style of TD-MPC2, trained on batting data, and MPPI over six swing parameters in its latent space."
sidebar:
  order: 1
---

:::note[Guiding question]
Could a network learn, from batting data alone, what the filter, the flight model and the impact model compute?
:::

## What it replaces, and what it does not

The analytic planner knows the ball's flight, the impact and the flight to the target. The learned planner knows none of them. It sees:

- the last 16 raw camera fixes with their ages (not the EKF's estimate);
- the arm's reference joint position and velocity;
- the target on the ground.

It chooses a **swing**: six numbers, the time to impact, the strike point $(y, z)$ on the plane 0.62 m in front of the robot, the azimuth and elevation of the bat's face, and the bat speed. The same IK and swing QP as the analytic planner turn the swing into joint motion. Arm tracking stays model-based.

## The model

Components follow TD-MPC2, reduced to what a CPU and a once-per-pitch reward allow:

- **Encoder** to a 64-dimensional latent, two layers of 256 with LayerNorm and Mish.
- **Latent dynamics** one decision interval (40 ms) ahead, with the swing held.
- **Outcome head:** hit or miss, and where the ball lands relative to the target. It plays the role of TD-MPC2's reward model; the predicted hit quality is computed from it.
- **Policy prior** that seeds the planner.

**Losses.** Outcome prediction (binary cross-entropy for the hit, Huber loss for the landing offset of hits); latent consistency (the rolled-out latent must match the encoding of the later observation, with a stop-gradient target, and must predict the same outcome); and advantage-weighted regression for the prior.

The reward of a pitch arrives once, at its end, so there is no bootstrapped value function: the outcome head is trained on what actually happened after the last planner call of each pitch, the swing having been held from there on.

## Planning in latent space

MPPI over the six swing parameters: 192 samples around the prior, scored by the predicted hit quality averaged along a two-step latent rollout, three iterations of re-fitting to the best 24.

## The batting data

48,000 pitches played by a **noisy analytic batter**: its strike, perturbed by up to ±8 cm, ±12° and ±0.6 m/s, replanned at random moments so that the last plan of a pitch falls anywhere in the flight. Every step after that last call is a sample: 535,111 samples, of which 32 % come from pitches that were hit. Most of the data is about swings committed to too early.

:::caution[Not independent of the model]
The learned planner was trained on data produced by the analytic planner. At run time it replaces the EKF, the flight model and the impact model, but it learned what a good swing looks like from them. A world model trained from random swings would see almost no hits.
:::

:::tip[Where in the code]
`learn/wm.py` (model, losses, `train`, `validate`), `learn/wm_planner.py` (`SwingDecoder`, `DataPlanner`, `WorldModelPlanner`, `collect`), `scripts/m2_worldmodel.py`.
:::
