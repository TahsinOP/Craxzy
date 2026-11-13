---
title: "Shortcomings and Limits"
description: "What did not work, what the results do not show, and what was simplified."
sidebar:
  order: 2
---

:::note[Guiding question]
Where should a reader not trust these results, or not extend them?
:::

## What did not work

- **The classical event triggers lose to a fixed rate on ordinary pitches**, at every budget. At about 4 planner calls per pitch: innovation 0.45, drift 0.44 (3.4 calls), variance 0.29 (3.0 calls), against 0.91 for a fixed rate. See [Quality against compute](/when-to-think/results/).
- **The learned trigger does not go below about 2.5 calls per pitch.** At the highest price it was trained for (λ = 0.2) it still spends 2.5 calls for a quality of 0.68; the low-budget corner of its curve was not reached.
- **Reacting to a surprise does not recover the quality.** A gust takes the learned trigger from 0.94 to 0.64 and the fixed rate from 0.91 to 0.47; even replanning at every step reaches only 0.78 on gust pitches.
- **No trigger sees a spin change in time.** The change bends the path gradually; the innovation trigger replans within 150 ms of it in 28 % of pitches against 29 % without it.
- **The learned planner hits but does not aim**: 54.9 % on target against 96.9 %. Its outcome model's landing error (about 1 m) is the size of the target, and it was still improving with data at the largest budget tried.
- **The event trigger reacts sooner but does not hit better**, because the planner it wakes up still works from 30 ms-old frames.

## What the numbers do not show

- **Planning time is not charged.** A plan is applied in the step it was requested. The planner takes 9 to 21 ms against a 10 ms period, so on a real robot every call would also arrive late. `EnvParams.latency_steps` delays plans by whole steps; it is implemented and was not evaluated.
- **The PPO trigger was trained once.** Its evaluations use 3 seeds of pitches, but the policy itself comes from one training seed. The world models were trained with 3 seeds.
- **Sample sizes differ.** 200 pitches per seed and setting for M0, M1, M3 and M4; 128 for M2; 64 for the event-camera study.
- **The covariance is slightly conservative in mid-flight** (the 90 % ellipse holds the truth in up to 96 % of cases at 350 ms) and slightly optimistic at the plate (88 %).

## What was simplified

- **The impact is frictionless**, so the spin is unchanged by the hit and the impulse is along the face normal. This is what makes the impact invertible in closed form; a real bat grips the ball.
- **The bat is a flat blade**, and it collides with the ball only: it can pass through the arm, and joint limits during the follow-through are left to MuJoCo's limit constraints.
- **The pitches are slow** (9 to 13 m/s from 6.5 m) because the arm is slow (about 2 m/s at the bat).
- **The bat-ball contact is bouncier than a real one** (pair restitution 0.8) so that a 2 m/s bat can carry a ball several metres.
- **The camera is a delayed position sensor with noise**, not an image pipeline (the event camera of Part VII is the exception, and it only drives the trigger).
- **Surprises are of two kinds**, at random moments between 25 % and 60 % of the flight; real disturbances are less tidy.

## The stretch goals

All three stretch items are done: the [two-arm ping-pong rally](/rally/rally/), the [event camera](/surprises/events/), and Filament for the offline videos (every milestone video on this site is rendered with MuJoCo 4.0's physically based Filament renderer, offscreen, through Vulkan). The rally has its own limits, listed on [its page](/rally/results/).
