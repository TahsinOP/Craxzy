---
title: "The Big Idea"
description: "Why hitting a pitched ball is a forecasting problem with a deadline, and why the interesting question is not how to plan but when."
sidebar:
  order: 1
---

:::note[Guiding question]
If the planner gives a perfect swing for the forecast it is given, and the forecast keeps improving, how often should it be asked?
:::

## Hitting is forecasting

A pitch takes 480 to 760 ms to reach the plate. The arm needs most of that time to move the bat into place, so it must commit to a swing long before the ball arrives, on a forecast of where the ball will be. That forecast starts poor and gets better with every camera frame:

| time left to the plate | 575 ms | 350 ms | 175 ms | 75 ms | 25 ms |
|---|---|---|---|---|---|
| crossing-point error, median | 17 cm | 6.4 cm | 2.0 cm | 0.8 cm | 0.5 cm |

The bat is 9 cm wide. A swing planned on the first forecast misses most pitches; a swing re-planned on the last one meets almost every ball. In between lies the design space.

The forecast starts so poorly mainly because of **spin**. The pitching machine is zeroed in, so the spin does not change where a pitch arrives, but it changes the path: a ball with backspin floats, a ball with topspin dives, and for the first tenth of the flight the two look the same. The camera cannot see spin; the filter has to infer it from the curvature of the path, which takes time.

## The planner is the expensive part

Each planner call does a small search: two strike planes, three times of flight for the hit ball, and for each candidate an aiming problem, an inversion of the impact, an inverse-kinematics solve and a quadratic program per joint. It takes about 9 ms on one core (median; the slow tail reaches 20 ms and more), close to the 10 ms control period. Calling it at every step is affordable in simulation and is the baseline here, but it is not free, and on a real robot with a richer planner it would not be possible.

So the arm carries a **cached swing**, a joint trajectory that it tracks with computed torque, and at every step something must decide: keep executing the cached swing, or re-run the planner on the newest forecast? That something is a **compute trigger**, and comparing triggers is the research core of the project.

## The hybrid

The theme of the project series is *model what you can, learn the rest, and know when to trust each*. Here:

- **Modelled exactly:** the ball's flight (a closed-form copy of the simulator's step), the impact (frictionless, with an identified restitution), the arm's kinematics and its discrete dynamics.
- **Learned, as an alternative:** a latent world model that replaces the estimator, the flight model and the impact model with a network trained on batting data, planned over with MPPI.
- **Learned, as the decision of when to think:** a PPO policy that watches the innovation, the covariance and the time to impact and decides when to call the planner, with reward = hit quality − λ · planner calls.

## What came out

- The analytic planner, re-planned at every step, hits every pitch and lands 97.8 % in the target zone. Planned once, it hits 11.8 %.
- The classical event triggers (innovation, variance, send-on-delta) are **worse than a fixed rate** at the same number of calls. They spend their calls early and go quiet late.
- The learned trigger spends its calls late and beats the fixed rate at small budgets: 0.94 against 0.91 hit quality at about 4 calls, 0.81 against 0.58 at about 3.
- The learned planner hits but cannot aim: 55 % on target against 98 %.
- A mid-flight gust makes the data-driven triggers think more; a fixed schedule does not notice. No trigger recovers what the gust costs.

[Lessons and limits](/lessons/lessons/) collects what this means and what it does not.
