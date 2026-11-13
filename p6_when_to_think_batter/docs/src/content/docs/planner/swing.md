---
title: "The Swing"
description: "Five-degree-of-freedom IK, the strike velocity, a joint QP per joint in the simulator's own integrator, and discrete computed torque."
sidebar:
  order: 2
---

:::note[Guiding question]
How do seven joints get the blade to a point, facing a direction, moving at a speed, at a time, within their limits?
:::

## The strike pose

The task is five constraints: three for the blade centre, two for the face normal (rotation about the normal itself changes nothing). Damped least squares solves it, warm-started from the previous plan's solution for the same candidate, with the spare freedom pulled towards a *nominal strike posture* in the null space.

That nominal posture is the one, for a ball through the middle of the zone, from which the blade can move fastest along its normal within the joint speed limits: 2.04 m/s, found by 600 random restarts. The **ready pose** is that posture wound back along the swing, each joint by 1.5 times the distance it needs to reach its strike speed at its acceleration limit, so a swing starts with its run-up in place.

## The strike velocity

Only the speed along the normal is constrained. The joint velocity minimises the joint speeds relative to their limits, plus a penalty on the blade's motion along its face (so a timing error slides the contact point less):

$$
\dot q = \arg\min_{\dot q}\; \sum_j \Big(\tfrac{\dot q_j}{\dot q_{j,\max}}\Big)^2 + \mu\, \lVert J_t \dot q \rVert^2
\quad \text{s.t.}\quad n^\top J_v \dot q = s .
$$

Asking for pure translation of the blade instead would cap the bat at about 0.9 m/s.

## The joint trajectory

Each joint solves, in its accelerations $a_0 \dots a_{N-1}$,

$$
\min \sum_k a_k^2 + w \sum_k (a_{k+1} - a_k)^2
\quad\text{s.t.}\quad q_N = q_s,\ \dot q_N = \dot q_s,\ \text{position, velocity and acceleration limits,}
$$

written in the simulator's integrator ($\dot q_{k+1} = \dot q_k + h a_k$, $q_{k+1} = q_k + h \dot q_{k+1}$), so the reference is exactly trackable. Without active limits the problem is an equality-constrained least squares with a banded matrix, solved in closed form; if that solution violates a limit, OSQP solves the full QP. If even that is infeasible, because the strike is too far away for the time left, a soft version replaces the terminal equalities with heavy penalties and returns the closest approach the limits allow. That case matters: it is what makes a late replan useful even when the new strike can no longer be reached exactly.

The swing has three phases: the approach to $(q_s, \dot q_s)$, a short constant-velocity stretch through the predicted impact (so a timing error of a step moves the contact along the swing without changing the bat's velocity), and a brake to rest at the acceleration limit.

## Tracking

Computed torque in the integrator's own form, $\tau = (M + hD)\,a + c - f_\text{passive}$, with gains that put both poles of the sampled tracking error at $e^{-\omega h}$. With a reference that obeys the integrator, tracking is exact up to torque saturation, which did not occur in any of the runs reported here.

:::tip[Where in the code]
`control/arm.py` (`ik`, `strike_velocity`, `torque`, `NOMINAL_STRIKE`, `READY`), `control/traj.py` (`SwingQP`), `scripts/design_ready.py`.
:::
