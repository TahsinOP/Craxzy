---
title: "Action Primitives"
description: "One five-parameter primitive for three tools: its timeline, the minimum-jerk time law, inverse kinematics with a null-space term, and why the depth is an absolute height."
sidebar:
  order: 3
---

:::note[Guiding question]
A planner cannot search over 180 time steps of seven joint angles. What is the smallest description of "do something to the clay with this tool" that still covers pressing, rolling and stamping, and makes the arm's part of the problem disappear?
:::

## Five numbers

Every action of every tool is one primitive,

$$
a = (x,\; y,\; \psi,\; d,\; \ell).
$$

| parameter | meaning | range |
|---|---|---|
| $x, y$ | centre of the action on the table plane, measured from the centre of the block | $\pm 55$ mm |
| $\psi$ | yaw of the tool about the vertical | press and stamp: $\pm 45°$; roller: $\pm 90°$ |
| $d$ | depth: how far the bottom of the tool head goes below the **undeformed** top surface | 2 to 14 mm |
| $\ell$ | horizontal travel at depth, centred on $(x, y)$ | roller: 20 to 100 mm; press and stamp: 0 |

The yaw ranges are the symmetries of the heads. The flat press is a square and the stamp is a cross, so both look the same after a quarter turn; the roller looks the same after a half turn.

The travel is along the tool's own $y$ axis, which is perpendicular to the roller's axis. With the tool's $x$ axis at angle $\psi$ in the table plane, the travel direction is

$$
t(\psi) = (\sin\psi,\; -\cos\psi),
$$

and the tool moves from $(x, y) - \tfrac{\ell}{2}\,t$ to $(x, y) + \tfrac{\ell}{2}\,t$.

### Depth is a height, not a penetration

The bottom of the tool is sent to the absolute height $z_{\text{top}} - d$, where $z_{\text{top}} = 44$ mm is the top of the untouched block. It is not sent "$d$ below wherever the surface is now".

This choice has three consequences, all wanted. The action means the same thing whatever the current shape, so the geometric prior of [part V](/dynamics/prior/) is a formula in $a$ alone. Pressing twice in the same place does not go twice as deep: the second press reaches the same height and finds less clay, so the total deformation is bounded by the 14 mm limit. And the planner needs no contact detection to know where the tool will stop.

## The timeline

A primitive is a fixed number of 5 ms steps, the same for every action of a tool, so that a batch of rollouts runs for the same length:

| segment | duration | press, stamp | roller |
|---|---|---|---|
| descend from hover height (15 mm above the untouched top) | 0.40 s | yes | yes |
| dwell | 0.10 s | yes | yes |
| slide | 0.80 s | | yes |
| ascend | 0.30 s | yes | yes |
| settle | 0.10 s | yes | yes |
| total | | 0.90 s, 180 steps | 1.70 s, 340 steps |

Each moving segment is a straight line in Cartesian space with a **minimum-jerk** time law. For a segment of duration $T$ the path parameter is

$$
s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5,\qquad \tau = t/T, \tag{1}
$$

which starts and ends with zero velocity and zero acceleration and satisfies $s(\tau) + s(1 - \tau) = 1$. The peak speed is $\tfrac{15}{8}$ times the mean speed, reached at the midpoint. A 29 mm descent in 0.4 s therefore peaks at 136 mm/s and the tool is slowing down when it meets the clay; the longest roll, 100 mm in 0.8 s, peaks at 234 mm/s.

Speed matters physically. The clay model has Rayleigh damping, a viscous stress proportional to the strain rate, so pressing faster needs more force. With these durations the largest joint torque in a press, a stamp and a roll is 24, 33 and 24 N m, against the 87 N m limit of the shoulder joints.

## From a Cartesian line to joint references

The tool tip is commanded as a position $p$ and the yaw $\psi$, with the tool axis straight down. The target orientation of the tip frame is

$$
R(\psi) = \begin{bmatrix} \cos\psi & \sin\psi & 0 \\ \sin\psi & -\cos\psi & 0 \\ 0 & 0 & -1 \end{bmatrix},
$$

whose columns are the tip's $x$ axis in the table plane, its $y$ axis $t(\psi)$, and its $z$ axis pointing down.

### Inverse kinematics

The Panda has seven joints and the task has six constraints, so there is a one-parameter family of solutions. The solver is damped least squares on the six-dimensional error $e$ (position error, and the rotation vector of $R(\psi)R^\top$), with a second term that uses the spare degree of freedom to stay near the home configuration $q_h$:

$$
\Delta q = J^\top\big(JJ^\top + \lambda I\big)^{-1} e \;+\; \big(I - J^{+}J\big)\,k\,(q_h - q). \tag{2}
$$

The projector $I - J^{+}J$ maps any joint motion into the null space of the task Jacobian $J$, so the second term does not change the tip to first order. It makes the solution unique and a continuous function of the target, which is what lets consecutive points on a line be solved by warm-starting from the previous one. The iteration stops when the error is below $10^{-7}$ (metres and radians).

:::caution[A projector that is only nearly a projector]
The first version used the damped inverse in both terms, $\big(I - J^\top(JJ^\top + \lambda I)^{-1}J\big)$. That matrix is not exactly a null-space projector: it leaks a little of the home-seeking term into the task. The iteration then settles where the leak balances the correction, at an error of $1.6 \times 10^{-6}$, and never reaches the tolerance. Using the exact pseudo-inverse $J^{+}$ in the projector, and keeping the damping only in the task term, fixed it: the iteration now reaches the tolerance for every target.
:::

Tool heads are symmetric under a half turn, so the yaw is taken modulo $\pi$ and mapped to the branch nearest the home wrist angle. Joint 7 then stays roughly within a quarter turn of its home position, far from its limits. Over 200 random targets covering the whole workspace the solver always converges and every joint stays at least 0.05 rad inside its range.

### Knots, then interpolation

Solving (2) at every one of 180 steps would cost more than it is worth. Each straight segment is instead sampled at **knots** every 4 mm, the knots are solved in sequence, and joint angles are interpolated linearly between knots in the path parameter $s$. The deviation of the tip from the straight line between two knots is of order $\delta^2 / (8R)$ for knot spacing $\delta$ and arm reach $R$: with $\delta = 4$ mm and $R \approx 0.5$ m, about 4 μm.

The reference handed to the simulator is, for each step $k$ and each joint, the position at the end of the step and the velocity that gets there:

$$
q^{\text{ref}}_k = q(t_{k+1}),\qquad \dot q^{\text{ref}}_k = \frac{q(t_{k+1}) - q(t_k)}{h},\qquad h = 5\ \text{ms}.
$$

That is a $180 \times 14$ array for a press. It is computed once per candidate action, before the rollout starts; the rollout itself is open loop in the references and closed loop in the servos ([Scene and Tools](/bench/scene/)).

## Between primitives

The arm starts every primitive at rest at hover height above the start point. In rollouts the move from the end of one primitive to the start of the next is not simulated: the clay is at rest and nothing touches it. [State, Save and Restore](/bench/state/) gives the measured cost of that shortcut, 0.03 mm RMS.

:::tip[Where in the code]
`control/primitives.py`: `PrimCfg` (durations and bounds), `bounds`, `waypoints`, `minjerk`, `travel_direction`, `Planner.reference` (knots and interpolation), `Planner.transit_reference` (the free-space move, for continuous execution and video).
`control/arm.py`: `tip_rotation`, `wrap_yaw`, `Kinematics.ik` (equation (2)).
Tests: `test_ik_reaches_whole_workspace`, `test_home_pose_points_down` in `tests/test_bench.py`; `test_minjerk_endpoints_and_symmetry`, `test_travel_direction_is_the_tool_y_axis` in `tests/test_mpc.py`.
:::
