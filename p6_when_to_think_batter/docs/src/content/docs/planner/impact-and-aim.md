---
title: "Aiming and the Impact"
description: "From the forecast to the bat pose and speed that send the ball to the target, by inverting a frictionless impact."
sidebar:
  order: 1
---

:::note[Guiding question]
Given where the ball will be and where it should go, what must the bat be doing at the moment of contact?
:::

## One planner call

A call searches a small grid of strikes: two strike planes ($x_c$ = 0.56 and 0.68 m in front of the robot) and three times of flight for the hit ball ($T_f$ = 0.75, 0.95, 1.15 s). For each of the six candidates:

1. **Crossing.** Where and when the forecast crosses $x = x_c$: position $p_b$, velocity $v_b$, spin $\omega$, and the (fractional) number of steps to it.
2. **Aim.** The launch velocity $v_\text{out}$ that carries the ball from $p_b$ to the target in time $T_f$ under the flight model, with the spin it keeps. This is a shooting problem: start from the drag-free solution and correct it four times with the error of a coarse (50 ms) rollout. It lands within 5 cm of the target (a test checks it against the exact 10 ms model).
3. **Impact.** Invert the impact (below) for the face normal and the bat speed.
4. **Pose.** Five-degree-of-freedom inverse kinematics: blade centre at $p_b - (r + t/2)\,n$, face normal $n$.
5. **Velocity and swing.** See [The swing](/planner/swing/).

The cheapest candidate wins; its cost is the swing's effort, plus how close it runs to the joint speed limits, plus whatever it cannot reach, with a small bonus for the candidate the previous plan chose so the plan does not flip between strikes.

## Inverting the impact

The contact is frictionless, so the impulse is along the face normal $n$. With restitution $e$ and the bat's velocity $v_c$ at the contact,

$$
v_\text{out} = v_b - (1+e)\,\big((v_b - v_c)\cdot n\big)\, n .
$$

The change $v_\text{out} - v_b$ is parallel to $n$, so the normal is fixed by what the ball must do:

$$
n = \frac{v_\text{out} - v_b}{\lVert v_\text{out} - v_b \rVert}, \qquad
v_c \cdot n = v_b \cdot n + \frac{\lVert v_\text{out} - v_b \rVert}{1+e}.
$$

The bat only needs speed along its normal; motion along the face changes nothing. That leaves the arm two of its seven degrees of freedom to spend on comfort.

If the required normal speed is beyond the joint speed limits, it is scaled down: the ball is still hit, along the right direction, but lands short.

## The restitution the planner needs

The pair's restitution is 0.8, but the ball leaves the bat more slowly than that predicts, because the wrist gives under the impulse. The planner's $e$ was fitted on swings made with the true ball state (so the fit sees only the impact): the mean landing error along the range is linear in $e$ and crosses zero at **0.587**.

<figure class="plot">
<img class="fig-light" src="/fig/calibration-light.svg" alt="Mean landing error against the restitution assumed by the planner">
<img class="fig-dark" src="/fig/calibration-dark.svg" alt="Mean landing error against the restitution assumed by the planner">
<figcaption>200 swings with the true ball state per value. Fitting on the ball's velocity change against the commanded bat speed gives 0.573 instead; the landing fit also absorbs the small gap between commanded and realised bat speed.</figcaption>
</figure>

:::tip[Where in the code]
`control/planner.py` (`Aimer`, `SwingPlanner.strikes`, `plan`), `scripts/m1_hitter.py` (`calibrate`), `tests/test_planner.py`.
:::
