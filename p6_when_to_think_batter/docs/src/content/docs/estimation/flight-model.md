---
title: "The Flight Model"
description: "The ball's flight in closed form, identical to the simulator's step to 1e-9 m, and why that matters."
sidebar:
  order: 1
---

:::note[Guiding question]
Can the estimator's model of the ball be not approximately right, but exactly the simulator?
:::

## What MuJoCo computes

The ball uses MuJoCo's ellipsoid fluid model. For a sphere it reduces to three effects: drag with a linear (viscous) and a quadratic part, lift from spin, and a damping of the spin itself:

$$
\dot v = g - (k_1 + k_2 \lVert v \rVert)\, v + c\,(\omega \times v), \qquad
\dot\omega = -(k_3 + k_4 \lVert \omega \rVert)\,\omega .
$$

The lift coefficient $c$ is the Magnus term minus the gyroscopic part of the added-mass force. Five numbers describe the ball, and none of them has to be guessed: `FlightModel.identify` sets a velocity or a spin on the ball, runs `mj_forward` and reads `qfrc_passive`, twice per coefficient.

## Matching the step, not the equation

The project uses the `Euler` integrator, in which fluid forces are **explicit**. One simulator step is then exactly

$$
v_{k+1} = v_k + h\, a(v_k, \omega_k), \qquad p_{k+1} = p_k + h\, v_{k+1}, \qquad \omega_{k+1} = \omega_k + h\, \alpha(\omega_k),
$$

and `FlightModel.step` implements precisely this, with its analytic Jacobian for the filter. A test throws a spinning pitch in MuJoCo and in the model and compares: **positions agree to $10^{-9}$ m**, in the ball-only model and in the full cage with the arm present.

:::caution[Why not `implicitfast`?]
P1 used `implicitfast`, where the drag is treated linearly-implicitly and the closed form is a matrix inverse. With spin, the derivatives of the fluid force couple linear and angular velocity and that inverse no longer has a tidy form. With `Euler` the step is explicit and the model is exact by construction. The arm's joint damping is still implicit under `Euler`, which the computed-torque controller accounts for.
:::

## What exactness buys

Everything downstream compares forecasts with outcomes. If the model differed from the simulator by a few millimetres, every error budget would carry that floor, and the pitching machine, which aims through the same model, would miss its own aim points. Because the model is the simulator's step:

- the pitching machine hits its drawn aim point to 1.5 mm;
- the EKF's process noise can be tiny, so its covariance reflects measurement noise and unknown spin, not model error;
- the landing point of a hit ball can be extrapolated a few steps after the impact instead of simulating it to the ground (a test confirms the extrapolation follows the continued simulation to $10^{-8}$);
- in M4, a surprise is the only thing that can make the model wrong, so the innovation measures exactly that.

:::tip[Where in the code]
`estim/flight.py` (`FlightModel.identify`, `step`, `jacobian`, `rollout`, `landing`), `tests/test_flight.py`.
:::
