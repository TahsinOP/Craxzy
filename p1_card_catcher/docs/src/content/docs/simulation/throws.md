---
title: "Throws and Datasets"
description: "How a throw is aimed with the nominal model, randomised, simulated in batch, and filtered into a dataset."
sidebar:
  order: 3
---

:::note[Guiding question]
How do you generate thousands of throws that all land in a 40 cm window without making them all alike?
:::

## Aiming is a small shooting problem

A throw is defined by a launch position, a launch velocity, an attitude and a spin. Drawing the velocity at random would send most throws nowhere near the arm. Instead, each throw is **aimed**:

1. draw a launch position $p_0$ from a box 1.3 to 3.2 m in front of the arm (per object);
2. draw a target $p^\star$ uniformly in the catch window and a flight time $T$;
3. solve for the launch velocity $v_0$ that brings the *nominal* model, started at $(p_0, v_0)$, to $p^\star$ after $K = \operatorname{round}(T/h)$ steps.

Step 3 is a 3 × 3 root-finding problem, $\Phi_K(p_0, v_0) = p^\star$, where $\Phi_K$ is $K$ steps of the [discrete physics model](/estimation/discrete-model/). It is solved by Newton's method with a central-difference Jacobian,

$$
v_0 \leftarrow v_0 - \Big[\frac{\partial \Phi_K}{\partial v_0}\Big]^{-1}\big(\Phi_K(p_0, v_0) - p^\star\big),
$$

started from the drag-free solution of the same discrete scheme,

$$
v_0^{(0)} = \frac{p^\star - p_0 - \tfrac12\,g\,T\,(T + h)}{T}.
$$

The factor $T(T + h)$, where a textbook would have $T^2$, is the first appearance of a theme: the semi-implicit integrator moves a falling body by $\tfrac12 g\,T(T+h)$, and the aiming uses the integrator's own answer. Newton converges to $10^{-10}$ m in a few iterations for all throws at once.

## Randomisation

| | ball | plate | card |
|---|---|---|---|
| launch distance $x_0$ [m] | 2.4 – 3.2 | 2.0 – 2.8 | 1.25 – 1.75 |
| launch height $z_0$ [m] | 0.9 – 1.5 | 0.9 – 1.4 | 1.15 – 1.65 |
| nominal flight time [s] | 0.60 – 0.90 | 0.60 – 0.90 | 0.45 – 0.75 |
| attitude | uniformly random | edge-on, tilted (σ = 8°) | edge-on, tilted (σ = 15°) |
| spin about the normal [rad/s] | none | 25 – 50 | 20 – 60 |
| other angular velocity (σ) [rad/s] | 8 | 2 | 3 |

"Edge-on" means the flat object is launched like a frisbee, with its normal as close to vertical as the launch direction allows. The tilt and the wobble are what seed the flutter: two card throws with the same launch velocity and slightly different tilts land centimetres apart.

## From aim to dataset

The aim uses the nominal model; the flight is simulated with the full fluid model in `mujoco.batch`, a few thousand worlds per call. The real flight then differs from the aim by whatever the nominal model lacks. A throw is **accepted** if its simulated trajectory

- crosses the catch plane inside the window,
- does so at least 0.35 s after launch, so that there is something to estimate from,
- and with at least 0.5 m/s of speed normal to the plane, so that the crossing is well defined.

Rejected throws are redrawn. About 97 % of aimed ball throws are accepted, 31 % of plate throws and 8 % of card throws: the card is aimed with a model that misses by 23 cm at a window 40 cm wide. Rejection is on the true trajectory, so the accepted set is a fair sample of throws that do reach the arm.

| object | launch speed [m/s] | speed at the plane [m/s] | flight time [s] | miss of the nominal model from the true launch state |
|---|---|---|---|---|
| ball | 4.57 | 5.15 | 0.75 | 0.6 cm |
| plate | 4.41 | 4.13 | 0.74 | 10.3 cm |
| card | 3.92 | 2.41 | 0.63 | 22.8 cm |

The last column is the catch-point error of the nominal model given the *exact* launch state: the floor on what any drag-ballistic predictor can do without measurements. It is what the filter and the residual have to beat.

Each dataset holds, per throw, the positions, quaternions, linear and angular velocities at every step, and the true crossing (time, point, velocity, step index). Three seeds of 1,000 accepted throws per object are generated for evaluation, and larger sets with other seeds for training.

## What the measurements look like

The estimator never sees these arrays. It sees position fixes,

$$
z_k = p_k + \epsilon_k,\qquad \epsilon_k \sim \mathcal N(0,\ \sigma_m^2 I),\quad \sigma_m = 3\ \text{mm},
$$

one per simulator step (125 Hz), each lost independently with probability 0.10. The first fix of a track is always present. In closed loop the fixes stop as soon as the object touches the robot.

:::tip[Where in the code]
`models/throws.py` (`solve_launch_velocity`, `sample_throws`), `envs/flight.py` (`generate`, `FlightData`), `estim/runner.py` (`simulate_measurements`), `scripts/m0_dataset.py`.
:::
