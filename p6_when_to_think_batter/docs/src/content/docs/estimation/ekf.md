---
title: "The EKF and Its Forecast"
description: "A filter on delayed, range-dependent fixes; the forecast of the crossing point and its covariance; and a detector for its own surprises."
sidebar:
  order: 2
---

:::note[Guiding question]
How good is the forecast of where the ball will cross the plate, and does the filter know how good it is?
:::

## The filter

The state is position, velocity and spin, nine numbers. The process model is the flight model's step with its Jacobian; process noise is small (0.05 m/s² on the acceleration), because the model is exact. A fix is a position with noise that grows with range from the camera, $\sigma = 2\ \text{mm} + 2\ \text{mm/m} \cdot r$. The first fix initialises the position; velocity and spin start from a prior (the machine's typical release velocity, and up to 90 rad/s of spin uncertainty).

**Latency.** The camera's fix at step $k$ shows the ball at step $k - 3$. The filter therefore lives three steps in the past, and the forecast starts with three model steps to reach the present before continuing to the plate.

## The forecast

From the filter state, the flight model is rolled forward and the covariance is propagated with the Jacobians (no process noise over the forecast). Where the predicted path crosses the hitting plane gives the crossing point, the time to it, and its covariance in the plane. A position error $\delta p$ moves the crossing point by $\delta p - v\,\delta p_x / v_x$, which maps the 3-D covariance into the plane.

<figure class="plot">
<img class="fig-light" src="/fig/estimator-light.svg" alt="Crossing-point error and the filter's standard deviation against the time left to the plate; coverage of the 90 % ellipse">
<img class="fig-dark" src="/fig/estimator-dark.svg" alt="Crossing-point error and the filter's standard deviation against the time left to the plate; coverage of the 90 % ellipse">
<figcaption>3 seeds of 300 pitches, no arm involved. Left: the error falls by a factor of 30 over the flight; it crosses half the bat's width at about 250 ms. Right: the 90 % ellipse holds the truth in 88 to 96 % of cases, so the covariance can be trusted.</figcaption>
</figure>

| time left to the plate | 575 ms | 450 ms | 350 ms | 250 ms | 175 ms | 125 ms | 75 ms | 25 ms |
|---|---|---|---|---|---|---|---|---|
| error, median | 17.0 cm | 11.7 cm | 6.4 cm | 3.4 cm | 2.0 cm | 1.3 cm | 0.8 cm | 0.5 cm |
| filter's std | 21.3 cm | 14.0 cm | 9.0 cm | 4.8 cm | 2.7 cm | 1.7 cm | 1.0 cm | 0.6 cm |

Two properties of this curve shape the rest of the project. The error is large when the swing must start; and it shrinks by centimetres per frame early but by millimetres per frame late, while it is the late millimetres that decide whether a 9 cm blade meets a 7.3 cm ball. A trigger that thresholds "how much has the estimate changed" will see a great deal happening early and very little late.

## A detector for its own surprises

The filter knows nothing about gusts or spin changes. It does watch its normalised innovation squared, which under the model is χ² with 3 degrees of freedom. When the sum over the last 6 fixes exceeds 50 (a χ²(18) tail probability of about $10^{-4}$), it adds 0.8 m/s to its velocity standard deviation and 150 rad/s to its spin, a covariance reset, so it re-learns from the next frames instead of averaging the surprise away. On ordinary pitches it fires in fewer than 3 % of them. It trips on most gusts and on few spin changes; when it trips, it makes the covariance honest again but not the estimate better, because by the time a gust is visible above the camera noise few frames remain.

:::tip[Where in the code]
`estim/ekf.py` (`BallEKF`, `forecast`, `crossing_batch`), `scripts/m1_hitter.py` (`estimator`), `tests/test_ekf.py`.
:::
