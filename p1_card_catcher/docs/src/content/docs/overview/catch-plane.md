---
title: "The Catch Plane"
description: "The one geometric object every layer shares: a tilted plane, a window on it, and the definition of the catch point."
sidebar:
  order: 2
---

:::note[Guiding question]
"Where the object will be" is a curve. Which single point on it does the arm aim for?
:::

## A plane turns a trajectory into a point

A predicted trajectory is a curve in space and time. To aim, the arm needs one point and one instant. The project fixes a plane in front of the arm and defines the **catch point** as the place where the trajectory first crosses it:

$$
\Pi = \{\,x \in \mathbb R^3 : n^\top (x - c_0) = 0\,\},\qquad n = (\cos 30^\circ,\ 0,\ \sin 30^\circ),\quad c_0 = (0.52,\ 0,\ 0.752)\ \text{m}.
$$

The signed distance $s(x) = n^\top(x - c_0)$ is positive on the thrower's side. The plane is tilted by 30° so that objects, which arrive falling at 45° to 85° below the horizontal, cross it at a steep angle. A crossing at a shallow angle would turn a small error in height into a large error along the plane.

In-plane coordinates use an orthonormal basis,

$$
e_u = (0, 1, 0),\qquad e_w = n \times e_u = (-\sin 30^\circ,\ 0,\ \cos 30^\circ),\qquad (u, w) = \big(e_u^\top (x - c_0),\ e_w^\top (x - c_0)\big),
$$

so $u$ runs sideways and $w$ up the slope. Catch-point errors and covariances are always reported in $(u, w)$.

## The crossing, exactly as computed

Simulated and predicted trajectories are sequences of positions $p_0, p_1, \dots$ one step $h$ apart. The crossing is located inside the first step where the signed distance changes sign, by linear interpolation:

$$
s_k > 0 \ \ge\ s_{k+1},\qquad \alpha = \frac{s_k}{s_k - s_{k+1}},\qquad p_c = p_k + \alpha\,(p_{k+1} - p_k),\qquad t_c = (k + \alpha)\,h. \tag{1}
$$

The same function, `physics.plane_crossing`, is applied to the truth and to every prediction. This matters more than it looks: if truth and prediction used different definitions of "the crossing", the difference would show up as a prediction error of up to one step of travel, which is 4 cm for a ball at 5 m/s.

## The window

Only part of the plane is reachable in time. The window is

$$
u \in [-0.20,\ 0.20]\ \text{m},\qquad w \in [-0.14,\ 0.14]\ \text{m}.
$$

It was not chosen for convenience. A first version used ±28 cm by ±20 cm, and with a *perfect* predictor the arm still lost about 10 % of ball throws. Tracing them showed that targets in the bottom strip needed the elbow to travel up to 1.9 rad in 0.6 s, which the datasheet acceleration limit of 12.5 rad/s² does not allow. The window was re-centred on the region the arm does reach in time, and the perfect-predictor catch rate went to 99.6 %. A catch rate measured over an unreachable window would have mixed a geometry problem into every later result.

## What each layer takes from the plane

| layer | what it uses |
|---|---|
| throw generator | draws a target uniformly in the window, and keeps a throw only if the simulated flight crosses inside it |
| filter and forecast | equation (1) on the predicted path, and the projection of the position covariance onto the plane ([Forecast](/estimation/forecast/)) |
| controller | the catch point as the hand position; the arrival direction, limited to a cone around $n$, as the palm direction |
| triggers | the trace of the 2 × 2 in-plane covariance |

:::tip[Where in the code]
`card_catcher/config.py` (`Plane`), `configs/default.yaml` (`plane:`), `estim/physics.py` (`plane_crossing`).
:::
