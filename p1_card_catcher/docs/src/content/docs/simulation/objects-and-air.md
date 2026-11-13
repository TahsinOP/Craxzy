---
title: "Objects and Air"
description: "The three thrown objects, MuJoCo's ellipsoid fluid model, and the one drag coefficient the physics predictor keeps."
sidebar:
  order: 2
---

:::note[Guiding question]
What forces act on the object in flight, and which of them does the physics model know about?
:::

## Three objects, three difficulties

| object | geometry | mass | what makes it hard |
|---|---|---|---|
| ball | sphere, radius 25 mm | 40 g | nothing: a point mass with drag |
| plate | disc, radius 80 mm, 6 mm thick | 60 g | lift and a slowly precessing attitude; too wide for the gripper |
| card | 63 × 88 × 0.7 mm | 1.8 g | flutter; terminal speed 2.4 m/s; thinner than any contact margin |

The ball is small and soft so that a parallel gripper can hold it. The card is poker-sized.

## The fluid model

All three use MuJoCo's ellipsoid fluid model (`fluidshape="ellipsoid"`), which replaces the geom by its equivalent ellipsoid with semi-axes $(a, b, c)$ and applies drag, lift and added-mass forces to it. Its translational drag, for body-frame velocity $v$, is

$$
f_{\text{drag}} = -\Big[\ \underbrace{6\pi\beta\,r_{\text{eq}}}_{\text{Stokes}} + \rho\,|v|\,\big(C_{\text{blunt}}\,A_{\text{proj}}(v) + C_{\text{slender}}\,(A_{\max} - A_{\text{proj}}(v))\big)\Big]\,v, \tag{1}
$$

with $\rho = 1.2$ kg/m³ the air density, $\beta = 1.8\times10^{-5}$ Pa·s its viscosity, $A_{\text{proj}}(v)$ the area of the ellipsoid projected along $v$, $A_{\max}$ its largest projected area, $r_{\text{eq}} = (a + b + c)/3$, and the default coefficients $C_{\text{blunt}} = 0.5$, $C_{\text{slender}} = 0.25$.

The projected area of an ellipsoid along a unit direction $d$ is

$$
A_{\text{proj}}(d) = \pi\sqrt{(bc\,d_x)^2 + (ca\,d_y)^2 + (ab\,d_z)^2}.
$$

### For a sphere the drag is exactly isotropic

With $a = b = c = r$, $A_{\text{proj}} = A_{\max} = \pi r^2$ for every direction, and the quadratic part of (1) collapses to

$$
a_{\text{drag}} = -k_d\,|v|\,v,\qquad k_d = \frac{\rho\,C_{\text{blunt}}\,\pi r^2}{m} = \frac{1.2 \cdot 0.5 \cdot \pi \cdot 0.025^2}{0.040} = 0.0295\ \text{m}^{-1}. \tag{2}
$$

This is the drag form of the physics predictor. For the ball it is the simulator's drag, not an approximation of it; the value fitted from simulated flights in [The EKF](/estimation/ekf/) is 0.0295.

The general formula was checked against the simulator rather than trusted: `ObjectSpec.kd_axis` evaluates (1) for motion along each body axis, and the result agrees with `mjData.qfrc_fluid` to six digits.

### For a plate or a card it is not

The same formula gives a drag coefficient that depends strongly on attitude:

| object | $k_d$ edge-on [1/m] | $k_d$ face-on [1/m] |
|---|---|---|
| ball | 0.029 | 0.029 |
| plate | 0.104 | 0.201 |
| card | 0.731 | 1.451 |

and drag is the smaller part of the story. The fluid model also applies a Kutta lift perpendicular to the velocity, a Magnus force from spin, and added-mass moments that turn the object. A card thrown edge-on pitches up, stalls, and flips: in the dataset its surface normal changes sign 1.4 times per throw on average, it reaches 44 rad/s, and it slows from 3.9 m/s at launch to 2.4 m/s at the plane.

The physics predictor keeps the isotropic form (2) for every object, with one fitted coefficient each. Everything else is by definition the **residual**, and its size sets how hard the learning problem is:

<figure class="plot">
<img class="fig-light" src="/fig/residual-light.svg" alt="Residual acceleration along one throw of each object and its distribution over 1,000 throws">
<img class="fig-dark" src="/fig/residual-dark.svg" alt="Residual acceleration along one throw of each object and its distribution over 1,000 throws">
<figcaption>The acceleration that the drag-ballistic model does not explain. For the ball it is 0.04 m/s² rms, for the plate 1.8, for the card 5.2, which is half of gravity. The card's oscillation is flutter.</figcaption>
</figure>

## Why not a better physics model for the card?

One could write the anisotropic drag and the lift into the predictor. It would need the card's attitude and angular velocity, and the sensor gives position only. Estimating the attitude of a tumbling card from the curvature of its track is exactly the kind of inference a sequence model is good at, and that is the division of labour chosen here: the predictor keeps the part of the physics that needs no hidden state, and the network infers the rest from the history of the track.

:::tip[Where in the code]
`models/objects.py` (`ObjectSpec`, `kd_axis`, `projected_area`), `scripts/m0_dataset.py` (drag check against `qfrc_fluid`, flutter statistics).
:::
