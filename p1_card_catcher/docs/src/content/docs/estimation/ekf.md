---
title: "The Extended Kalman Filter"
description: "State, process and measurement models, the update in Joseph form, how the two noise parameters are fitted, and the test that the filter is consistent."
sidebar:
  order: 2
---

:::note[Guiding question]
From noisy position fixes with gaps, how do you get a velocity, a drag coefficient, and an honest statement of how well each is known?
:::

## State and process model

The filter state is position, velocity and the drag coefficient,

$$
x = \begin{bmatrix} p \\ v \\ k_d \end{bmatrix} \in \mathbb R^7 .
$$

Including $k_d$ lets a filter correct the nominal coefficient within a throw. The process model is the [discrete physics model](/estimation/discrete-model/) with noise:

$$
\begin{aligned}
v_{k+1} &= v_k + h\,\big(w(v_k;\,k_d) + r_k + n_k\big), & n_k &\sim \mathcal N(0,\ \sigma_a^2 I),\\
p_{k+1} &= p_k + h\,v_{k+1},\\
k_{d,k+1} &= k_{d,k}.
\end{aligned} \tag{1}
$$

$r_k$ is a known input: zero here, the network's prediction in part IV. $n_k$ is an unmodelled acceleration, white in time, with one standard deviation $\sigma_a$ per object.

### Jacobian and process noise

Linearising (1) with the blocks $A = \partial v_{k+1}/\partial v_k$ and $c = \partial v_{k+1}/\partial k_d$ from the previous page,

$$
F = \frac{\partial x_{k+1}}{\partial x_k} = \begin{bmatrix} I & hA & hc \\ 0 & A & c \\ 0 & 0 & 1 \end{bmatrix}.
$$

The first row follows from $p_{k+1} = p_k + h\,v_{k+1}$: position inherits $h$ times every sensitivity of the new velocity.

The noise $n_k$ enters the velocity as $h\,n_k$ and, through the same position update, the position as $h^2 n_k$. Hence

$$
Q = G\,(\sigma_a^2 I)\,G^\top,\qquad G = \begin{bmatrix} h^2 I \\ h I \\ 0 \end{bmatrix}. \tag{2}
$$

This $Q$ has rank 3: position and velocity noise within a step are perfectly correlated, as they must be when both come from one acceleration.

## Predict and update

**Predict.** Propagate the mean with (1) and the covariance with $P^- = F P F^\top + Q$.

**Update.** A fix $z_k = p_k + \epsilon_k$, $\epsilon_k \sim \mathcal N(0, \sigma_m^2 I)$, has measurement matrix $H = [\,I\ \ 0\ \ 0\,]$. With innovation $\nu = z - H x^-$,

$$
S = H P^- H^\top + \sigma_m^2 I,\qquad K = P^- H^\top S^{-1},\qquad x = x^- + K\nu,
$$

$$
P = (I - KH)\,P^-\,(I - KH)^\top + \sigma_m^2\,K K^\top. \tag{3}
$$

Equation (3) is the Joseph form. It is algebraically equal to $(I - KH)P^-$ for the optimal gain, but it is a sum of two positive semidefinite terms and so stays symmetric positive semidefinite under rounding, which matters for a filter that later exposes its covariance to a controller.

**When a fix is missing** (dropout), only the predict step runs. **After the update**, $k_d$ is clipped at zero: drag cannot be negative.

### Initialisation

A track starts at its first fix: position at the fix with variance $\sigma_m^2$, velocity zero with a standard deviation of 10 m/s (a deliberately diffuse prior), $k_d$ at the fitted value with a 5 % standard deviation. Forecasts are issued once a track has four fixes.

## Fitting the two noise parameters

The measurement noise $\sigma_m = 3$ mm is known. Two parameters per object are fitted on 500 throws that are used for nothing else.

**Drag coefficient.** By least squares on the step map itself,

$$
\hat k_d = \arg\min_{k}\ \sum_{\text{throws},\,k}\Big\|\frac{v_{k+1} - v_k}{h} - w(v_k;\,k)\Big\|^2,
$$

a smooth one-dimensional problem solved by bounded scalar minimisation. Fitting against the discrete map, and not against a continuous drag law, is what makes the ball's fitted value equal the analytic one.

**Process noise.** By maximum likelihood of the innovations. If the filter's model is right, each innovation is $\nu_k \sim \mathcal N(0, S_k)$, so the average negative log-likelihood per update is

$$
\mathcal L(\sigma_a) = \frac{1}{N}\sum_k \tfrac12\Big(\nu_k^\top S_k^{-1}\nu_k + \log\det S_k + 3\log 2\pi\Big). \tag{4}
$$

Too small a $\sigma_a$ makes $S_k$ too small and the quadratic term explodes; too large makes $\log\det S_k$ grow. The minimiser over a grid is taken.

| object | fitted $k_d$ [1/m] | spread of per-throw $k_d$ | $\sigma_a$ [m/s²] | $\mathcal L$ |
|---|---|---|---|---|
| ball | 0.0295 | 0.0000 | 0.2 | −12.93 |
| plate | 0.1960 | 0.0046 | 4 | −12.57 |
| card | 1.4524 | 0.0668 | 8 | −12.36 |

The ball's $k_d$ equals $\rho C\pi r^2/m$ and does not vary between throws. The card's process noise, 8 m/s², is most of gravity: the filter is being told that its model of the card is nearly worthless beyond a few steps, which is true.

## Is the filter consistent?

A filter is consistent when its covariance describes its actual errors. The test statistic is the normalised estimation error squared of position and velocity,

$$
\text{NEES}_k = (x_k - \hat x_k)^\top P_k^{-1} (x_k - \hat x_k) \ \sim\ \chi^2_6 \quad\text{if consistent},
$$

whose mean should be 6. On ball throws the test in `tests/test_ekf.py` checks exactly that. For the ball the filter passes; this is the ground on which the forecast covariance of the next page stands.

## Results

<figure class="plot">
<img class="fig-light" src="/fig/m1-light.svg" alt="Catch-point error and covariance coverage against time before the catch for ball, plate and card">
<img class="fig-dark" src="/fig/m1-dark.svg" alt="Catch-point error and covariance coverage against time before the catch for ball, plate and card">
<figcaption>Physics-only predictor, 3 seeds of 1,000 throws. Left: catch-point error. Right: how often the truth falls inside the predicted 90 % ellipse.</figcaption>
</figure>

| object | error at 200 ms before the catch | inside the 90 % ellipse |
|---|---|---|
| ball | 0.35 ± 0.00 cm | 91 % |
| plate | 5.00 ± 0.05 cm | 57 % |
| card | 6.64 ± 0.03 cm | 78 % |

Two conclusions. The ball needs nothing more: 3.5 mm is a fraction of the ball's radius. And for the plate and the card the filter is not only inaccurate but **over-confident**: the truth is inside its 90 % ellipse far less than 90 % of the time. The cause is the whiteness assumption in (1). An aerodynamic force that the model lacks now is still lacking 100 ms from now; errors that persist accumulate much faster than errors that average out. Part IV addresses the error, and the next page addresses the over-confidence.

:::tip[Where in the code]
`estim/ekf.py` (`BatchEKF`: all filters of a batch advance together in numpy; `fit_drag`), `estim/runner.py` (`run_filter`, `innovation_nll`), `scripts/m1_ekf.py`.
:::
