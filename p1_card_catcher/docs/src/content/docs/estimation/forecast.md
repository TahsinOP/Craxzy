---
title: "Forecasting the Catch Point"
description: "Rolling the state forward to the catch plane, projecting its covariance onto the plane, and why time-correlated model error changes the answer by a factor of six."
sidebar:
  order: 3
---

:::note[Guiding question]
The filter knows the state now. What is the distribution of the point where the object will cross the plane?
:::

## The mean

Starting from the filter's current estimate, the mean is rolled forward with the discrete model, with the future residuals $r_{k+j}$ if a network supplies them, until the signed distance to the plane changes sign. The crossing is then interpolated exactly as for the truth ([The Catch Plane](/overview/catch-plane/), equation 1):

$$
p_c = p_k + \alpha\,(p_{k+1} - p_k),\qquad t_c = (k + \alpha)\,h,\qquad v_c = \frac{p_{k+1} - p_k}{h}.
$$

## From state covariance to catch-point covariance

Alongside the mean, the covariance is propagated step by step, $P \leftarrow F P F^\top + Q$. At the crossing step this gives the covariance $\Sigma_p$ of the **position at the nominal crossing instant**. That is not yet the covariance of the catch point, because an error in position also changes *when* the object crosses.

Let $\delta p$ be the position error at the nominal crossing time $t_c$. The perturbed trajectory near $t_c$ is $p_c + \delta p + v\,(t - t_c)$. It crosses the plane when

$$
n^\top\big(\delta p + v\,\delta t\big) = 0 \quad\Longrightarrow\quad \delta t = -\frac{n^\top \delta p}{n^\top v},
$$

and at that moment it is at

$$
\delta p_c = \delta p + v\,\delta t = \underbrace{\Big(I - \frac{v\,n^\top}{n^\top v}\Big)}_{J}\,\delta p. \tag{1}
$$

So

$$
\Sigma_c = J\,\Sigma_p\,J^\top,\qquad \operatorname{var}(t_c) = \frac{n^\top \Sigma_p\,n}{(n^\top v)^2}. \tag{2}
$$

$J$ is an oblique projection onto the plane along the velocity: $n^\top J = 0$, so $\Sigma_c$ is singular along $n$, as it should be for a point confined to a plane. Its in-plane block, $E\,\Sigma_c\,E^\top$ with $E = [e_u\ e_w]^\top$, is a 2 × 2 matrix whose 90 % ellipse is drawn below and whose trace drives the variance trigger of part VI.

Equation (1) also explains the choice of a tilted plane. The projection stretches errors by $1/\cos$ of the angle between $v$ and $n$; an object arriving nearly parallel to the plane would turn millimetres of position error into centimetres of catch-point error.

<figure class="plot">
<img class="fig-light" src="/fig/ellipses-light.svg" alt="Predicted catch point and 90 percent ellipse on the catch plane at four times before the catch, for the physics-only and hybrid predictors">
<img class="fig-dark" src="/fig/ellipses-dark.svg" alt="Predicted catch point and 90 percent ellipse on the catch plane at four times before the catch, for the physics-only and hybrid predictors">
<figcaption>One card throw. Each ellipse is the 90 % region of the forecast at that time before the catch; the star is where the card actually crossed. The physics-only ellipses shrink too fast and at 100 ms the truth is outside. The hybrid's contain it throughout.</figcaption>
</figure>

The linearisation (1)–(2) is tested against Monte Carlo: sampling states from the filter covariance, rolling each forward, and comparing the sample covariance of the crossing points with $\Sigma_c$. They agree within 12 %.

## Model error that persists

The filter assumes the unmodelled acceleration is white. Over one step that is harmless. Over a forecast of 50 steps it is badly wrong, and the two assumptions can be compared in closed form for a single axis.

Let the acceleration error have standard deviation $\sigma$ at every step, and forecast over $T = Nh$.

- **White.** Each step adds an independent $h\,\sigma$ to the velocity. The velocity error has variance $N h^2\sigma^2 = \sigma^2 h\,T$, and integrating,
$$
\operatorname{var}(\delta p) \approx \tfrac13\,\sigma^2\,h\,T^3 .
$$
- **Fully persistent.** The same error at every step is a constant acceleration bias, $\delta p = \tfrac12\,\sigma\,T^2$:
$$
\operatorname{var}(\delta p) = \tfrac14\,\sigma^2\,T^4 .
$$

The ratio of standard deviations is $\sqrt{3T/(4h)}$: for $T = 0.4$ s and $h = 8$ ms, a factor of **6**. Note also that the white-noise answer shrinks with the timestep, which is a warning sign by itself: the uncertainty of a physical forecast should not depend on how finely one integrates.

### The AR(1) model

Real aerodynamic error is in between. The forecast models it as

$$
e_j = \sigma_j\,\eta_j,\qquad \eta_{j+1} = \varphi\,\eta_j + \sqrt{1 - \varphi^2}\;\xi_j,\qquad \xi_j \sim \mathcal N(0, I),\quad \eta_0 \sim \mathcal N(0, I), \tag{3}
$$

a unit-variance first-order autoregressive process per axis, scaled by the step's standard deviation $\sigma_j$. The factor $\sqrt{1-\varphi^2}$ keeps $\operatorname{var}(\eta_j) = 1$ for all $j$, so $\sigma_j$ keeps its meaning. $\varphi = 0$ is white noise; $\varphi \to 1$ is a constant bias.

Coloured noise is handled by **state augmentation**: the error state becomes $(\delta x, \eta) \in \mathbb R^{10}$ with

$$
\begin{bmatrix}\delta x_{j+1} \\ \eta_{j+1}\end{bmatrix} = \underbrace{\begin{bmatrix} F_j & B_j \\ 0 & \varphi I\end{bmatrix}}_{F^a_j}\begin{bmatrix}\delta x_j \\ \eta_j\end{bmatrix} + \begin{bmatrix}0 \\ \sqrt{1-\varphi^2}\,\xi_j\end{bmatrix},\qquad B_j = \begin{bmatrix} h^2\operatorname{diag}(\sigma_j) \\ h\operatorname{diag}(\sigma_j) \\ 0\end{bmatrix},
$$

and covariance recursion

$$
P^a_{j+1} = F^a_j\,P^a_j\,{F^a_j}^{\!\top} + \begin{bmatrix}0 & 0\\ 0 & (1-\varphi^2) I\end{bmatrix},\qquad P^a_0 = \begin{bmatrix}P & 0 \\ 0 & I\end{bmatrix}. \tag{4}
$$

The cross-covariance between $\delta x$ and $\eta$ that builds up in $P^a$ is precisely the memory that white noise lacks. With $\varphi = 0$, (4) reduces to $P \leftarrow FPF^\top + G\,\sigma^2 G^\top$, the filter's own recursion. The coefficient $\varphi$ is fitted on held-out data in [Calibration](/residual/calibration/); for the card it is 0.92, a correlation time of about 12 steps or 100 ms.

The slider below runs recursion (4) for one axis, drag-free, starting from a well-known state.

<figure class="widget" id="w-ar">
<div class="widget-controls">
<label for="w-ar-phi">correlation φ</label><input id="w-ar-phi" type="range" min="0" max="0.99" step="0.01" value="0.92"><output id="w-ar-phi-out">0.92</output>
<label for="w-ar-sig">acceleration error σ</label><input id="w-ar-sig" type="range" min="0.2" max="8" step="0.1" value="4"><output id="w-ar-sig-out">4.0 m/s²</output>
</div>
<svg id="w-ar-svg" role="img" aria-label="Standard deviation of the forecast position against forecast horizon, for white and for correlated acceleration error"></svg>
<div class="widget-summary">At 400 ms ahead: <strong id="w-ar-w">–</strong> cm if the error is white, <strong id="w-ar-c">–</strong> cm with correlation φ, a factor of <strong id="w-ar-r">–</strong>.</div>
<figcaption>One axis, h = 8 ms, initial uncertainty 3 mm and 5 cm/s. The dashed curve is what a filter that assumes white noise believes; the solid curve is recursion (4).</figcaption>
</figure>
<script src="/widgets.js"></script>
<script>
(function(){
var pI=document.getElementById('w-ar-phi'),sI=document.getElementById('w-ar-sig');
var draw=CCPlot(document.getElementById('w-ar-svg'),{xlabel:'forecast horizon [s]',ylabel:'std of the forecast position [cm]'});
function run(phi,sig){var h=0.008,n=63,P=[[9e-6,0,0],[0,2.5e-3,0],[0,0,1]],out=[Math.sqrt(P[0][0])*100];
for(var k=0;k<n;k++){var F=[[1,h,h*h*sig],[0,1,h*sig],[0,0,phi]],FP=[[0,0,0],[0,0,0],[0,0,0]],N=[[0,0,0],[0,0,0],[0,0,0]],i,j,m;
for(i=0;i<3;i++)for(j=0;j<3;j++)for(m=0;m<3;m++)FP[i][j]+=F[i][m]*P[m][j];
for(i=0;i<3;i++)for(j=0;j<3;j++)for(m=0;m<3;m++)N[i][j]+=FP[i][m]*F[j][m];
N[2][2]+=1-phi*phi;P=N;out.push(Math.sqrt(P[0][0])*100);}
return out;}
function redraw(){var phi=parseFloat(pI.value),sig=parseFloat(sI.value);
document.getElementById('w-ar-phi-out').textContent=phi.toFixed(2);
document.getElementById('w-ar-sig-out').textContent=sig.toFixed(1)+' m/s²';
var a=run(phi,sig),b=run(0,sig),t=a.map(function(_,i){return i*0.008;});
draw([{x:t,y:a,color:'var(--sl-color-accent)',label:'correlated error, AR(1) with φ'},{x:t,y:b,color:'var(--sl-color-gray-3)',dash:'5 4',label:'white error (φ = 0)'}],{xmax:0.5});
document.getElementById('w-ar-w').textContent=b[50].toFixed(2);
document.getElementById('w-ar-c').textContent=a[50].toFixed(2);
document.getElementById('w-ar-r').textContent=(a[50]/b[50]).toFixed(1);}
pI.addEventListener('input',redraw);sI.addEventListener('input',redraw);redraw();
})();
</script>

The augmented forecast is tested against Monte Carlo with a sampled AR(1) sequence; it agrees within 15 %.

## What the forecast returns

Per world, in one call: whether a crossing was found, the time to it, the catch point, the arrival velocity, $\Sigma_c$, its in-plane block and $\operatorname{var}(t_c)$. The controller uses the first four; the triggers use the in-plane block.

:::tip[Where in the code]
`estim/ekf.py` (`forecast`), `estim/predictor.py` (`PhysicsPredictor`, `HybridPredictor`, `OraclePredictor`, one interface: `step`, `forecast`, `reset`). Tests: `test_forecast_covariance_matches_monte_carlo`, `test_forecast_covariance_with_correlated_noise_matches_monte_carlo`.
:::
