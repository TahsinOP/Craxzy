---
title: "The Discrete Physics Model"
description: "MuJoCo's implicitfast step for a body with gravity and quadratic drag, in closed form, and why the two obvious alternatives are wrong by 3 mm and 3 cm."
sidebar:
  order: 1
---

:::note[Guiding question]
If the predictor's model and the simulator's integrator differ by "only discretisation error", how large is that, and can it be removed entirely?
:::

## The continuous model

A thrown body with isotropic quadratic drag obeys

$$
\dot p = v,\qquad \dot v = a(v) + r,\qquad a(v) = g - k_d\,|v|\,v, \tag{1}
$$

where $g = (0, 0, -9.81)$ m/s², $k_d$ is the drag coefficient of [Objects and Air](/simulation/objects-and-air/) and $r$ is whatever acceleration the model leaves out: zero for the pure physics predictor, learned in part IV.

The simulator does not solve (1). It takes steps of $h = 8$ ms with a particular integrator, and over a 0.8 s flight the difference between "the ODE" and "what the simulator does" is larger than the whole error budget of a ball catch. So the predictor uses the simulator's step, not the ODE.

## What `implicitfast` does to a free body

MuJoCo's `implicitfast` integrator treats velocity-dependent forces **linearly-implicitly**. For a body of unit mass with acceleration $a(v)$ it solves

$$
\big(I - h\,\tfrac{\partial a}{\partial v}\big)\,(v_{k+1} - v_k) = h\,a(v_k),\qquad p_{k+1} = p_k + h\,v_{k+1}. \tag{2}
$$

Two things are worth noticing. The position update uses the **new** velocity (semi-implicit Euler). And the matrix on the left multiplies the whole right-hand side, gravity included: the damping of the drag term slightly reduces the effect of gravity within a step. Forgetting that second point costs 3 mm.

### The drag Jacobian

Write $s = |v|$ and $u = v/s$. Differentiating $a(v) = g - k_d\,s\,v$,

$$
\frac{\partial a}{\partial v} = -k_d\Big(s\,I + \frac{v\,v^\top}{s}\Big) = -k_d\,s\,(I + u\,u^\top) =: -K.
$$

$K$ is symmetric positive definite: drag resists motion in every direction, and twice as stiffly along the direction of travel.

### Closed form by Sherman–Morrison

With $\alpha = h\,k_d\,s$, the matrix to invert is $I + hK = (1+\alpha)\,I + \alpha\,u\,u^\top$, a multiple of the identity plus a rank-one term. The Sherman–Morrison formula gives

$$
(I + hK)^{-1} = \frac{1}{1+\alpha}\Big(I - \frac{\alpha}{1+2\alpha}\,u\,u^\top\Big).
$$

Applying it to $a(v) = g - k_d\,s^2\,u$, and using that $u$ is an eigenvector of $I + hK$ with eigenvalue $1 + 2\alpha$, the **effective acceleration of one step** is

$$
\boxed{\ w(v) := (I + hK)^{-1} a(v) = \frac{g}{1+\alpha} - \frac{\alpha\,(u^\top g)}{(1+\alpha)(1+2\alpha)}\,u - \frac{k_d\,s}{1+2\alpha}\,v\ } \tag{3}
$$

and the discrete model used everywhere in the project is

$$
v_{k+1} = v_k + h\,\big(w(v_k) + r_k\big),\qquad p_{k+1} = p_k + h\,v_{k+1}. \tag{4}
$$

As $h \to 0$, $\alpha \to 0$ and $w(v) \to g - k_d\,|v|\,v$: the continuous model is recovered.

## How much the alternatives cost

<figure class="plot">
<img class="fig-light" src="/fig/integrator-light.svg" alt="Distance from the MuJoCo trajectory of three discretisations of the same model, against flight time">
<img class="fig-dark" src="/fig/integrator-dark.svg" alt="Distance from the MuJoCo trajectory of three discretisations of the same model, against flight time">
<figcaption>A non-spinning ball simulated in MuJoCo, against three models with the same gravity and the same drag coefficient. The linearly-implicit step differs only by the Stokes term of the fluid model; in inviscid air the agreement is 1e-9 m.</figcaption>
</figure>

| model | error after 0.8 s | where it comes from |
|---|---|---|
| continuous-time ODE | 29 mm | a body falling under semi-implicit Euler is at $-\tfrac12 g\,T(T+h)$, not $-\tfrac12 g\,T^2$: an offset of $\tfrac12 g\,h\,T$ |
| semi-implicit Euler with explicit drag | 3.4 mm | drops the factor $(I + hK)^{-1}$ in (2) |
| equation (4) | 0.2 mm | the viscous (Stokes) drag $6\pi\beta r\,v/m$, which the fitted $k_d$ absorbs |
| equation (4), inviscid air | $10^{-9}$ m | round-off |

The first row deserves a derivation because it is not a small-$h$ subtlety. With $v_k = -g\,k\,h$ and $p_{k+1} = p_k + h\,v_{k+1}$,

$$
p_K = -g\,h^2\sum_{k=1}^{K} k = -\tfrac12\,g\,h^2 K(K+1) = -\tfrac12\,g\,T\,(T + h),
$$

so the simulated body is lower than the textbook parabola by $\tfrac12 g\,h\,T = 3.1$ cm after 0.8 s at $h = 8$ ms. It is the same for every throw, which is exactly why it must be in the model: a constant 3 cm offset in the predicted catch point would be absorbed by nothing downstream.

Try it: the figure below integrates one throw three ways in your browser and plots the distance of the two "obvious" models from the linearly-implicit step.

<figure class="widget" id="w-int">
<div class="widget-controls">
<label for="w-int-h">timestep h</label><input id="w-int-h" type="range" min="1" max="16" step="1" value="8"><output id="w-int-h-out">8 ms</output>
<label for="w-int-kd">drag coefficient k_d</label><input id="w-int-kd" type="range" min="0" max="1.5" step="0.01" value="0.03"><output id="w-int-kd-out">0.03 /m</output>
</div>
<svg id="w-int-svg" role="img" aria-label="Distance from the linearly-implicit step of the continuous model and of explicit-drag Euler, against flight time"></svg>
<div class="widget-summary">After 0.8 s: continuous ODE off by <strong id="w-int-c">–</strong> mm, explicit-drag Euler by <strong id="w-int-e">–</strong> mm. The gravity offset ½·g·h·T alone is <strong id="w-int-g">–</strong> mm.</div>
<figcaption>Launch at (−3.5, 0.2, 3.0) m/s. k_d = 0.03 is the ball, 0.2 the plate, 1.45 the card. With no drag the Euler curve vanishes and only the gravity offset remains.</figcaption>
</figure>
<script src="/widgets.js"></script>
<script>
(function(){
var hI=document.getElementById('w-int-h'),kI=document.getElementById('w-int-kd');
var draw=CCPlot(document.getElementById('w-int-svg'),{xlabel:'flight time [s]',ylabel:'distance from the simulator step [mm]'});
var g=[0,0,-9.81];
function acc(v,kd){var s=Math.hypot(v[0],v[1],v[2]);return [g[0]-kd*s*v[0],g[1]-kd*s*v[1],g[2]-kd*s*v[2]];}
function wimp(v,kd,h){var s=Math.hypot(v[0],v[1],v[2])||1e-12,u=[v[0]/s,v[1]/s,v[2]/s],al=h*kd*s,ug=u[2]*g[2];
var c1=1/(1+al),c2=al/((1+al)*(1+2*al)),f=kd*s/(1+2*al);return [c1*g[0]-c2*ug*u[0]-f*v[0],c1*g[1]-c2*ug*u[1]-f*v[1],c1*g[2]-c2*ug*u[2]-f*v[2]];}
function redraw(){
var h=parseFloat(hI.value)/1000,kd=parseFloat(kI.value),T=0.8,n=Math.round(T/h);
document.getElementById('w-int-h-out').textContent=(h*1000).toFixed(0)+' ms';
document.getElementById('w-int-kd-out').textContent=kd.toFixed(2)+' /m';
var pi=[0,0,0],vi=[-3.5,0.2,3.0],pe=[0,0,0],ve=[-3.5,0.2,3.0],pc=[0,0,0],vc=[-3.5,0.2,3.0];
var t=[0],ec=[0],ee=[0],sub=40;
for(var k=0;k<n;k++){
var w=wimp(vi,kd,h);for(var i=0;i<3;i++){vi[i]+=h*w[i];pi[i]+=h*vi[i];}
var a=acc(ve,kd);for(i=0;i<3;i++){ve[i]+=h*a[i];pe[i]+=h*ve[i];}
for(var j=0;j<sub;j++){var hh=h/sub,a1=acc(vc,kd),vm=[vc[0]+hh/2*a1[0],vc[1]+hh/2*a1[1],vc[2]+hh/2*a1[2]],a2=acc(vm,kd);
for(i=0;i<3;i++){pc[i]+=hh*vm[i];vc[i]+=hh*a2[i];}}
t.push((k+1)*h);
ec.push(1000*Math.hypot(pc[0]-pi[0],pc[1]-pi[1],pc[2]-pi[2]));
ee.push(1000*Math.hypot(pe[0]-pi[0],pe[1]-pi[1],pe[2]-pi[2]));}
draw([{x:t,y:ec,color:'var(--sl-color-orange)',label:'continuous-time ODE'},{x:t,y:ee,color:'var(--sl-color-accent)',label:'semi-implicit Euler, explicit drag'}],{xmax:T});
document.getElementById('w-int-c').textContent=ec[n].toFixed(1);
document.getElementById('w-int-e').textContent=ee[n].toFixed(2);
document.getElementById('w-int-g').textContent=(1000*0.5*9.81*h*n*h).toFixed(1);}
hI.addEventListener('input',redraw);kI.addEventListener('input',redraw);redraw();
})();
</script>

## The Jacobian, analytically

The filter needs the derivatives of the step map (4). With $c_1 = 1/(1+\alpha)$, $c_2 = \alpha/((1+\alpha)(1+2\alpha))$, $f = k_d\,s/(1+2\alpha)$ and $\gamma = u^\top g$, equation (3) reads $w = c_1\,g - c_2\,\gamma\,u - f\,v$, and

$$
\begin{aligned}
\frac{\partial w}{\partial v} &= h k_d\,c_1'\,g\,u^\top - h k_d\,c_2'\,\gamma\,u u^\top - \frac{c_2}{s}\,u\,(g - \gamma u)^\top - \frac{c_2\,\gamma}{s}\,(I - u u^\top) - f\,I - \frac{k_d\,s}{(1+2\alpha)^2}\,u u^\top,\\[4pt]
\frac{\partial w}{\partial k_d} &= h s\,c_1'\,g - h s\,c_2'\,\gamma\,u - \frac{s}{(1+2\alpha)^2}\,v,
\end{aligned}
$$

with $c_1' = -1/(1+\alpha)^2$ and $c_2' = (1 - 2\alpha^2)/\big((1+\alpha)^2(1+2\alpha)^2\big)$. Then

$$
A := \frac{\partial v_{k+1}}{\partial v_k} = I + h\,\frac{\partial w}{\partial v},\qquad c := \frac{\partial v_{k+1}}{\partial k_d} = h\,\frac{\partial w}{\partial k_d}. \tag{5}
$$

These expressions are tested against central finite differences to $10^{-7}$.

## The residual is exact by construction

Given any simulated velocity sequence, define

$$
r_k := \frac{v_{k+1} - v_k}{h} - w(v_k). \tag{6}
$$

Substituting (6) into (4) returns $v_{k+1}$ identically, and the position update is the simulator's own. So the discrete model **with** the residual reproduces the simulated trajectory exactly, for any object. "True minus physics" is therefore a well-defined quantity per step with no integration error mixed into it, which is what makes it a clean learning target in [part IV](/residual/definition/).

:::tip[Where in the code]
`estim/physics.py`: `model_accel` is (3), `step` is (4), `model_jacobian` is (5), `residual_accel` is (6). Tests: `tests/test_ekf.py` (agreement with MuJoCo, the $h \to 0$ limit, the Jacobian, exact reproduction with the residual).
:::
