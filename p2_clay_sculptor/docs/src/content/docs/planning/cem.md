---
title: "The Planner"
description: "Greedy receding-horizon planning over tool primitives with the cross-entropy method: the search space, a proposal that looks at the target, circular statistics for the yaw, and the stopping rule."
sidebar:
  order: 1
---

:::note[Guiding question]
The planner will be run with five different rollout models, from an exact simulator to a network that is wrong in ways nobody has measured yet. What has to be true of the planner so that a difference in the outcome can be blamed on the model and on nothing else?
:::

## One decision

A target is a height map $H^\star$ and a tool. At each decision the planner looks at the current shape and chooses one [primitive](/bench/primitives/) $a = (x, y, \psi, d, \ell)$ for that tool:

$$
a^\ast = \arg\min_{a \in \mathcal{A}}\; c\big(\hat H(a),\, H^\star\big), \tag{1}
$$

where $\hat H(a)$ is the height map the **rollout model** predicts after executing $a$, and $c$ is the height-map RMSE of [Nodes, Vertices, Height Map](/shape/representation/). Then $a^\ast$ is executed in the simulator, the true shape is observed, and the next decision starts from there.

This is model-predictive control with a horizon of one primitive. It is greedy, and that is a real limitation: the planner never accepts a worse shape now for a better one later. It is also what keeps the comparison clean, because the only thing a rollout model is ever asked is "what does this one action do to this shape?", which is exactly what the learned models are trained on.

The rollout model enters only through $\hat H$. Everything else on this page is identical for the simulator, the prior, the two graph networks and the U-Net.

## The cross-entropy method

Problem (1) has five continuous parameters (four for the press and the stamp, whose travel is zero), a cost with no usable gradient through the simulator, and several local minima: a dent can be removed from the cost by pressing at its centre or, less well, at any place that overlaps it. The cross-entropy method handles this with a population.

1. Draw a population of $P = 32$ actions.
2. Evaluate the cost of each with the rollout model.
3. Keep the $E = 6$ best of everything evaluated so far (the elite set).
4. Fit a Gaussian to the elite set, one mean and one standard deviation per parameter, and draw the next population from it, wrapped back into the bounds.
5. Repeat steps 2 to 4 for $I = 3$ iterations and return the best action ever evaluated.

One decision therefore costs $P \cdot I = 96$ model evaluations, whatever the model. With the simulator as the model, that is 96 simulated primitives.

Two details keep a small population from collapsing. The elite set is taken over **all** iterations, so a good early sample is never lost. And the standard deviation of each parameter is floored at 3 % of its range, so the search does not freeze on the first cluster it finds.

### The yaw is an angle

The press and the cross stamp look the same after a quarter turn, the roller after a half turn, so the yaw lives on a circle of period $T = \pi/2$ or $\pi$ and is **wrapped**, not clipped, into its interval.

Fitting a Gaussian to wrapped values needs care. Suppose the elite yaws of the stamp are $+43°$, $-43°$, $+44°$ and $-44°$. These are four nearly identical orientations, two degrees either side of $\pm 45°$, which is one orientation. Their arithmetic mean is $0°$, the orientation that is *furthest* from all of them. The planner uses the circular mean instead: map each yaw to an angle on the full circle, average the unit vectors, and map back,

$$
\theta_k = \frac{2\pi}{T}\psi_k,\qquad \bar\psi = \frac{T}{2\pi}\,\operatorname{atan2}\!\Big(\tfrac1E\textstyle\sum_k \sin\theta_k,\; \tfrac1E\sum_k \cos\theta_k\Big),\qquad \sigma_\psi = \frac{T}{2\pi}\sqrt{-2\ln R}, \tag{2}
$$

with $R$ the length of the mean unit vector. For the four yaws above, (2) gives $\bar\psi = \pm 45°$ and a spread of 1.6°.

## A proposal that looks at the target

Uniform samples waste most of a population of 32: the clay is 140 mm wide and a tool head covers a small part of it. Half of the first population is therefore drawn from a **guided proposal** that uses the one thing known for certain, where the clay is too high.

- Position: a pixel $(i, j)$ drawn with probability proportional to the squared excess height, $\max(H_{ij} - H^\star_{ij}, 0)^2$, plus 4 mm of noise.
- Depth: the one that puts the bottom of the tool at the target height of that pixel, $d = z_{\text{top}} - H^\star_{ij}$, plus 1 mm of noise.
- Yaw and travel: uniform.

The other half is uniform over the bounds, so nothing is excluded.

The depth rule assumes that the clay keeps the whole stroke. This clay gives back about a third ([The Clay](/bench/clay/)), so the proposal starts too shallow, and three iterations of the search do not fully correct it. The consequence is measured in [The Oracle](/planning/oracle/#why-an-exact-model-does-not-reach-zero).

:::caution[The proposal is structure too]
"Press where it is too high, down to where it should be" is the geometric prior of [part V](/dynamics/prior/) turned into a sampling rule. It is given to every planner, including the ones labelled pure learning. What is compared across rollout models is therefore the *evaluation* of candidates, with the same candidates on the table. A planner with a uniform first population would need a larger budget to reach the same result; this was not measured.
:::

## When to stop

After the search, the best predicted cost is compared with the current true cost. The planner stops when the prediction does not improve on it by at least 0.02 mm:

$$
\text{stop if}\quad c\big(\hat H(a^\ast), H^\star\big) \;\ge\; c(H, H^\star) - 0.02\ \text{mm}. \tag{3}
$$

Each target also has a hard cap on the number of actions (3 for the single-action targets, 6 for the slab).

Rule (3) trusts the model. An exact model stops exactly when nothing helps. A model that predicts improvement where there is none could keep acting and make the shape worse, and a model that underestimates what an action achieves could stop early. In [part VI](/learned-mpc/results/) the graph networks are optimistic (the shape after an action is on average 0.1 mm further from the target than they predicted) and the prior is pessimistic by about as much. Neither turned into the failure: across 30 episodes per model, at most one executed action made the shape worse.

## What is counted

Every episode reports:

- the three shape distances between the final true shape and the target;
- the number of executed actions;
- the wall-clock **planning time** per decision (the search only, not the motion);
- the number of **oracle rollouts used for planning**. Executing the chosen action is one more simulated primitive per step; it is the experiment itself, not planning, and is not counted.

:::tip[Where in the code]
`control/mpc.py`: `cem`, `guided_proposal`, `wrap_action`, `_circular_stats`, `run_episode`, and the evaluators `OracleEval`, `ModelEval`, `HybridEval`.
Tests in `tests/test_mpc.py`: `test_cem_finds_a_known_optimum`, `test_circular_mean_across_the_wrap`, `test_yaw_wraps_instead_of_clipping`, `test_guided_proposal_goes_where_the_clay_is_too_high`.
:::
