---
title: "How Good Are the Models?"
description: "Milestone 2: one-step and multi-step prediction error of the prior, the graph network, the structured model and the U-Net on 600 held-out transitions, their speed, and what happens when the clay is changed."
sidebar:
  order: 3
---

:::note[Guiding question]
Before any model is put inside the planner: how far off is its prediction of one primitive, how does the error grow when predictions are chained, and does the prior make the learned model better or only different?
:::

## The test

The last 100 of the 1,100 random episodes, 600 transitions, were never used in training or model selection ([Three Models](/dynamics/models/)). Every model predicts the shape after each of them, and the prediction is turned into a $32 \times 32$ height map by the same chain as the true shape. The number reported is the height-map RMSE $c(\hat H', H')$ of [Nodes, Vertices, Height Map](/shape/representation/), the quantity the planner minimises. Each learned model was trained with three seeds; the table gives mean ± standard deviation over those.

Two reference points frame the numbers. Predicting **no change** costs 1.74 mm: that is how much a random primitive changes the map. Predicting the **prior** costs what the swept-volume formula leaves out.

## One step

| model | all tools | press | roller | stamp | parameters | time per prediction |
|---|---|---|---|---|---|---|
| no change | 1.736 mm | 1.461 | 1.992 | 1.738 | | |
| prior only (classical) | 0.614 mm | 0.486 | 0.685 | 0.667 | 0 | 0.9 ms |
| GNN (pure learning) | 0.323 ± 0.005 mm | 0.234 | 0.381 | 0.349 | 74,595 | 8.5 ms |
| structured: prior + GNN (hybrid) | 0.267 ± 0.004 mm | 0.200 | 0.334 | 0.264 | 74,739 | 9.5 ms |
| U-Net on the height map (pure learning) | **0.148 ± 0.003 mm** | 0.122 | 0.169 | 0.152 | 470,401 | 0.5 ms |

Times are per candidate in a batch of 32 on eight threads, height-map conversion included. The simulator takes 183 ms per press or stamp and 356 ms per roll in the same setting ([Scene and Tools](/bench/scene/)), so the U-Net is about 340 times faster than the oracle and the graph networks about 20 times.

Three things can be read off the table.

**The prior alone does two thirds of the work.** One formula, no parameters, no data: it removes 65 % of the error of predicting no change. It is weakest on the roller, whose primitive moves the most material sideways, which the prior does not model at all ([The Geometric Prior](/dynamics/prior/)).

**The prior makes the graph network better, by 17 %.** The structured model and the GNN differ in exactly one input, the prior's displacement, and are otherwise identical ([Three Models](/dynamics/models/)). The gap, 0.323 against 0.267 mm, is twelve times the spread across seeds, and it holds for every tool. The largest gain is on the stamp (24 %), the tool with the most complex shape: the pure network has to discover the cross from data, the structured one is handed it.

**The U-Net is better than both, by a factor of two.** It sees less (no interior, no side walls) and is told nothing about the tools. Why it wins is discussed [below](#why-does-the-u-net-win).

## Several steps

A planner that looks ahead chains predictions. The models were run open loop over the six primitives of each test episode, each prediction fed back as the next input, and compared with the true shape after every step.

| primitives predicted open loop | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| no change | 2.53 | 3.56 | 4.25 | 4.90 | 5.40 | 5.78 |
| prior only | 0.80 | 1.12 | 1.34 | 1.55 | 1.71 | 1.80 |
| GNN | 0.39 | 0.50 | 0.56 | 0.59 | 0.61 | 0.63 |
| structured | 0.32 | 0.41 | 0.46 | 0.49 | 0.51 | 0.54 |
| U-Net | 0.15 | 0.20 | 0.24 | 0.26 | 0.28 | 0.30 |

(The first column differs from the one-step table because here every step starts from the true *first* state of an episode, the untouched block, where primitives remove more material.)

The errors grow, and they grow more slowly than the error of predicting no change: after six steps every learned model is still below a tenth of it. None of the models diverges. The ranking does not change with the horizon, and the ratio between structured and GNN stays between 0.82 and 0.86 throughout: the prior's advantage neither compounds nor wears off.

The prior on its own accumulates error faster than the learned models, by a factor of 2.3 from step 1 to step 6 against 1.6 for the GNN. What it leaves out at every step (the bulge, the sideways flow, the spring-back) is never corrected, and the next press acts on a shape that is already wrong.

<figure class="plot">
<img class="fig-light" src="/fig/m2-errors-light.svg" alt="Bar chart of one-step height-map error per model and tool, and line chart of the error against the number of chained predictions">
<img class="fig-dark" src="/fig/m2-errors-dark.svg" alt="Bar chart of one-step height-map error per model and tool, and line chart of the error against the number of chained predictions">
<figcaption>Left: one-step error per tool. Right: error against the number of primitives predicted open loop. Error bars are the spread over three training seeds.</figcaption>
</figure>

<figure class="plot">
<img class="fig-light" src="/fig/m2-examples-light.svg" alt="Height maps of four held-out transitions: before, true after, and the prediction of each model">
<img class="fig-dark" src="/fig/m2-examples-dark.svg" alt="Height maps of four held-out transitions: before, true after, and the prediction of each model">
<figcaption>Four held-out transitions, one row each. The prior draws the tool's footprint at full depth with sharp edges and no bulge; the learned models soften the edge and raise the surroundings, as the clay does.</figcaption>
</figure>

## Why the node error is larger than the map error

The structured model's error is 0.27 mm on the map but 0.59 mm on the node positions and 0.69 mm on the top vertices. These are not in conflict. The map is an average over 4 mm pixels of a surface interpolated from the top vertices, so a vertex that is predicted 1 mm too far *sideways* on a flat region changes the map hardly at all. Most of the node error is lateral displacement of this kind: the clay flows sideways by 1.7 mm RMS per primitive, against 1.4 mm vertically.

The planner does not care where on the surface a piece of clay sits as long as the surface is at the right height. The height map is the right yardstick for a model that serves this planner, and it is blind in the same way as the planner's cost ([The Oracle](/planning/oracle/#what-the-height-map-does-not-see)).

## The projection step of the prior

The prior's last step turns a displacement of the surface into a displacement of the cage nodes ([The Geometric Prior](/dynamics/prior/)). Two versions were compared on the 600 test transitions:

| projection | one-step map RMSE |
|---|---|
| lumped: each node takes the displacement of the top vertices that depend on it, weighted by their interpolation weight | 0.614 mm |
| least squares: the node displacement whose interpolated top surface matches the target displacement best | 0.801 mm |
| no change | 1.736 mm |

The least-squares projection fits the commanded surface exactly where the tool is, and pays for it with oscillations around the footprint: on a cage of 12 mm cells, a sharp step cannot be represented, and the least-squares solution overshoots on both sides of it (Gibbs-like ringing). The lumped projection gives each node a weighted mean of the displacements of the vertices it influences, with non-negative weights that sum to one. A convex combination of downward displacements is a downward displacement no larger than the largest of them, and the interpolated surface is again a convex combination of node displacements: the lumped prior can round a dent off, but it can never raise the clay or push it below the tool. It is the default.

## Different clay

The models were trained on one material. Four further sets of 48 episodes (288 transitions each) were generated with one material parameter changed, and the models were tested on them unchanged.

| clay | no change | prior | GNN | structured | U-Net |
|---|---|---|---|---|---|
| training material | 1.736 | 0.614 | 0.323 | 0.267 | 0.148 |
| Young's modulus halved, 75 kPa | 1.686 | 0.569 | 0.322 | 0.269 | 0.178 |
| Young's modulus doubled, 300 kPa | 1.655 | 0.609 | 0.323 | 0.269 | 0.168 |
| yield strain halved, 0.01 | 1.764 | 0.738 | 0.385 | 0.363 | 0.317 |
| yield strain doubled, 0.04 | 1.435 | 0.696 | 0.534 | 0.500 | 0.451 |

<figure class="plot">
<img class="fig-light" src="/fig/m2-shift-light.svg" alt="Grouped bar chart of one-step error for each model on the training material and four changed materials">
<img class="fig-dark" src="/fig/m2-shift-dark.svg" alt="Grouped bar chart of one-step error for each model on the training material and four changed materials">
<figcaption>One-step error on the training material and on four changed materials, models unchanged.</figcaption>
</figure>

**A stiffer or softer clay makes almost no difference**, to any model. This has a simple reason. The arm is position-controlled and tracks its path to within a millimetre at the end of every stroke ([Scene and Tools](/bench/scene/)), so the tool imposes a *displacement*, not a force. The yield criterion of the clay is written in strain ([The Clay](/bench/clay/)). In a quasi-static deformation driven by prescribed displacements, the strain field is fixed by the kinematics, and $E$ only scales the stress needed to produce it. Changing $E$ changes the force on the arm, not the shape of the clay. What remains is gravity and inertia, whose effect relative to the elastic stresses is of order $\rho g L / E = 1500 \cdot 9.81 \cdot 0.04 / 1.5 \times 10^{5} \approx 4 \times 10^{-3}$ for this block.

**A different yield strain is a different material**, and every model gets worse. The elastic strain the clay can store before yielding is exactly what it gives back when the tool leaves, so the yield strain sets the spring-back. Doubling it makes the clay spring back more, and the no-change error drops from 1.74 to 1.44 mm because less deformation stays.

The U-Net, the best model on the training material, loses the most: its error grows by a factor of 2.1 when the yield strain is halved and 3.0 when it is doubled. For the structured model the factors are 1.4 and 1.9, for the GNN 1.2 and 1.7, for the prior alone 1.2 and 1.1. The more a model knows about geometry and the less it has fitted to the material, the less it loses. The prior did not make the graph network more robust, though: in relative terms the GNN degrades slightly less than the structured model. In absolute terms, the U-Net is still the most accurate on every material tested.

## Why does the U-Net win?

This was not the expected outcome, and it should be read carefully, because the comparison is not one of equals.

1. **It is trained on the quantity it is tested on.** The node models minimise the node error, of which most is lateral motion that the map cannot see (above). Their capacity goes into a problem that the evaluation, and the planner, do not reward. The U-Net spends all of its capacity on the map.
2. **It is six times larger and was trained to convergence.** Its validation error changed by less than 3 % over the last five of 30 epochs. The graph networks were trained for 6 epochs and were still improving in the last one (0.329 → 0.315 mm for GNN seed 0), with the one-cycle schedule ending the run. At equal wall-clock time (12 minutes for the U-Net, 16 to 18 for the graph networks), the convolution on a $32 \times 32$ image simply gets far more learning done on a CPU than message passing over 3,510 edges ([Sized for a CPU](/dynamics/models/#sized-for-a-cpu)).
3. **Its problem is smaller.** The height map has 1,024 numbers; the cage has 2,028. And within a single primitive, the map's change is almost local and translation-invariant, which is the inductive bias of a convolution.

What can be concluded: *at the size a CPU could train*, the image model is more accurate and much faster than the node models, and adding the prior closes about a third of the gap between the GNN and the U-Net. What cannot be concluded: that a graph network with a prior is in general worse than a U-Net. A larger graph network, or a node model whose loss weighted the top surface, could change the order, and neither was tried.

The more useful question for this project is whether any of these differences matter once the model is inside the planner, which is the subject of [Planning with Learned Models](/learned-mpc/results/).

:::tip[Where in the code]
`learn/train.py`: `one_step_metrics`, `multi_step_metrics`. `scripts/m2_eval.py` → `results/m2_eval.json`, `results/M2.md`; the shifted materials come from `scripts/m2_dataset.py --set clay.young=…` and `--set clay.yield_strain=…`.
:::
