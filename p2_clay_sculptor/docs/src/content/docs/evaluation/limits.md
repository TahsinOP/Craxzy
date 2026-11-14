---
title: "Limits"
description: "A plain account of what the results do not show: properties of the simulated material, blind spots of the cost, the greedy planner, the small models, and the stretch goals that were not attempted."
sidebar:
  order: 3
---

:::note[Guiding question]
Which of the conclusions would survive a finer simulation, a different cost, a larger model or real clay, and which are properties of the particular choices made here?
:::

## The material is a coarse simulation of clay

**Spring-back is numerical.** A third of every stroke comes back, and the share barely moves when the yield strain is halved or doubled ([The Clay](/bench/clay/)). It comes from storing one plastic strain per cage cell, not from the material law, which on its own would keep more than 90 %. The finer discretisation that keeps more of the stroke could not be used: it creeps for seconds and inverts elements. So the central quantity the residual models learn here, how much of a press comes back, is a property of a $12 \times 12 \times 3$ cage. The method does not depend on where the spring-back comes from, but the numbers do.

**Sharp features do not exist.** Cage nodes are 11.7 mm apart. The 18 mm bars of the stamp leave a cross that is recognisable and soft, and no tool can leave a vertical wall. Targets were produced by demonstration partly for this reason: a hand-drawn target would not be reachable.

**Volume is lost.** 2.4 % after three presses, 5.1 % after three roller passes. Real clay is close to incompressible.

**Contact is frictionless and the roller does not spin.** With friction a tool would drag the surface along with it, and the response would depend on the direction of travel in a way it does not here.

**The bottom is glued down.** The block cannot slide, lift or be turned over, and it bulges above its base instead of spreading.

## The cost has blind spots

Every planner minimises the height-map RMSE. The map covers the top surface inside the outline of the block and nothing else.

**Side walls are invisible.** The block bulges sideways by up to 12 mm and the cost does not know.

**Sideways motion of material is invisible.** For the slab target the oracle brings the height-map error down to 16 % of its initial value while the earth mover's distance, which follows material points, only falls from 11.6 mm to 8.2 mm ([The Oracle](/planning/oracle/)). The surface is at the right height with the clay in the wrong place.

A cost on the full surface point cloud would see both. It was not used for planning because the U-Net baseline predicts height maps only, and the comparison needed one cost for every model.

## The planner is greedy

One action is planned at a time. The planner cannot accept a worse shape now for a better one later, and it cannot foresee that an action is irreversible.

The slab shows the price. The oracle's first pass is as deep as the tool is allowed to go, because that removes the most error at once, and it leaves the centre of the block 2 mm lower than the target. Clay that has been pressed too far cannot be raised. Even with an exact model, 13.9 % of the initial error is left on average, on targets for which zero is reachable. On the slab, the planner with the crudest model, the prior, does better than the planner with the exact one, because its error happens to make it cautious ([Planning with Learned Models](/learned-mpc/results/#the-slab-a-wrong-model-that-helps)).

The search budget is small as well, 96 candidates per decision, and with an exact model the single-action targets are still missed by 0.2 to 1.0 mm for that reason alone. Part of that miss is systematic: the guided proposal sets the depth as if the clay kept the whole stroke, so the first stamp of the X was too shallow in all ten seeds ([The Oracle](/planning/oracle/#why-an-exact-model-does-not-reach-zero)). Correcting the proposal for spring-back is a one-line change that was not made, so that all planners stay compared at the same search.

Planning two or more actions ahead is where a fast learned model should pay off most, since the simulator's cost grows with the horizon and the model's hardly does. That experiment was not run.

## The models are small and the data few

The graph networks have 75,000 parameters and were trained on 5,400 transitions for six epochs, on a CPU ([Three Models](/dynamics/models/)). The comparison between the pure and the structured model is at that one capacity and that one data size. A prior matters most when capacity and data are short, so the gap reported here is likely near its largest. How it closes with more of either was not measured.

The node models and the U-Net are not compared at equal capacity. The U-Net has six times the parameters and converged; the graph networks were stopped after six epochs, still improving, because they are slow to train without a GPU. That the U-Net predicts best ([How Good Are the Models?](/dynamics/results/#why-does-the-u-net-win)) is a statement about what a CPU can train in a quarter of an hour, not about graph networks in general.

All models see the shape only. The plastic strain inside the block is hidden from them, and it is part of the simulator's state. Two blocks of identical shape and different history respond differently, and no model of this form can tell them apart.

The random data cover the planner's states only loosely. Training episodes are six random actions deep; a slab episode applies six purposeful, overlapping roller passes, and ends in shapes that random actions rarely produce.

## What "the oracle" is

The oracle is the simulator that generated the training data and that executes the plans. It is exact here by construction. On a real system there is no such thing: the best available simulator is itself a model of the clay, and "ask the oracle" would mean "ask the expensive model", with its own error. The fallback rule of [part VII](/fallback/ensemble/) would still trade cost for accuracy, but it could no longer be judged against the truth.

There is also no perception. The planner reads the exact node positions after every action. RoboCraft and RoboCook reconstruct the shape from cameras, and a large part of their difficulty is there.

## Stretch goals

| goal from the brief | status |
|---|---|
| sculpt a letter | done: the letter T with the flat press ([A Letter, and a Choice of Tool](/evaluation/stretch/)) |
| multi-tool sequencing with a planner that chooses the tool | done in its simplest form: search with every tool, execute the best prediction ([A Letter, and a Choice of Tool](/evaluation/stretch/)). There is no learned high-level policy as in RoboCook. |
| plastic paper folding (`paper.xml`) | not attempted |
| two arms, one holding the clay | not attempted |

## Compute

Everything ran on one laptop: 8 cores, no GPU build of PyTorch, about 5 GB of free memory. The budgets (6,600 transitions, 75,000 parameters, 96 candidates, three training seeds per model, ten planner seeds per target) were chosen to fit. The closed-loop comparisons are made on 30 paired episodes per planner, with paired tests and 95 % intervals in [Planning with Learned Models](/learned-mpc/results/#which-differences-are-real) and [Does the Fallback Pay?](/fallback/results/). Differences of one or two percentage points of the initial error are at the edge of what these runs resolve. The stretch targets have three seeds of the learned planners and one of the oracle. Every script takes its budget as an argument ([Running It](/reference/running-it/)).
