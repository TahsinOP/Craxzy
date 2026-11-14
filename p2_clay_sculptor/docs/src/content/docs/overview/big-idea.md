---
title: "The Central Idea"
description: "Shaping clay is a modelling problem: the tool is known exactly and the material is not. What to derive, what to learn, and how to tell when the learned part should not be trusted."
sidebar:
  order: 1
---

:::note[Guiding question]
A robot that shapes clay must predict what a press will do before it presses. Part of that prediction is geometry anyone could write down, and part is material behaviour nobody can. Where exactly is the line, and what should stand on each side of it?
:::

## Shaping is prediction

Rigid objects forgive a poor model. Push a block and it slides; push it again and it slides back. Clay does not. Every action changes the object permanently, a press that goes too deep cannot be undone, and the shape after ten actions depends on all ten.

So a planner for clay needs a model it can ask "what will the shape be if I do this?", many times per decision, before anything moves. The whole project is about that model: how accurate it has to be, how fast, and what it should be made of.

Three facts about this clay make the question concrete. All three are measured in [part II](/bench/clay/) and [part V](/dynamics/results/).

- **A third of every stroke comes back.** A 10 mm press leaves a 6.7 mm dent.
- **Material moves sideways at least as much as it moves down.** Over the 600 test transitions, the free nodes of the block move 1.7 mm sideways (RMS) per action and 1.4 mm vertically.
- **Asking the simulator is slow.** One simulated action takes 0.2 to 0.4 s with all cores busy. A decision that compares 96 candidates takes 18 to 34 s.

## Where the line is

Some of what happens in a press has nothing to do with clay.

**Known exactly.** The shape of the tool head, and the path the arm takes it along. Together they give the volume the tool sweeps through. Whatever the material does, it is not inside that volume while the tool is there. This is geometry, and [one formula](/dynamics/prior/) covers the press, the roller and the stamp.

**Not known.** Where the displaced material goes. How much of the dent springs back. How the surface rises around the edge of the contact. How the response differs after the clay has already been worked. No formula is offered for these; they are what the simulator computes with 1,528 degrees of freedom and an implicit solver.

The line between the two is the line between a **prior** and a **residual**:

$$
\underbrace{\hat N'}_{\text{next shape}} \;=\; \underbrace{P(N, a)}_{\text{geometry: derived}} \;+\; \underbrace{\Delta_\theta(N, a)}_{\text{material: learned}} .
$$

The prior $P$ clips the clay to the swept volume and does nothing else. The network $\Delta_\theta$ learns the difference between that and what the simulator does. This is the **structured** model.

Two pure-learning models stand next to it, a graph network on the same three-dimensional state and a U-Net on a height map, and so does the prior alone, with no learning at all. The four are compared on one test set and inside one planner.

## Hybrid at three levels

The same split, a derived part where one is exact and a learned part where it has run out, appears three times.

| level | derived | learned | where |
|---|---|---|---|
| the **model** of one action | the swept volume of the tool | the residual: spring-back, sideways flow, bulge | [part V](/dynamics/prior/) |
| the **search** for an action | "press where the clay is too high, down to where it should be" proposes candidates | the model evaluates them | [part IV](/planning/cem/) |
| **trust** in the model | the simulator, exact and slow | an ensemble of five models, fast and sometimes wrong | [part VII](/fallback/ensemble/) |

The third level is the one the project is named for. A learned model is wrong somewhere, and it does not say where. Five copies of it, trained from different random starts, agree where the data pinned them down and scatter where it did not. When they scatter about the action the planner is about to take, the planner stops and asks the simulator about its best few candidates. Otherwise it goes ahead. The simulator is consulted only when the model is unsure. That was the plan; [Does the Fallback Pay?](/fallback/results/) shows how it fared, and why the ensemble's doubt turned out to be the wrong kind.

## One rule that everything obeys

**Every rollout model is scored in the same currency.** The planner minimises one cost, the root-mean-square difference between a predicted height map and the target height map. The simulator, the prior, both graph networks and the U-Net are all converted to a height map before they are compared with anything. Chamfer and earth mover's distances are reported too, but only for true simulator states, so no model can look good by predicting a flattering point cloud.

The search is also the same for all of them: 96 candidates per decision, the same proposals, the same stopping rule. When two planners end at different shapes, the model is the only thing that differed.

## What is simulated and what is not

Everything here happens in MuJoCo 4.0 on a CPU. There is no camera and no perception: the planner reads the true shape of the clay from the simulator after each action. There is no real clay either, and [The Clay](/bench/clay/) is explicit about which behaviours of the simulated block belong to the material law and which to its coarse discretisation. The claims of this site are about modelling and planning on *this* simulated material.

## What came out

The short version of [All Results](/evaluation/results/):

- **The prior helps a learned model, modestly.** It cuts the graph network's one-step error by 17 %, and in the planner leaves about 1.5 points less of the initial error. A U-Net with no prior at all still predicts twice as well.
- **Better predictions buy little in closed loop.** Every learned planner ends within 0.7 to 3.4 points of the simulator-driven planner (13.9 % of the initial error left), at 30 to 400 times less planning time. On the slab, the prior alone, the worst predictor, planned better than the simulator, because its error made it cautious.
- **The ensemble cannot tell when it is wrong, in the way that matters.** Its spread predicts its own error well, but triggering the simulator on it is worse than triggering at random. Asking the simulator to judge the model's best eight candidates does pay: half of the simulator planner's advantage at 1/12 of its simulations.
- **The search is the limit.** No planner reached any target, though all of them were reachable exactly. A one-step greedy planner on clay that cannot be un-pressed stops short, whatever its model.
