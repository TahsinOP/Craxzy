---
title: "Planning with Learned Models"
description: "Milestone 3: the planner of milestone 1 with each model in place of the simulator. Shape error against planning time over 30 episodes per model, paired comparisons, and a slab target on which a wrong model beats the exact one."
sidebar:
  order: 1
---

:::note[Guiding question]
The U-Net predicts one primitive twice as accurately as the structured model, and the structured model 17 % more accurately than the pure graph network. Inside a planner that sees the true shape after every action, how much of that is left?
:::

## The experiment

Everything of [The Oracle](/planning/oracle/) is kept: the three targets, the cross-entropy search with 96 candidates and the guided proposal, the stopping rule, the cap on actions. Only the rollout model changes. Each candidate is scored by the model's predicted height map instead of a simulation, and the chosen action is then executed **in the simulator**, so every shape in this page is a true one, and the planner observes it before the next decision.

Ten planner seeds per target and model, 30 episodes per model. The three trained seeds of each model are cycled: planner seed $i$ plans with the model trained with seed $i \bmod 3$. The oracle row is the ten-seed run of milestone 1.

## Results

| rollout model | slab | dent | X | initial error left, all targets | planning time per decision | relative to the oracle |
|---|---|---|---|---|---|---|
| oracle (MuJoCo) | 1.05 ± 0.13 mm | 0.27 ± 0.05 mm | 0.65 ± 0.20 mm | 13.9 ± 1.9 % | 24.4 s | 1× |
| prior only | **0.82 ± 0.07 mm** | 0.43 ± 0.05 mm | 0.64 ± 0.17 mm | 15.0 ± 1.8 % | 0.10 s | 244× faster |
| GNN | 1.18 ± 0.20 mm | 0.40 ± 0.11 mm | 0.78 ± 0.21 mm | 17.3 ± 2.3 % | 0.79 s | 31× faster |
| structured | 1.13 ± 0.09 mm | 0.36 ± 0.06 mm | 0.69 ± 0.18 mm | 15.8 ± 1.9 % | 0.76 s | 32× faster |
| U-Net | 1.07 ± 0.08 mm | 0.31 ± 0.07 mm | 0.67 ± 0.21 mm | 14.6 ± 2.0 % | 0.06 s | 400× faster |

Untouched block: 6.97, 2.30 and 4.39 mm. "Initial error left" averages the three targets within a seed; the ± is the spread over the ten seeds.

<figure class="plot">
<img class="fig-light" src="/fig/m3-tradeoff-light.svg" alt="Share of the initial error left against planning time per decision, one point per rollout model, logarithmic time axis">
<img class="fig-dark" src="/fig/m3-tradeoff-dark.svg" alt="Share of the initial error left against planning time per decision, one point per rollout model, logarithmic time axis">
<figcaption>Shape error against planning time. Every learned planner sits within a few percentage points of the oracle at 30 to 400 times less time per decision.</figcaption>
</figure>

<figure class="plot">
<img class="fig-light" src="/fig/m3-shapes-light.svg" alt="Height maps reached on the three targets by the planner with each rollout model, first seed">
<img class="fig-dark" src="/fig/m3-shapes-dark.svg" alt="Height maps reached on the three targets by the planner with each rollout model, first seed">
<figcaption>Shapes reached in the first seed, one row per target. The differences between models are smaller than the differences between seeds of the same model.</figcaption>
</figure>

## Which differences are real?

The spreads in the table overlap almost everywhere, but they overstate the uncertainty of a *comparison*. Two planners run with the same seed on the same target start from the same state and draw the same first random numbers, so much of the seed-to-seed variation is shared. The right test pairs them. For each pair of models, the difference of the error left was taken episode by episode (30 pairs), with a 95 % t-interval:

| comparison | difference in initial error left | p (paired t-test) |
|---|---|---|
| structured − GNN | **−1.5 ± 1.3 points** | 0.029 |
| U-Net − structured | −1.1 ± 1.3 points | 0.10 |
| prior − structured | −0.7 ± 1.6 points | 0.38 |
| oracle − U-Net | −0.8 ± 0.9 points | 0.10 |
| oracle − structured | −1.9 ± 1.4 points | 0.010 |
| oracle − GNN | −3.4 ± 1.8 points | < 0.001 |
| U-Net − GNN | −2.6 ± 1.5 points | 0.002 |

Ten pairwise comparisons were made among five planners, so a p-value near 0.03 is suggestive, not conclusive; with a Bonferroni correction (threshold 0.005), only "U-Net and oracle beat the GNN" survive. A Wilcoxon signed-rank test, which assumes nothing about the shape of the distribution, gives p = 0.06, 0.06, 0.37, 0.14, 0.004, 0.001 and 0.003 for the rows in order: the same picture, slightly weaker for the first row.

What this supports:

- **The prior still helps the graph network in closed loop**, by about 1.5 points of the initial error. It is the same direction as the 17 % one-step gain of [M2](/dynamics/results/), and much smaller.
- **The U-Net planner is not separated from the oracle**, at 1/400 of the time per decision.
- **The prior alone is not separated from the structured model**, overall. Per target, it is far from equal; see the slab, below.

## Accuracy shrinks in closed loop

In [M2](/dynamics/results/), the structured model's one-step error is 1.8 times the U-Net's, and the GNN's is 2.2 times. In the planner, those become differences of 1.1 and 2.6 percentage points of the initial error. Three things absorb most of a model's error:

1. **The loop is closed.** Every decision starts from the true shape. An error in one prediction costs one imperfect action, and the next decision sees its result. Errors never compound as they do in the open-loop rollouts of M2.
2. **The planner needs the ranking, not the values.** An error that is the same for all candidates, for example a dent predicted 0.3 mm too shallow wherever it is made, does not change which candidate is best. Only the part of the error that differs between candidates can mislead the search.
3. **The search is the larger error.** With the exact model, 13.9 % of the initial error is left ([The Oracle](/planning/oracle/#why-an-exact-model-does-not-reach-zero)). Every learned planner starts from that floor, and the model can only add to it.

## The slab: a wrong model that helps

On the slab, the prior alone leaves 11.8 % of the initial error. That is better than every learned model and better than the simulator itself (15.1 %). Paired with the structured model on the same seeds, the prior is 4.3 points better (p < 0.001). On the dent it is 3.4 points worse (p = 0.006).

The prior is not a better predictor of a roll. Its one-step error is 2.3 times the structured model's. It is a different *kind* of wrong. The prior has no spring-back: it believes that the clay ends where the tool bottom went. So, to bring the surface down to the target, it asks for less stroke than is needed. The other planners, which know about spring-back, ask for the full stroke:

| rollout model | depth of the first roll | mean depth of all rolls | mean signed error of the final map | mean depth by which the final map is *too low* |
|---|---|---|---|---|
| prior only | 8.8 mm | 7.6 mm | −0.13 mm | 0.38 mm |
| structured | 12.0 mm | 8.8 mm | −0.74 mm | 0.83 mm |
| U-Net | 12.9 mm | 8.7 mm | −0.60 mm | 0.73 mm |
| oracle | 13.0 mm | 8.8 mm | −0.63 mm | 0.74 mm |

(Mean over the ten seeds; the last two columns are averaged over the map, the last counting only pixels below the target.)

A greedy planner on the slab is tempted to dig too deep, because one deep pass removes the most error now and the cost of having gone too far shows only later ([The Oracle](/planning/oracle/#why-an-exact-model-does-not-reach-zero)). Clay that is too low cannot be raised. The accurate models dig the first pass 12 to 13 mm deep and end with the slab 0.6 to 0.7 mm too low on average. The prior underestimates what each pass does, digs 8.8 mm, and ends 0.13 mm too low. Its error makes it cautious, and caution is what a one-step planner lacks on a target where actions cannot be undone.

The same bias costs it on the dent, where the target is a single 11 mm press: the prior's first press is 8.8 mm deep on average (structured 9.4 mm, oracle 9.8 mm), and three presses at that depth do not add up to the one deep one.

This is the most instructive result of the milestone, and it is not an argument for inaccurate models. It says that with a greedy planner, *which way* a model is wrong can matter more than *how much*. A planner that looked ahead, or a cost that penalised going too low more than staying too high, would make the comparison one of accuracy again.

<figure class="sim">
<video src="/video/structured.mp4" loop muted playsinline controls preload="metadata"></video>
<figcaption>The structured planner, first seed of each target, replayed in real time. Each decision took under a second of planning.</figcaption>
</figure>

## The other metrics

Chamfer and EMD on the final shapes follow the height-map error on the dent and the X, and differ little between models: 0.29 to 0.35 mm (Chamfer) on the dent, 0.60 to 0.73 mm on the X. On the slab, every planner, the oracle included, has an EMD of 6 to 8 mm, against 11.6 mm for the untouched block. That is the blind spot of the height-map cost described in [The Oracle](/planning/oracle/#what-the-height-map-does-not-see), and no rollout model changes it, because all of them plan against the same cost.

:::tip[Where in the code]
`control/mpc.py`: `ModelEval` (one model or an ensemble as the rollout model). `scripts/m3_mpc.py` → `results/m3_mpc.json`, `results/M3.md` (with the paired comparisons); `scripts/docs_videos.py` → `structured.mp4`.
:::
