---
title: "Does the Fallback Pay?"
description: "Milestone 4, the results: the ensemble's disagreement predicts its own error well, yet as a trigger for the simulator it does worse than firing at random. Asking the simulator pays; knowing when to ask was not achieved."
sidebar:
  order: 2
---

:::note[Guiding question]
The planner of [Asking the Simulator](/fallback/ensemble/) re-ranks its eight best candidates with MuJoCo when the ensemble is unsure. Does that make better shapes for fewer simulations than the alternatives, and is it the *signal* that makes the difference, or only the simulations?
:::

## The experiment

The ensemble is five structured models, trained like the one of [M3](/learned-mpc/results/) with five different seeds. The planner, the targets, the search and the ten planner seeds are those of [M3](/learned-mpc/results/), so every episode here can be paired with the same target and seed of every other planner. Eleven planners were run, 30 episodes each:

- **never**: the ensemble alone, no simulation (the pure learned planner of this milestone);
- **always**: re-rank the eight best candidates with the simulator at every decision;
- **disagreement**, three thresholds: re-rank when the spread $u(a^\ast)$ of the members' maps exceeds $\tau$, eq. (2);
- **misrank**, three thresholds: re-rank when the misranking probability $p_{\text{mis}}$ exceeds $\tau$, eq. (3);
- **random**, three rates: re-rank with probability 75 %, 50 % or 25 %, whatever the model says.

The thresholds are the 25 %, 50 % and 75 % quantiles of each signal over the decisions of *never* ($u$: 0.065, 0.122 and 0.163 mm; $p_{\text{mis}}$: 0.03, 0.24 and 0.40), so that each triggered planner fires about as often as one of the random ones. Every trigger draws its random number at every decision whether it uses it or not, so paired runs share the same random stream until their decisions first differ.

The ensemble plans at 4.6 s per decision (five models, about five times the single structured model). A re-ranking adds 8 simulated primitives, about 2 s.

## Does the ensemble know when it is wrong?

About its *predictions*, yes. On the 600 held-out transitions of [M2](/dynamics/results/), the spread $u$ ranks the true error of the ensemble mean with a Spearman correlation of **0.79**, and the quarter of transitions with the largest spread carries 40 % of the total error. The mean of five members is also a better predictor than one of them: 0.250 mm against 0.272 mm.

<figure class="plot">
<img class="fig-light" src="/fig/m4-calibration-light.svg" alt="Scatter of the ensemble spread against the true one-step error on 600 held-out transitions, with the mean error in eight equal-count bins rising from 0.14 to 0.48 mm">
<img class="fig-dark" src="/fig/m4-calibration-dark.svg" alt="Scatter of the ensemble spread against the true one-step error on 600 held-out transitions, with the mean error in eight equal-count bins rising from 0.14 to 0.48 mm">
<figcaption>Spread against true error, one point per held-out transition. Binned by spread, the mean error rises steadily, from 0.14 mm in the lowest eighth to 0.48 mm in the highest.</figcaption>
</figure>

By the usual test of an uncertainty estimate, this one is good. The rest of the page is about why that did not make it a good trigger.

## Shape error against simulator calls

| planner | fires at | error left | simulated primitives per episode | slab [mm] | dent [mm] | X [mm] |
|---|---|---|---|---|---|---|
| never | 0 % of decisions | 15.8 ± 1.3 % | 0 | 1.03 | 0.39 | 0.69 |
| always | 100 % | 14.8 ± 2.0 % | 30.4 | 1.05 | 0.32 | 0.67 |
| disagreement, $\tau$ = 0.065 mm | 76 % | 15.5 ± 2.0 % | 23.2 | 1.05 | 0.37 | 0.68 |
| disagreement, $\tau$ = 0.122 mm | 45 % | 16.0 ± 1.7 % | 13.9 | 1.07 | 0.39 | 0.70 |
| disagreement, $\tau$ = 0.163 mm | 24 % | 16.0 ± 1.4 % | 7.5 | 1.07 | 0.39 | 0.68 |
| misrank, $\tau$ = 0.03 | 74 % | 14.8 ± 1.9 % | 22.9 | 1.05 | 0.32 | 0.67 |
| misrank, $\tau$ = 0.24 | 47 % | 15.1 ± 1.3 % | 14.7 | 1.02 | 0.35 | 0.67 |
| misrank, $\tau$ = 0.40 | 27 % | 15.3 ± 1.4 % | 8.3 | 1.01 | 0.37 | 0.68 |
| random, 75 % | 85 % | 15.0 ± 1.9 % | 26.1 | 1.05 | 0.33 | 0.68 |
| random, 50 % | 51 % | 14.9 ± 1.6 % | 15.7 | 0.99 | 0.35 | 0.67 |
| random, 25 % | 29 % | 15.3 ± 1.6 % | 8.8 | 1.01 | 0.36 | 0.69 |
| oracle for every candidate ([M1](/planning/oracle/)) | | 13.9 ± 1.9 % | 371 | 1.05 | 0.27 | 0.65 |

The per-target columns are the final height-map RMSE, mean over the ten seeds. The ± is the spread over seeds, a few points for every planner, which hides every difference in this table. As in [M3](/learned-mpc/results/#which-differences-are-real), the comparisons that can be trusted are paired.

## Paired comparisons

Each episode minus the episode of *never* on the same target and seed (30 pairs, 95 % t-interval, negative = better than never):

<figure class="plot">
<img class="fig-light" src="/fig/m4-pareto-light.svg" alt="Paired difference to the pure learned planner against simulated primitives per episode. The random and misranking planners lie together between minus 0.5 and minus 1 point; the disagreement planners lie near zero or above it; always is at minus 1 point with 30 simulated primitives">
<img class="fig-dark" src="/fig/m4-pareto-dark.svg" alt="Paired difference to the pure learned planner against simulated primitives per episode. The random and misranking planners lie together between minus 0.5 and minus 1 point; the disagreement planners lie near zero or above it; always is at minus 1 point with 30 simulated primitives">
<figcaption>Gain over the pure learned planner against the simulations it cost. The misranking trigger lies on the random planner's curve; the disagreement trigger lies above both.</figcaption>
</figure>

| planner | difference to never | p (paired t) | simulated primitives per episode |
|---|---|---|---|
| always | −1.0 ± 0.9 points | 0.044 | 30.4 |
| disagreement 76 % | −0.3 ± 0.7 | 0.38 | 23.2 |
| disagreement 45 % | +0.3 ± 0.4 | 0.15 | 13.9 |
| disagreement 24 % | +0.2 ± 0.3 | 0.20 | 7.5 |
| misrank 74 % | −1.0 ± 0.8 | 0.014 | 22.9 |
| misrank 47 % | −0.7 ± 0.6 | 0.026 | 14.7 |
| misrank 27 % | −0.5 ± 0.4 | 0.040 | 8.3 |
| random 85 % | −0.8 ± 0.8 | 0.059 | 26.1 |
| random 51 % | −0.8 ± 0.5 | 0.002 | 15.7 |
| random 29 % | −0.5 ± 0.5 | 0.058 | 8.8 |

And the question the random planners were run for, each trigger against random at the same firing rate:

| comparison | difference | p (paired t) | p (Wilcoxon) |
|---|---|---|---|
| misrank 74 % − random 85 % | −0.22 ± 0.44 points | 0.3 | 0.4 |
| misrank 47 % − random 51 % | +0.13 ± 0.49 | 0.6 | 0.4 |
| misrank 27 % − random 29 % | +0.02 ± 0.32 | 0.9 | 0.6 |
| disagreement 76 % − random 85 % | +0.49 ± 0.66 | 0.1 | 0.09 |
| disagreement 45 % − random 51 % | **+1.09 ± 0.55** | **0.0004** | **0.00004** |
| disagreement 24 % − random 29 % | +0.65 ± 0.55 | 0.02 | 0.009 |

What the three tables say, with the Bonferroni caveat of [M3](/learned-mpc/results/#which-differences-are-real) in mind (the second table alone makes six comparisons; its threshold is 0.008):

- **Asking the simulator pays, a little.** Re-ranking at every decision lowers the error left by one point, at 30 simulated primitives per episode. The oracle planner gains 1.9 points with 371. The judge recovers half of the gap between the learned planner and the exact one at 8 % of the simulations, and at 6.6 s per decision against 24.4.
- **The disagreement trigger is worse than firing at random.** At the median threshold it gives up the whole gain of the random planner and a little more (+1.1 points, p = 0.0004, which survives the correction). At the other two thresholds it points the same way.
- **The misranking trigger is as good as random, and no better.** It gains what the random planner gains at the same rate (differences of −0.2 to +0.1 points, all p ≥ 0.3). Pooled over the three rates, misrank minus random is −0.02 points (p = 0.8).

So the fallback pays, but neither signal says *when* to ask. The rest of this page is about why.

## Why the disagreement picks the wrong moments

The disagreement trigger does not fire by decision. It fires **by target**:

| planner | slab | dent | X |
|---|---|---|---|
| disagreement 45 % | 70 % of decisions | **0 %** | 37 % |
| misrank 47 % | 57 % | 43 % | 32 % |
| random 51 % | 45 % | 57 % | 57 % |

At the median threshold, the dent never gets a re-ranking and the slab nearly always does. The spread is large on the slab (0.19 mm averaged over its decisions) and small on the dent (0.06 mm), because it grows with the size of the change being predicted. A roll that moves a whole band of clay leaves the members more room to differ than a single press. Over the decisions of *always*, the spread of the chosen action and its predicted cost have a Spearman correlation of 0.81: the spread mostly says *how much is still to be done*.

What a re-ranking gains does not follow that. Per decision of *always*, the mean gain was 0.023 mm on the slab, 0.026 mm on the dent and 0.014 mm on the X, and the dent is where re-ranking showed most in the final shapes (0.39 → 0.32 mm). The disagreement spends its budget on the slab, where every planner ends near 1.05 mm whatever it does, and none on the dent.

That is the distinction drawn in [Asking the Simulator](/fallback/ensemble/#a-second-signal-doubt-about-the-decision): $u$ is uncertainty about a *shape*. It is well calibrated for that (the figure above), and a re-ranking does not repair shapes, it repairs *orderings*. An error shared by all eight candidates, such as a roll predicted too deep wherever it is placed, raises $u$ and leaves the ranking right.

## Why the misranking probability is no better than chance

The misranking probability measures the right thing in principle, doubt about the order of the candidates. It still fails, for three measurable reasons.

**The gains are small and sparse.** At the 114 decisions of *always*, the re-ranking changed the true cost by 0.022 mm on average, and at 46 % of them by nothing: the ensemble's first choice was already the truly best of its eight. A signal has little to find.

**It barely correlates with the gain.** Each signal was recorded at every decision of *always* and compared with what the re-ranking gained there:

| signal | Spearman with the gain | share of the gain in the quarter of decisions with the largest signal |
|---|---|---|
| spread $u$ | −0.09 | 40 % |
| expected regret | +0.12 | 48 % |
| misranking probability $p_{\text{mis}}$ | +0.13 | 24 % |

With 114 decisions, a Spearman correlation of 0.13 is not distinguishable from zero (p = 0.17; for the regret, 0.20). The quarter of decisions where $p_{\text{mis}}$ is largest holds a quarter of the gain, exactly what a random quarter holds.

**The members share their errors.** Equation (3) treats the five members as five samples of the true dynamics. They are five networks of one architecture trained on one dataset, and where they are wrong, they tend to be wrong together. All five ranked the chosen candidate first at 32 % of the decisions of *never*, yet the simulator found a better one among the eight at about half of all decisions. Their agreement about the order is not evidence that the order is right.

The expected regret shows a trace of signal (its top quarter holds 48 % of the gain), but at 32 % of decisions it is exactly zero, so it cannot be thresholded below that, as noted in [Asking the Simulator](/fallback/ensemble/#a-second-signal-doubt-about-the-decision). It was recorded and analysed, not run as a trigger.

## What this means

The brief's third bet was that the ensemble can tell the planner when to ask the simulator. On this bench, it cannot. The ensemble's spread is a good error bar on a prediction, and that is a different thing from knowing when the decision is wrong.

What does work is cheaper than it sounds: re-ranking the model's few best candidates with the simulator, at every decision or at random ones, at 1/12 of the simulations of the oracle planner, for half of the oracle's advantage.

Things that were not tried and might change the answer:

- **More diverse members**: models of different architectures (a U-Net beside the graph networks), or members trained on bootstrap resamples, would share fewer errors.
- **A learned trigger**: the gain was recorded at 114 decisions with every signal beside it. With a few thousand, a small classifier could be trained to predict the gain directly, rather than assuming that disagreement means it.
- **A larger re-ranking budget** when the trigger fires, so that the simulator could also find candidates the model ranked low.

A first pass with three seeds (kept in `results/three_seeds/`) could not separate any of these planners from each other. With ten, only the paired comparisons can, and the differences are fractions of a point. These are small effects on a planner whose own search leaves 14 % of the error ([The Oracle](/planning/oracle/#why-an-exact-model-does-not-reach-zero)).

:::tip[Where in the code]
`control/mpc.py`: `HybridEval.signals` and `HybridEval.choose`. `scripts/m4_fallback.py` (with `--triggers` and `--resume`) → `results/m4_fallback.json`; `scripts/make_reports.py` → `results/M4.md`, with every table on this page, the head-to-head comparisons and the per-target firing rates.
:::
