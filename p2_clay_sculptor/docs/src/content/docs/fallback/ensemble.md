---
title: "Asking the Simulator"
description: "Milestone 4, the method: a deep ensemble of the structured model, two ways for it to say it is unsure (about the shape, about the decision), and a planner that re-ranks its best candidates with the simulator only then."
sidebar:
  order: 1
---

:::note[Guiding question]
A learned model is cheap and sometimes wrong. The simulator is right and expensive. Can the model itself say when it is about to be wrong, well enough that the simulator is needed only then?
:::

## Five models instead of one

The structured model of [part V](/dynamics/models/) is trained five times. The data, the architecture and the optimiser are identical; only the random seed differs, which changes the initial weights and the order of the mini-batches. This is a **deep ensemble** (Lakshminarayanan et al., 2017).

The five members were all pushed towards the same training targets, so where the data constrain the answer they agree. Where the data are thin, or the situation is unlike anything in the training set, nothing forces them to agree, and each lands wherever its own initialisation took it. Their spread is therefore a measure of how much the answer is the data's and how much it is an accident of training.

For a candidate action $a$ the members predict height maps $\hat H_1(a), \dots, \hat H_5(a)$. Two numbers are taken from them.

**The prediction** is the mean map, and its cost is what the search minimises:

$$
\bar H(a) = \frac{1}{5}\sum_{m=1}^{5} \hat H_m(a),\qquad c(a) = \operatorname{RMSE}\big(\bar H(a),\, H^\star\big). \tag{1}
$$

**The disagreement** is the spread of the members, pixel by pixel, summarised as one length:

$$
u(a) = \sqrt{\frac{1}{32^2}\sum_{i,j} \operatorname{Var}_m\big[\hat H_m(a)\big]_{ij}}\,. \tag{2}
$$

Both are in millimetres. $c$ says how good the action is believed to be; $u$ says how much that belief should be trusted.

## The fallback rule

The search runs exactly as in [The Planner](/planning/cem/), with (1) as its cost. It ends with 96 evaluated candidates, sorted by predicted cost. Then:

1. Compute the disagreement $u(a^\ast)$ of the best candidate.
2. If $u(a^\ast) \le \tau$, execute $a^\ast$. No simulation is used.
3. If $u(a^\ast) > \tau$, simulate the **eight best** candidates with MuJoCo, and execute the one whose true result is closest to the target. The stopping rule of the planner is then applied to that true cost, so the simulator also decides whether to stop.

A firing costs 8 simulated primitives. A decision of the pure oracle costs 96. The threshold $\tau$ sets how often the rule fires: $\tau = \infty$ never asks (the pure learned planner), $\tau = 0$ always asks.

The re-ranking can only choose among candidates the model already liked. If the model's eight favourites are all poor, the simulator picks the least poor. The oracle is used as a **judge**, not as a **search**.

## A second signal: doubt about the decision

The disagreement $u$ measures doubt about a *shape*. The planner does not execute a shape, it executes a *choice*: the candidate ranked first. A re-ranking can only change the outcome if the true order of the top candidates differs from the predicted one, and an error that all candidates share does not change the order. If every member predicts every dent 0.3 mm too shallow, $u$ may be large while the ranking, and therefore the decision, is right.

So a second signal looks at the ranking directly. Let $a_1, \dots, a_8$ be the eight best candidates by the ensemble cost (1), and $c_m(a_j) = \operatorname{RMSE}(\hat H_m(a_j), H^\star)$ the cost of $a_j$ according to member $m$. Treat the five members as five samples of the true dynamics. The difference

$$
\delta_m(a_j) = c_m(a_j) - c_m(a_1), \qquad j = 2, \dots, 8,
$$

is positive when member $m$ agrees that $a_1$ is better than $a_j$. With $g_j$ and $s_j$ its mean and standard deviation over the members, and a normal fit to the five samples, the probability that $a_j$ is in truth better than $a_1$ is $\Phi(-g_j / s_j)$, where $\Phi$ is the standard normal distribution function. The **misranking probability** of the decision is the largest of these:

$$
p_{\text{mis}} = \max_{j = 2, \dots, 8} \Phi\!\left(-\frac{g_j}{s_j}\right) \in [0, 1]. \tag{3}
$$

It is near 0 when every member puts $a_1$ well ahead of every alternative, $\tfrac12$ when some alternative is a coin flip, and above $\tfrac12$ when the members on average prefer an alternative (possible, because $a_1$ is ranked by the cost of the *mean* map, not by the mean of the costs). The planner with this signal fires when $p_{\text{mis}} > \tau$.

A simpler statistic, the members' expected regret $\frac{1}{5}\sum_m \big(c_m(a_1) - \min_j c_m(a_j)\big)$, is a poor trigger: it is exactly zero whenever all five members rank $a_1$ first, which was the case at 32 % of the decisions of the pure learned planner, so its lower quantiles collapse to zero and a threshold cannot separate those decisions. Equation (3) stays graded in that case.

Both signals are computed at every decision of every run and recorded, so that afterwards each can be tested against what a re-ranking actually gained.

### The control experiment

A fallback rule can help for two different reasons: because the simulator is consulted *at the right moments*, or simply because it is consulted *at all*. To separate them, every triggered planner is compared with a **random** one that fires with a fixed probability, at three rates. If a signal carries information, the planner it triggers should reach a lower error than the random planner at the same number of simulator calls.

The thresholds are set from the signal itself: the 25 %, 50 % and 75 % quantiles of $u$ and of $p_{\text{mis}}$ over all decisions of the pure learned planner, so that each triggered planner fires at roughly 75 %, 50 % and 25 % of decisions, the same rates as the three random planners.

:::tip[Where in the code]
`learn/models.py`: `Ensemble`. `control/mpc.py`: `ModelEval.maps` and `disagreement` (equations (1) and (2)), `HybridEval.signals` (equation (3) and the regret), `HybridEval.choose` (the rule, with triggers `disagreement`, `misrank`, `random`, `always`, `never`).
`scripts/m4_fallback.py` → `results/m4_fallback.json`, `results/M4.md`.
:::
