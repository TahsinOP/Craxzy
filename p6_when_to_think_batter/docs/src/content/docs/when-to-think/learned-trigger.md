---
title: "The Learned Trigger"
description: "PPO on CPU with mujoco.batch, one policy for every price of compute."
sidebar:
  order: 2
---

:::note[Guiding question]
Can a policy learn when the expensive planner is worth calling, from the outcome alone?
:::

## The decision process

At every step at which a plan exists and the ball is more than 50 ms from the plate, the policy chooses between executing the cached swing and calling the planner. A call costs $\lambda$; a hit landing $d$ from the target pays $\exp(-d^2/2\sigma^2)$ at the end of the pitch. There is no discount (a pitch is under a second long), and advantages use GAE with $\lambda_\text{GAE}$ = 0.95.

**One policy for every price.** $\lambda$ is drawn log-uniformly between 0.002 and 0.2 for every pitch, and $\log\lambda$ is an input. Sweeping $\lambda$ at evaluation traces the policy's whole trade-off curve from one training run.

## Training on a CPU

Speculative contacts are CPU only, so the rollouts run on `mujoco.batch` worlds spread over a pool of 24 processes, 8 worlds each: 150 iterations of 192 pitches, about 7 minutes. The policy is small and the problem has short episodes, which is what makes a CPU budget enough, as the brief anticipated.

<figure class="plot">
<img class="fig-light" src="/fig/ppo-light.svg" alt="PPO training curves: return, hit quality and planner calls per pitch">
<img class="fig-dark" src="/fig/ppo-dark.svg" alt="PPO training curves: return, hit quality and planner calls per pitch">
<figcaption>Training with λ drawn per pitch. The policy starts by replanning with probability 0.18 per step (about 10 calls) and settles near 4 to 5 calls per pitch on the λ mix, keeping the quality above 0.9.</figcaption>
</figure>

At evaluation the policy is sampled, not thresholded.

A second policy was trained with half of its pitches carrying a surprise (Part VII). It behaves almost identically on surprised pitches; the one that never saw a surprise already reacts to them, because a surprise moves the innovation and the drift it reads.

:::tip[Where in the code]
`learn/ppo.py` (`features`, `TriggerPolicy`, `collect`, `update`), `eval/sweeps.py` (`train_ppo`), `eval/runner.py` (`collect_parallel`).
:::
