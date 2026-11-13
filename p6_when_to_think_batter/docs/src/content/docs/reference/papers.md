---
title: "Papers"
description: "The references in the brief and how each is used."
sidebar:
  order: 4
---

- **Hansen, Su, Wang, *TD-MPC2* (ICLR 2024)** — [arXiv:2310.16828](https://arxiv.org/abs/2310.16828). The structure of the learned planner: an encoder to a latent state, latent dynamics trained by consistency with later encodings, a reward-like outcome head, and a policy prior that seeds sampling-based planning in latent space. Reduced here: no value function (the reward arrives once per pitch) and swing parameters as actions.
- **Williams et al., *MPPI*.** The sampling planner over the swing parameters in the latent space.
- **Sung, Kaelbling, Lozano-Pérez, *Learning When to Quit: Meta-Reasoning for Motion Planning* (IROS 2021)** — [arXiv:2103.04374](https://arxiv.org/abs/2103.04374). The framing of the trigger as meta-reasoning: a learned policy that decides whether more planning is worth its cost, with that cost in the reward.
- **Trimpe & D'Andrea, *Event-Based State Estimation with Variance-Based Triggering* (IEEE TAC 2014).** The variance trigger, which fires on covariances and never on data, so its schedule can be computed in advance.
- **Heemels, Johansson, Tabuada, *An Introduction to Event-triggered and Self-triggered Control* (CDC 2012).** The send-on-delta (drift) trigger and the periodic baseline it is measured against.
- **Jayanth et al., *Neural Inertial Odometry from Lie Events* (RSS 2025)** — [arXiv:2505.09780](https://arxiv.org/abs/2505.09780). Event-triggered sampling: process data when it changes, not on a clock.
- **Gehrig & Scaramuzza, *Low-latency automotive vision with event cameras* (Nature 2024)** — [doi:10.1038/s41586-024-07409-w](https://doi.org/10.1038/s41586-024-07409-w). The case for event cameras as latency-free change detectors, behind the event-camera stretch study.
