# P5: Holonomic Swarm

**From 4 to 256 robots: a swarm that learns many jobs on top of classical controllers.**

## Goal
A swarm of **holonomic (omnidirectional) robots** learns a *family* of cooperative tasks with one shared, decentralised policy. The policy is trained on small teams and must **scale** to much larger ones. Each robot runs a **classical controller library** (consensus, coverage, a CBF safety filter); the learned graph policy decides *how to combine and correct* these behaviours for each task. This isn't transport. The tasks are coverage, formation, sorting objects, gate passage, and emergent division of labour.

## Why it's a hybrid research question
- **Model-based structure that already works for swarms:**
  - Lloyd/Voronoi coverage control;
  - consensus/formation control;
  - **control-barrier-function (CBF) safety filters** for collision avoidance (with guarantees).
- **Learned:** a permutation-invariant **graph neural network** (GNN) policy that outputs (a) weights over the classical primitives and (b) a residual velocity, conditioned on a **task embedding**.
- **Questions:**
  1. Does the structured prior make learning **scale** (train with 8, deploy with 256)?
  2. Do roles emerge (scouts, pushers, guards) when tasks need them?
  3. How much does the CBF filter cost in performance versus what it buys in safety?
- This is multi-robot learning with physical interaction, and it scales, which matches your "multi-robot + hybrid" interest without repeating transport.

## MuJoCo 4.0 + Playground 1.0 features used
- **A holonomic robot model:** a cylinder body with two `slide` joints (x, y) and one `hinge` (yaw), driven by `velocity` actuators (or `pid`). It's cheap and contact-friendly.
- **`<replicate>`** and `MjSpec` to spawn N robots and M objects procedurally.
- **MuJoCo Warp** in Playground 1.0 for **thousands of parallel worlds × hundreds of robots** on GPU. No speculative contacts needed, so the whole project stays on the GPU path.
- `surfacevel` (conveyor zones) and `adhesion` (some robots can grab blocks) for the stretch tasks.
- A camera `rangefinder` sensor (new) for local sensing instead of global positions (stretch).

## Task family (same policy, task embedding as input)
1. **Coverage:** spread over a region weighted by a density map (the prior is Lloyd's algorithm).
2. **Formation:** form shapes such as letters (the prior is consensus with offsets); reassign targets with Hungarian or optimal transport.
3. **Sorting:** push coloured blocks into their matching zones; needs cooperation for heavy blocks.
4. **Gate passage:** a large swarm passes through a narrow gate without deadlock (the CBF prior alone tends to deadlock).
5. **Herding (crazy):** herd passive "sheep" agents (a Reynolds/boids model) into a pen.

## Milestones

**M0: Swarm arena on Warp**
- Build an arena with N robots (4–256) and blocks via `<replicate>`. Write a Playground environment with `impl="warp"`.
- *Accept when:* throughput in steps/s against N and the number of parallel worlds; a video of 256 robots running random velocities.

**M1: Classical library**
- Implement Lloyd coverage, consensus formation with optimal-transport target assignment, and a decentralised CBF-QP safety filter (each robot solves a small QP, batched in JAX or Warp).
- *Accept when:* each classical controller solves its own task; CBF collision rate is 0; deadlock cases at the gate are documented.

**M2: Pure-learning baseline**
- GNN policy (message passing over k-nearest neighbours), MAPPO-style training with a shared policy and centralised critic, trained on N = 8.
- *Accept when:* per-task performance, sample efficiency, and collision rate without a filter.

**M3: Hybrid policy**
- The GNN outputs primitive weights plus a residual velocity, then passes through the CBF filter.
- Ablations: no residual (weights only); no weights (residual only); with and without the CBF; task embedding vs separate policies.
- *Accept when:* a per-task table and a training-curve comparison against M2.

**M4: Scaling and emergence**
- Train on 8, evaluate on 16, 32, 64, 128 and 256 robots.
- Analyse emergent roles in sorting and herding: cluster robots by behaviour statistics.
- *Accept when:* scaling curves for pure vs hybrid, and a role-emergence analysis with videos.

**Stretch (crazy)**
- **Heterogeneous swarm:** some robots have `adhesion` grippers and some are fast scouts with a camera `rangefinder`.
- **Communication limits:** drop graph edges randomly and learn robustness.
- **Conveyor world:** zones with `surfacevel` that robots must exploit or fight.
- **Physical swarm path:** the robot model matches Robotarium-style or omni-wheel micro-robots.

## Evaluation
- Per-task success and quality (coverage cost, formation error, sorting accuracy, time through the gate), collisions, deadlocks, sample efficiency, generalisation to larger N, compute per robot.

## Papers
- Tolstaya et al., *Learning Decentralized Controllers for Robot Swarms with Graph Neural Networks* (CoRL 2019): https://arxiv.org/abs/1903.10527
- Wang, Ames, Egerstedt, *Safety Barrier Certificates for Collisions-Free Multirobot Systems* (T-RO 2017): https://scholar.google.com/scholar?q=Safety+Barrier+Certificates+for+Collisions-Free+Multirobot+Systems
- Zhang et al., *GCBF+* (T-RO 2025), learned graph CBFs that scale: https://arxiv.org/abs/2401.14554
- Cortés et al., *Coverage Control for Mobile Sensing Networks* (IEEE TRA 2004), Lloyd/Voronoi: https://scholar.google.com/scholar?q=Coverage+control+for+mobile+sensing+networks+Cortes
- Yu et al., *MAPPO* (NeurIPS 2022): https://arxiv.org/abs/2103.01955
- Rubenstein, Cornejo, Nagpal, *Programmable self-assembly in a thousand-robot swarm* (Science 2014): https://scholar.google.com/scholar?q=Programmable+self-assembly+in+a+thousand-robot+swarm
- Bettini et al., *VMAS: A Vectorized Multi-Agent Simulator* (2022), a useful comparison point: https://arxiv.org/abs/2207.03530
- Pickem et al., *The Robotarium* (ICRA 2017), holonomic-ish swarm hardware reference: https://scholar.google.com/scholar?q=The+Robotarium+A+remotely+accessible+swarm+robotics+research+testbed

## Pitfalls
- Contact counts grow with N. Use the Warp backend and set `nconmax`/`njmax` generously; profile throughput early.
- Batch the per-robot CBF-QPs: use a closed-form single-constraint solution or a batched QP, not a Python loop.
- Fix the graph construction (k-nearest neighbours with a radius) and keep it identical between training and scaling tests.
