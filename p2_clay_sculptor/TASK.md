# P2: Clay Sculptor

**Press plastic clay into a target shape with learned dynamics and MPC.**

## Goal
A Panda arm with interchangeable tools (flat press, cylinder roller, cross-shaped stamp) deforms a **plastic 3D solid** (MuJoCo 4.0 `<elasticity yield=...>`) to match a target shape: flatten, make a dent pattern, roll into a slab. Clay has no simple closed-form model, so learn its dynamics from MuJoCo rollouts and plan with sampling-based MPC.

## Why it's a hybrid research question
- The tool and arm kinematics are known exactly (structure). The material response isn't (learn it).
- **Hybrid comparison:** (a) a pure learned particle/graph dynamics model; (b) a structured model, i.e. a geometric "tool sweeps out volume" prior plus a learned residual for spring-back and bulge; (c) MuJoCo itself used as the model, in-the-loop MPC (the expensive oracle).
- Plasticity is new in MuJoCo 4.0. Very few people have built manipulation benchmarks on it yet.

## MuJoCo 4.0 features used
- 3D solid flex with `trilinear` or `quadratic` cages, or `<flexcomp type="mesh">`, plus `<elasticity young=... poisson=... yield=...>` (start from `model/plastic/clay.xml` and `roller.xml` in the MuJoCo repo).
- Implicit `integrator="discrete"` for stability at larger timesteps.
- The plastic state is stored in `mjData.plastic`; save and restore it with `mjSTATE_PLASTIC` for MPC rollouts (copy state → simulate candidates → restore).
- Studio hot-reload for tuning material parameters.

## Milestones

**M0: Clay bench**
- Load `clay.xml`, attach a Panda tool via `MjSpec`/`<attach>`, script presses, and confirm permanent deformation (dents stay after the tool leaves).
- Define the shape representation: vertex positions of the flex, plus a height map over the table plane.
- *Accept when:* scripted press, roll and stamp sequences produce repeatable shapes, and state save/restore with `mjSTATE_PLASTIC` reproduces rollouts bit-for-bit.

**M1: Oracle MPC**
- Use CEM or MPPI over tool action primitives (press: x, y, depth, angle; roll: start, end, depth), with MuJoCo itself as the rollout model.
- Cost: Chamfer distance (or height-map L2) to the target.
- *Accept when:* you reach 3 targets (flat slab, single dent, "X" stamp pattern) with reported Chamfer error and wall-clock time. This is the upper bound.

**M2: Learned dynamics**
- Dataset of (shape, action) → next shape from random primitives (CPU, `mujoco.batch`).
- Models:
  - (a) a GNN over subsampled flex vertices (RoboCraft-style);
  - (b) a structured model: geometric swept-volume prior plus a learned per-vertex residual;
  - (c) a 2D height-map U-Net baseline.
- *Accept when:* you have one-step and 5-step prediction error for all three, plus speed.

**M3: Learned-model MPC**
- Replace the oracle with each learned model.
- *Accept when:* you have a table of shape error against planning time for oracle, GNN, structured and height-map models.

**M4: Know when the model is wrong**
- Train an ensemble (or use prediction variance). When disagreement is high, fall back to an oracle rollout for that step ("think harder only when unsure").
- *Accept when:* a Pareto plot of shape error against oracle calls shows the hybrid beats both pure options.

**Stretch (crazy)**
- Multi-tool sequencing, with a high-level planner choosing the tool (RoboCook-style).
- Sculpt a letter ("T" for Tahsin).
- Combine with plastic **paper folding** (`paper.xml`), i.e. origami creases.
- Two arms: one holds the clay, one shapes it.

## Evaluation
- Chamfer and Earth Mover's distance to the target; number of actions; planning time; model error across horizons; robustness to different `young` and `yield` values (domain shift).

## Papers
- Shi et al., *RoboCraft: Learning to See, Simulate, and Shape Elasto-Plastic Objects with Graph Networks* (RSS 2022): https://arxiv.org/abs/2205.02909
- Shi et al., *RoboCook: Long-Horizon Elasto-Plastic Object Manipulation with Diverse Tools* (CoRL 2023): https://arxiv.org/abs/2306.14447
- Huang et al., *PlasticineLab: A Soft-Body Manipulation Benchmark with Differentiable Physics* (ICLR 2021): https://arxiv.org/abs/2104.03311
- Sanchez-Gonzalez et al., *Learning to Simulate Complex Physics with Graph Networks* (ICML 2020): https://arxiv.org/abs/2002.09405
- Williams et al., *Information-Theoretic MPC (MPPI)*: https://scholar.google.com/scholar?q=Information+theoretic+MPC+for+model-based+reinforcement+learning+Williams
- Lakshminarayanan et al., *Deep Ensembles* (NeurIPS 2017): https://arxiv.org/abs/1612.01474
- MuJoCo docs: `<flexcomp>`, `<elasticity>` with `yield`, the `discrete` integrator, `mjSTATE_PLASTIC`.

## Pitfalls
- Plastic solids are expensive. Keep the vertex count low (a coarse cage), and profile early.
- Speculative contacts don't cover flexes yet. Keep tool speeds moderate, or use small timesteps for tool–clay contact.
- Make sure save/restore captures *all* state (`mjSTATE_PLASTIC` included); otherwise MPC rollouts silently diverge.
