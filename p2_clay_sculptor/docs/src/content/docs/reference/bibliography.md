---
title: "Bibliography"
description: "The papers and documentation this project builds on, each with a line on where it is used."
sidebar:
  order: 4
---

## Manipulating plastic material

**Shi, H., Xu, H., Huang, Z., Li, Y., & Wu, J. (2022).** *RoboCraft: Learning to See, Simulate, and Shape Elasto-Plastic Objects with Graph Networks.* Robotics: Science and Systems. [arXiv:2205.02909](https://arxiv.org/abs/2205.02909)
The template for [the graph-network model](/dynamics/models/): particles, a graph network that predicts their displacement, and model-predictive control on a shape distance. Here the "particles" are the cage nodes of the flex, so they are the simulator's own degrees of freedom and no perception step is needed.

**Shi, H., Xu, H., Clarke, S., Li, Y., & Wu, J. (2023).** *RoboCook: Long-Horizon Elasto-Plastic Object Manipulation with Diverse Tools.* Conference on Robot Learning. [arXiv:2306.14447](https://arxiv.org/abs/2306.14447)
Several tools and a planner that picks among them. The [stretch target with three tools](/evaluation/stretch/) is the smallest version of that idea: search with every tool, execute the best prediction.

**Huang, Z., Hu, Y., Du, T., Zhou, S., Su, H., Tenenbaum, J. B., & Gan, C. (2021).** *PlasticineLab: A Soft-Body Manipulation Benchmark with Differentiable Physics.* International Conference on Learning Representations. [arXiv:2104.03311](https://arxiv.org/abs/2104.03311)
A benchmark for the same kind of task built on a differentiable material-point simulator. This project takes the other route: a non-differentiable finite-element simulator and sampling-based planning.

## Learned simulators

**Sanchez-Gonzalez, A., Godwin, J., Pfaff, T., Ying, R., Leskovec, J., & Battaglia, P. (2020).** *Learning to Simulate Complex Physics with Graph Networks.* International Conference on Machine Learning. [arXiv:2002.09405](https://arxiv.org/abs/2002.09405)
The encode-process-decode architecture used in [the models](/dynamics/models/), and the trick of adding noise to the inputs during training so that multi-step rollouts stay stable.

**Ronneberger, O., Fischer, P., & Brox, T. (2015).** *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
The architecture of the height-map baseline.

**Johannink, T., et al. (2019).** *Residual Reinforcement Learning for Robot Control.* International Conference on Robotics and Automation. [arXiv:1812.03201](https://arxiv.org/abs/1812.03201)
The general pattern behind [the structured model](/dynamics/prior/): keep what a model gets right and learn only the difference.

## Planning and uncertainty

**Rubinstein, R. Y. (1999).** *The Cross-Entropy Method for Combinatorial and Continuous Optimization.* Methodology and Computing in Applied Probability, 1, 127–190.
The optimiser of [the planner](/planning/cem/).

**Williams, G., Wagener, N., Goldfain, B., Drews, P., Rehg, J. M., Boots, B., & Theodorou, E. A. (2017).** *Information Theoretic MPC for Model-Based Reinforcement Learning.* International Conference on Robotics and Automation.
MPPI, the other standard sampling-based planner. It weights all samples instead of keeping an elite set; with five parameters and a population of 32 the two behave alike, and the cross-entropy method was used.

**Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018).** *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models.* NeurIPS. [arXiv:1805.12114](https://arxiv.org/abs/1805.12114)
Cross-entropy planning through an ensemble of learned dynamics models, the configuration of [part VII](/fallback/ensemble/) before the oracle is added.

**Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017).** *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)
Independently initialised and trained copies of one network; their disagreement is the uncertainty signal tested as a trigger for [asking the simulator](/fallback/ensemble/), and found [no better than chance](/fallback/results/) for that purpose.

## Mechanics

**Simo, J. C., & Hughes, T. J. R. (1998).** *Computational Inelasticity.* Springer.
Return mapping for $J_2$ plasticity, the algorithm behind the `yield` attribute described in [the clay page](/bench/clay/).

**Sederberg, T. W., & Parry, S. R. (1986).** *Free-Form Deformation of Solid Geometric Models.* SIGGRAPH.
A lattice of control points that deforms everything inside it by trilinear (or higher) interpolation: the idea behind MuJoCo's `dof="trilinear"` flex.

## Software

**MuJoCo 4.0 documentation**, [mujoco.readthedocs.io](https://mujoco.readthedocs.io): `flexcomp`, `elasticity` and its `yield` attribute, the `discrete` integrator, `mjSTATE_PLASTIC`, `mujoco.batch`, the `pid` actuator, `MjSpec.attach`. The example models `model/plastic/clay.xml` and `roller.xml` in the MuJoCo repository were the starting point for the material and contact parameters.

**MuJoCo Menagerie**, Franka Emika Panda model (`franka_emika_panda/panda_nohand.xml`).
