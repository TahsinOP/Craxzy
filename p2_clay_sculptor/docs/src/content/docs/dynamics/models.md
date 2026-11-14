---
title: "Three Models"
description: "A graph network on the cage nodes, the same network on top of the geometric prior, and a U-Net on the height map: what each is given, what each must work out for itself, and how they are trained on a CPU."
sidebar:
  order: 2
---

:::note[Guiding question]
If a pure-learning model and a hybrid model are to be compared fairly, they must be told the same things about the action. What exactly is "the same", when one of them is also handed a formula for the tool?
:::

## The task of a model

A dynamics model replaces one simulated primitive:

$$
\hat s' = f_\theta(s, \text{tool}, a),
$$

where $s$ is a shape and $a$ the five parameters of a [primitive](/bench/primitives/). The simulator needs 180 to 340 implicit steps of a 1,528-dimensional system for this. The model does it in one forward pass.

The model sees the *shape* only. It does not see the plastic strain inside the clay, which the simulator carries in its state ([State, Save and Restore](/bench/state/)). Two blocks with the same shape and different histories look identical to it. That hidden state puts a floor under the error of any model of this form, however large.

## What every model is told about the action

All three learned models receive the action as the same six **geometric features** per point, a point being a cage node for the node models and a pixel for the U-Net. With $(l_x, l_y)$ the point's coordinates in the tool frame ([The Geometric Prior](/dynamics/prior/)):

| feature | meaning |
|---|---|
| $l_x / 50\ \text{mm}$ | position along the tool's $x$ axis (the roller axis) |
| $l_y / 50\ \text{mm}$ | position along the travel direction |
| $\operatorname{sign}(l_y)\max(\lvert l_y\rvert - \ell/2,\, 0) / 50\ \text{mm}$ | distance beyond the ends of the travel |
| $\ell / 100\ \text{mm}$ | travel length |
| $\big(h - (z_{\text{top}} - d)\big) / 10\ \text{mm}$ | how far the point's surface is above the commanded tool-bottom plane |
| $d / 10\ \text{mm}$ | commanded depth |

plus a one-hot code of the tool. These features say *where the point is relative to the tool's motion* and *which tool it is*. They do not say what shape the tool has. A pure-learning model has to discover from data that the press is a 40 mm square with rounded edges and the stamp a cross.

The structured model gets one thing more: the output of the prior, which contains the tool's shape. That is the entire difference between "pure learning" and "hybrid" in this project, and it is one formula.

## (a) GNN: a graph network on the cage nodes

The pure-learning model on the simulator's own degrees of freedom, in the style of Sanchez-Gonzalez et al. (2020) and RoboCraft.

**Graph.** The 676 cage nodes, pinned ones included. Each node is connected to its face neighbours on the lattice, in both directions: 3,510 directed edges.

**Node input**, 16 numbers: the reference position of the node (3), its displacement from the reference in units of 10 mm (3), a flag for pinned nodes (1), and the nine action features above.

**Edge input**, 6 numbers: the reference offset between the two nodes and how much the current offset differs from it, both in units of 10 mm. The second triple is the local strain along the edge.

**Encode, process, decode.** Encoders lift nodes and edges to 48 dimensions. Four message-passing steps follow, each with its own weights:

$$
e_{ij} \leftarrow e_{ij} + \phi_e\big(e_{ij},\, h_i,\, h_j\big),\qquad
h_j \leftarrow h_j + \phi_h\Big(h_j,\, \sum_{i \to j} e_{ij}\Big), \tag{1}
$$

where $\phi_e$ and $\phi_h$ are two-layer perceptrons with layer normalisation. A decoder maps each node's final vector to a displacement, multiplied by 10 mm and by zero on pinned nodes:

$$
\hat N' = N + \Delta_\theta(N, \text{tool}, a). \tag{2}
$$

Four steps reach four cells, 47 mm, in every direction: wider than any tool head, and enough for the material pushed out of a dent to be accounted for next to it.

## (b) Structured: the prior plus a residual

The hybrid. The [swept-volume prior](/dynamics/prior/) $P$ gives a first guess of the next nodes. The same graph network as in (a) predicts what the prior left out:

$$
\hat N' = P(N, \text{tool}, a) + \Delta_\theta\big(N, \text{tool}, a,\; P(N, \text{tool}, a) - N\big). \tag{3}
$$

The network receives the prior's displacement as three extra node inputs, so it knows what has already been accounted for. Everything else, architecture, size, optimiser, data, is identical to (a). Any difference in the results between (a) and (b) is the prior.

What the residual has to learn is exactly the list of things the prior cannot do: the sideways flow of material, the bulge around a dent, and the spring-back that leaves a dent shallower than the stroke.

## (c) U-Net on the height map

The pure-learning model in two dimensions. The state is the $32 \times 32$ height map itself, so the model never sees the interior of the block or the side walls.

**Input**, 12 channels: the height relative to the untouched top (1), the nine action features evaluated at each pixel, and the pixel's own coordinates (2).

**Network.** A U-Net with two down-samplings ($32 \to 16 \to 8$), 32, 64 and 128 channels, group normalisation and skip connections; a final $1 \times 1$ convolution gives the change of the map:

$$
\hat H' = H + \Delta_\theta(H, \text{tool}, a). \tag{4}
$$

This model is a different kind of bet. It gives up the three-dimensional state for an image, and convolutions on a small image are fast.

## One currency

The node models output nodes and the U-Net outputs a map. Node predictions are converted to height maps with the exact chain nodes → vertices → map of [Nodes, Vertices, Height Map](/shape/representation/), so every model can be scored, and planned with, on the same quantity.

## Training

| | GNN and structured | U-Net |
|---|---|---|
| parameters | 74,595 and 74,739 | 470,401 |
| loss | mean squared error of the predicted next shape over the free nodes, in units of 10 mm | the same over the pixels |
| optimiser | AdamW, one-cycle learning rate up to $2 \times 10^{-3}$, batch 16, gradient norm clipped at 1 | the same |
| epochs | 6 (2,022 updates) | 30 |
| input noise | Gaussian, 0.2 mm, on the free nodes | Gaussian, 0.2 mm, on the map |

The input noise follows Sanchez-Gonzalez et al.: the model is trained on slightly wrong inputs with the true next shape as the target, so that it has seen inputs like its own imperfect predictions before it is asked to chain them.

The data are 6,600 transitions: 1,100 episodes of six random primitives each, starting from the untouched block. Half of the episodes use one tool throughout and half draw a tool at every step; the parameters are uniform over their ranges. The first 900 episodes train, the next 100 validate and the last 100 are the test set of the [next page](/dynamics/results/). No transition failed.

### Sized for a CPU

There is no GPU build of PyTorch in this project, and the machine has about 5 GB of free memory. The graph network is the expensive model, and memory, not time, set its size. A first benchmark with 26 neighbours per node and 96 hidden units at a batch of 32 was killed by the operating system at 5.6 GB. A network with diagonal edges (9,558 instead of 3,510) and 64 hidden units needed 305 ms per training sample against 58 ms for the same width with face neighbours only. Even the final model, face neighbours only, 48 hidden units and four steps, takes 2.5 GB to train at a batch of 32; three of those side by side exhausted the machine, so it is trained at a batch of 16 (1.4 GB), two at a time.

This matters for reading the results. The comparison between (a) and (b) is at a fixed, small capacity. A larger network or more data would narrow whatever gap the prior opens, and how fast it narrows was not measured.

:::tip[Where in the code]
`learn/models.py`: `action_features`, `GraphNet`, `UNet`, `DynamicsModel` (the common interface: `forward`, `heightmap`, `encode`), `Ensemble`.
`learn/train.py`: `Transitions`, `loss_fn`, `train`, `one_step_metrics`, `multi_step_metrics`.
`learn/data.py`: `generate` (the random-primitive policy).
Tests in `tests/test_models.py`: `test_shapes_and_pinned_nodes`, `test_graph_has_the_grid_edges`.
:::
