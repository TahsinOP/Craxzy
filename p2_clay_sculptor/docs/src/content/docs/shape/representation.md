---
title: "Nodes, Vertices, Height Map"
description: "Three views of one clay shape, each an exact function of the one before: the cage nodes the simulator integrates, the vertices it collides and draws, and the height map every planner is scored on."
sidebar:
  order: 1
---

:::note[Guiding question]
A planner compares shapes thousands of times per decision. What should a "shape" be, so that the simulator, three different learned models and the cost function all mean the same thing by it?
:::

## The cage nodes are the state

The clay is a MuJoCo flex with a **trilinear cage** ([The Clay](/bench/clay/)): $13 \times 13 \times 4$ nodes on a regular lattice around a $140 \times 140 \times 40$ mm block. The bottom layer of 169 nodes is pinned to the board. Each of the other 507 nodes is a body with three slide joints, so the configuration of the clay is exactly

$$
N \in \mathbb{R}^{13 \times 13 \times 4 \times 3},
$$

the node positions, with $3 \times 507 = 1521$ free numbers. Positions are expressed in **clay coordinates**: origin at the centre of the block on the board, $x$ away from the robot, $z$ up.

This is the representation the graph-network models work on. It is not a subsample chosen for convenience: it is the simulator's own set of degrees of freedom, so nothing about the shape is lost.

## Vertices: a fixed linear map

MuJoCo collides and draws the flex through $25 \times 25 \times 4$ interpolated vertices. A vertex with reference coordinate $(\xi, \eta, \zeta) \in [0,1]^3$ inside a cage cell is the trilinear blend of the cell's eight corner nodes,

$$
v = \sum_{a,b,c \in \{0,1\}} w_a(\xi)\,w_b(\eta)\,w_c(\zeta)\; n_{abc},\qquad w_0(t) = 1 - t,\quad w_1(t) = t. \tag{1}
$$

The weights depend only on the reference coordinates, which never change, so all vertices are one constant matrix applied to the nodes:

$$
V = W N,\qquad W \in \mathbb{R}^{2500 \times 676},\quad W \ge 0,\quad W\mathbf{1} = \mathbf{1}. \tag{2}
$$

Every second vertex along $x$ and $y$ coincides with a node; the others sit halfway. Equation (2) is not an approximation of what the simulator does, it is what the simulator does: the largest difference between $WN$ and `mjData.flexvert_xpos` after a press is $10^{-16}$ m.

Two consequences are used later. A learned model that predicts nodes has predicted every vertex, exactly. And a displacement given on the vertices can be carried back to the nodes through $W^\top$, which is how the [geometric prior](/dynamics/prior/) enters the node models.

## The height map

Targets are compared on the **top surface**. The top vertex layer, $25 \times 25$ points, is a deformed regular grid. Index it by continuous coordinates $(u, v) \in [0, 24]^2$ and interpolate bilinearly inside each quad; that defines a surface

$$
(u, v) \;\mapsto\; \big(x(u,v),\; y(u,v),\; z(u,v)\big).
$$

The height map is this surface sampled on a fixed raster of the table plane: $32 \times 32$ pixel centres $(X_i, Y_j)$, 4 mm apart, spanning $\pm 62$ mm. A vertex is a sphere of radius $\rho = 2$ mm, so

$$
H_{ij} = z\big(u_{ij}, v_{ij}\big) + \rho,\qquad\text{where}\quad x(u_{ij}, v_{ij}) = X_i,\;\; y(u_{ij}, v_{ij}) = Y_j. \tag{3}
$$

The catch is the second condition. Clay moves sideways when it is pressed, so the vertex that was above a pixel is no longer above it, and $H$ cannot be read off the grid. The $(u, v)$ belonging to a pixel has to be solved for.

### Inverting the lateral map by fixed-point iteration

Let $s = 140/24$ mm be the reference spacing of the top vertices. In the undeformed block $x = s\,u - 70$ mm, so $u = (X + 70\,\text{mm})/s$ is exact. For the deformed surface, start there and repeat

$$
u \leftarrow u + \frac{X - x(u, v)}{s},\qquad v \leftarrow v + \frac{Y - y(u, v)}{s}. \tag{4}
$$

Write the lateral position as the undeformed one plus a displacement, $x = s\,u - 70 + d_x(u,v)$. Then (4) is the fixed-point iteration $u \leftarrow (X + 70 - d_x(u, v))/s$, a contraction whenever

$$
\frac{1}{s}\left\lVert \frac{\partial (d_x, d_y)}{\partial (u, v)} \right\rVert < 1,
$$

that is, whenever the lateral **strain** of the surface is below 100 %. The largest lateral displacement in this project is about 13 mm over a 140 mm block, with strains of a few tens of percent at most, so the error shrinks by at least a factor of three per pass. Six passes are used.

:::caution[Why not rasterise triangles]
Splitting each quad into two triangles and interpolating linearly gives another valid surface. It differs from the bilinear one by up to 0.15 mm on a pressed block, because a bent quad is not two flat triangles. Neither is "the" surface; what matters is that one definition is used everywhere. The bilinear one was chosen because (4) is a handful of tensor operations, runs on a batch of hundreds of candidate shapes at once, and is differentiable.
:::

The implementation is checked on a case with a known answer. Shear the top surface sideways by a smooth field, then tilt it so that $z = c + a\,x + b\,y$ in the *deformed* coordinates. Bilinear patches preserve a linear relation between $x$, $y$ and $z$, so the recovered map must be that plane whatever the shear. It is, to $10^{-9}$ m.

### What the map leaves out

The raster stops 8 mm inside the edge of the block, so the side walls and their bulge are not in $H$. This is deliberate: a pixel on the edge would jump between 44 mm and 0 mm when a wall moves by a millimetre, and that jump would dominate any cost built on $H$. The side walls are covered by the Chamfer distance below.

## Three distances between shapes

All three are in millimetres and all are zero only for identical shapes.

**Height-map RMSE.** The cost every planner minimises:

$$
c(H, H^\star) = \sqrt{\frac{1}{32^2}\sum_{i,j}\big(H_{ij} - H^\star_{ij}\big)^2}. \tag{5}
$$

It is cheap, it is sensitive to depth errors of any size, and it can be computed from the output of every model in the project, which is the reason it is the common currency.

**Chamfer distance.** Between two point clouds $A$ and $B$ (here the 817 surface vertices: the top layer and the side walls above the pinned layer),

$$
d_{\mathrm{CD}}(A, B) = \frac{1}{2}\left(\frac{1}{|A|}\sum_{a \in A}\min_{b \in B}\lVert a - b\rVert + \frac{1}{|B|}\sum_{b \in B}\min_{a \in A}\lVert a - b\rVert\right). \tag{6}
$$

Distances are not squared, so the unit is a length. Chamfer sees the side walls, but it saturates: once two surfaces are further apart than the spacing of the points, the nearest neighbour is found sideways and the distance stops growing.

**Earth mover's distance.** Between the two top layers (625 points each), the mean distance under the best one-to-one assignment $\pi$:

$$
d_{\mathrm{EMD}}(A, B) = \min_{\pi}\; \frac{1}{|A|}\sum_{a \in A}\lVert a - \pi(a)\rVert. \tag{7}
$$

It is computed exactly with the Hungarian algorithm. Unlike Chamfer it cannot match many points to one, so it also penalises material that sits in the wrong place.

Chamfer and EMD are evaluation metrics only. They are always computed on **true simulator states**, never on a model's prediction, so a model cannot look good by predicting a flattering point cloud.

:::tip[Where in the code]
`envs/shape.py`: `Geometry` (index maps, $W$, `nodes_from_qpos`, `vertices`, `top`, `heightmap`), `_surface_height` (iteration (4)), `chamfer`, `emd`, `map_rmse`.
Tests in `tests/test_bench.py`: `test_vertices_are_trilinear_in_the_nodes` (equation (2) against MuJoCo), `test_heightmap_exact_on_a_sheared_plane`.
:::
