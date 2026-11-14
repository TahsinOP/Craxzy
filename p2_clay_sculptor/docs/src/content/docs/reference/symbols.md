---
title: "Symbols"
description: "Every symbol used in this site, with its unit and where it is defined."
sidebar:
  order: 3
---

Lengths are in metres in the code and in millimetres in the text, unless a unit is given. Clay coordinates have their origin at the centre of the block on the board, $x$ away from the robot, $z$ up.

## Clay and simulation

| symbol | meaning | value or unit | defined in |
|---|---|---|---|
| $E$, $\nu$ | Young's modulus, Poisson ratio | 150 kPa, 0.3 | [The Clay](/bench/clay/) |
| $\mu$, $\lambda$ | Lamé parameters | 57.7 kPa, 86.5 kPa | [The Clay](/bench/clay/) |
| $K$ | bulk modulus, $E / (3(1 - 2\nu))$ | 125 kPa | [The Clay](/bench/clay/) |
| $\varepsilon$, $\varepsilon_e$, $\varepsilon_p$ | total, elastic and plastic strain | dimensionless | [The Clay](/bench/clay/) |
| $\varepsilon_y$ | yield strain | 0.02 | [The Clay](/bench/clay/) |
| $e$ | deviatoric part of the elastic strain | dimensionless | [The Clay](/bench/clay/) |
| $h$ | timestep | 5 ms | [The Clay](/bench/clay/) |
| $z_{\text{top}}$ | top of the untouched block (vertex centres plus radius) | 44 mm | [Action Primitives](/bench/primitives/) |
| $\rho$ | collision radius of a flex vertex | 2 mm | [Nodes, Vertices, Height Map](/shape/representation/) |

## Shape

| symbol | meaning | size | defined in |
|---|---|---|---|
| $N$ | cage node positions, the state of the clay | $13 \times 13 \times 4 \times 3$ | [Nodes, Vertices, Height Map](/shape/representation/) |
| $V$ | interpolated vertex positions | $25 \times 25 \times 4 \times 3$ | same |
| $W$ | trilinear interpolation matrix, $V = WN$ | $2500 \times 676$ | same |
| $H$, $H^\star$ | height map of the top surface, of the target | $32 \times 32$, 4 mm pixels | same |
| $(u, v)$ | continuous grid coordinates of the top vertex layer | | same |
| $c(H, H^\star)$ | height-map RMSE, the planning cost | mm | same, (5) |
| $d_{\mathrm{CD}}$, $d_{\mathrm{EMD}}$ | Chamfer and earth mover's distances | mm | same, (6) and (7) |

## Actions and the arm

| symbol | meaning | range or unit | defined in |
|---|---|---|---|
| $a = (x, y, \psi, d, \ell)$ | a primitive | | [Action Primitives](/bench/primitives/) |
| $x, y$ | centre of the action, from the block centre | $\pm 55$ mm | same |
| $\psi$ | tool yaw | press, stamp $\pm 45°$; roller $\pm 90°$ | same |
| $d$ | depth below the untouched top | 2 to 14 mm | same |
| $\ell$ | travel of the roller | 20 to 100 mm | same |
| $t(\psi)$ | travel direction, $(\sin\psi, -\cos\psi)$ | unit vector | same |
| $s(\tau)$ | minimum-jerk path parameter | $[0, 1]$ | same, (1) |
| $R(\psi)$ | target rotation of the tool tip | | same |
| $J$, $J^{+}$ | tip Jacobian and its pseudo-inverse | $6 \times 7$ | same, (2) |
| $q$, $q_h$ | joint angles, home configuration | rad | same |
| $k_p$, $k_v$, $k_i$ | servo gains | N m/rad, N m s/rad, N m/(rad s) | [Scene and Tools](/bench/scene/) |
| $T_i$ | integral time, $k_p / k_i$ | 0.05 s | same |

## Prior and models

| symbol | meaning | defined in |
|---|---|---|
| $z_b$ | commanded height of the tool bottom, $z_{\text{top}} - d$ | [The Geometric Prior](/dynamics/prior/) |
| $r$ | edge radius of a tool head | same |
| $(f_x, f_y)$ | half-sizes of the flat footprint | same |
| $(l_x, l_y)$ | coordinates in the tool frame | same |
| $\delta$ | horizontal distance to the footprint | same |
| $z_{\text{env}}$, $z^{\rho}_{\text{env}}$ | lower envelope of the tool, inflated by $\rho$ | same, (1) and (2) |
| $P(N, a)$ | prior prediction of the next nodes | same |
| $\Delta_\theta$ | learned displacement (pure model) or residual (structured) | [Three Models](/dynamics/models/) |
| $\hat N'$, $\hat H'$ | predicted next nodes, next height map | same |

## Planning and the fallback

| symbol | meaning | value | defined in |
|---|---|---|---|
| $P$, $E$, $I$ | population, elite set, iterations of the cross-entropy method | 32, 6, 3 | [The Planner](/planning/cem/) |
| $T$ | symmetry period of the yaw | $\pi/2$ or $\pi$ | same, (2) |
| $\hat H_m(a)$ | height map predicted by ensemble member $m$ | | [Asking the Simulator](/fallback/ensemble/) |
| $\bar H(a)$, $c(a)$ | ensemble mean map and its cost | | same, (1) |
| $u(a)$ | ensemble disagreement | mm | same, (2) |
| $c_m(a)$ | cost of $a$ according to ensemble member $m$ | mm | same |
| $p_{\text{mis}}$ | misranking probability of the chosen action | $[0, 1]$ | same, (3) |
| $\Phi$ | standard normal distribution function | | same |
| $\tau$ | threshold on $u$ or $p_{\text{mis}}$ above which the simulator is asked | mm or probability | same |
