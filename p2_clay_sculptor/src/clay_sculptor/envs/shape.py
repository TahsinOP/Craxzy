"""Shape representation of the clay.

Three views of one shape, each an exact function of the previous one:

* **nodes** ``(nx, ny, nz, 3)``: positions of the trilinear cage nodes, in metres,
  in clay coordinates (origin at the block centre on the board, z up).  The free
  nodes are MuJoCo's degrees of freedom of the flex, so this is the full
  configuration of the clay.  The bottom layer is pinned.
* **vertices** ``(vx, vy, vz, 3)``: the interpolated flex vertices,
  ``V = W N`` with the trilinear weights ``W`` (the same linear map MuJoCo applies;
  ``tests/test_shape.py`` checks it against ``mjData.flexvert_xpos``).
* **height map** ``(G, G)``: height of the top surface above the board over a
  square of the table plane, sampled at pixel centres.  The top vertex layer is a
  deformed regular grid, so the surface is the graph ``(u, v) -> (x, y, z)(u, v)``
  with bilinear patches; the height at a pixel is found by inverting the
  ``(u, v) -> (x, y)`` map with a fixed-point iteration, which converges because
  the lateral strain of the surface is far below 1.

Everything here works on batches and on both numpy arrays and torch tensors.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from clay_sculptor.models.scene import ClayCfg


@dataclass(frozen=True)
class MapCfg:
    size: int = 32              # pixels per side
    half: float = 0.062         # pixel centres span [-half, half] in x and y [m]

    @property
    def pitch(self) -> float:
        return 2 * self.half / (self.size - 1)

    @property
    def axis(self) -> np.ndarray:
        return np.linspace(-self.half, self.half, self.size)


class Geometry:
    """Index bookkeeping between MuJoCo's flex arrays and the grids used here."""

    def __init__(self, m: mujoco.MjModel, clay: ClayCfg = ClayCfg(), mp: MapCfg = MapCfg()):
        self.clay, self.map = clay, mp
        self.nn = tuple(c + 1 for c in clay.cells)          # nodes per axis
        self.nv = tuple(clay.count)                         # vertices per axis
        self.origin = np.array([clay.centre[0], clay.centre[1], 0.0])
        n_node = int(m.flex_nodenum[0])
        assert n_node == np.prod(self.nn)
        node0 = m.flex_node0[m.flex_nodeadr[0]:m.flex_nodeadr[0] + n_node]
        self.node0 = (node0 - self.origin).reshape(*self.nn, 3)      # reference node positions
        # degrees of freedom: node (i, j, k > 0) -> three slide joints (x, y, z) on its own body
        body = m.flex_nodebodyid[m.flex_nodeadr[0]:m.flex_nodeadr[0] + n_node].reshape(self.nn)
        assert (body[:, :, 0] == 0).all() and (body[:, :, 1:] > 0).all(), "bottom layer pinned, the rest free"
        self.node_qpos = np.full((*self.nn, 3), -1, dtype=np.int64)
        for idx in np.ndindex(*self.nn):
            b = body[idx]
            if b > 0:
                j = m.body_jntadr[b]
                assert m.body_jntnum[b] == 3 and np.allclose(m.jnt_axis[j:j + 3], np.eye(3))
                self.node_qpos[idx] = m.jnt_qposadr[j:j + 3]
        self.free = self.node_qpos[..., 0] >= 0
        # trilinear weights: vertex (reference coordinate on [0,1]^3) -> 8 surrounding nodes
        v0 = m.flex_vert0[m.flex_vertadr[0]:m.flex_vertadr[0] + m.flex_vertnum[0]]
        self.W = self._weights(v0)                           # (nvert, nnode) dense float64
        self.vert_adr, self.vert_num = int(m.flex_vertadr[0]), int(m.flex_vertnum[0])
        assert self.vert_num == np.prod(self.nv)
        self.radius = clay.radius
        # top vertex layer as a matrix acting on nodes: (vx*vy, nnode)
        vid = np.arange(self.vert_num).reshape(self.nv)
        self.W_top = self.W[vid[:, :, -1].ravel()]
        self.W_top_t = torch.from_numpy(self.W_top.astype(np.float32))
        self.W_t = torch.from_numpy(self.W.astype(np.float32))
        # surface vertices used for Chamfer: top layer and the side walls above the pinned layer
        side = np.zeros(self.nv, dtype=bool)
        side[0], side[-1], side[:, 0], side[:, -1] = True, True, True, True
        side[:, :, 0] = False
        side[:, :, -1] = True
        self.surface_ids = vid[side]

    def _weights(self, v0: np.ndarray) -> np.ndarray:
        cells = np.array(self.clay.cells)
        w = np.zeros((len(v0), int(np.prod(self.nn))))
        g = np.clip(v0, 0.0, 1.0) * cells
        base = np.minimum(np.floor(g).astype(int), cells - 1)
        t = g - base
        nid = np.arange(np.prod(self.nn)).reshape(self.nn)
        rows = np.arange(len(v0))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    wt = (t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * (t[:, 2] if dz else 1 - t[:, 2])
                    np.add.at(w, (rows, nid[base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz]), wt)
        return w

    # ---- state vector <-> nodes -------------------------------------------------
    def nodes_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """``qpos`` ``(..., nq)`` -> nodes ``(..., nx, ny, nz, 3)`` (float64)."""
        disp = np.where(self.node_qpos >= 0, qpos[..., np.maximum(self.node_qpos, 0)], 0.0)
        return self.node0 + disp

    def qpos_from_nodes(self, nodes: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """Write node positions into a copy of ``qpos`` (inverse of ``nodes_from_qpos``)."""
        out = np.array(qpos, dtype=float, copy=True)
        disp = nodes - self.node0
        out[..., self.node_qpos[self.free]] = disp[..., self.free, :]
        return out

    # ---- nodes -> vertices -> height map ---------------------------------------
    def vertices(self, nodes):
        """Nodes ``(..., nx, ny, nz, 3)`` -> vertices ``(..., vx, vy, vz, 3)``."""
        if isinstance(nodes, torch.Tensor):
            flat = nodes.reshape(*nodes.shape[:-4], -1, 3)
            return (self.W_t.to(nodes) @ flat).reshape(*nodes.shape[:-4], *self.nv, 3)
        flat = nodes.reshape(*nodes.shape[:-4], -1, 3)
        return (self.W @ flat).reshape(*nodes.shape[:-4], *self.nv, 3)

    def top(self, nodes):
        """Nodes -> top vertex layer ``(..., vx, vy, 3)``."""
        flat = nodes.reshape(*nodes.shape[:-4], -1, 3)
        w = self.W_top_t.to(nodes) if isinstance(nodes, torch.Tensor) else self.W_top
        return (w @ flat).reshape(*nodes.shape[:-4], self.nv[0], self.nv[1], 3)

    def surface_points(self, nodes: np.ndarray) -> np.ndarray:
        """Nodes -> surface vertex cloud ``(..., ns, 3)`` (top layer and side walls)."""
        flat = nodes.reshape(*nodes.shape[:-4], -1, 3)
        return self.W[self.surface_ids] @ flat

    def heightmap(self, nodes, iters: int = 6):
        """Nodes -> height map ``(..., G, G)`` [m]: top surface height (vertex centre
        + collision radius) above the board at the pixel centres."""
        is_np = not isinstance(nodes, torch.Tensor)
        n = torch.as_tensor(nodes, dtype=torch.float64 if is_np else nodes.dtype)
        top = self.top(n)
        lead = top.shape[:-3]
        top = top.reshape(-1, self.nv[0], self.nv[1], 3)
        h = _surface_height(top, self.map, self.clay, iters) + self.radius
        h = h.reshape(*lead, self.map.size, self.map.size)
        return h.numpy() if is_np else h


def _bilinear(grid: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Sample ``grid`` ``(B, nx, ny, C)`` at fractional indices ``u, v`` ``(B, P)`` -> ``(B, P, C)``."""
    nx, ny = grid.shape[1], grid.shape[2]
    u = u.clamp(0, nx - 1)
    v = v.clamp(0, ny - 1)
    i0 = u.floor().long().clamp(max=nx - 2)
    j0 = v.floor().long().clamp(max=ny - 2)
    a, b = (u - i0).unsqueeze(-1), (v - j0).unsqueeze(-1)
    flat = grid.reshape(grid.shape[0], nx * ny, -1)

    def take(i, j):
        idx = (i * ny + j).unsqueeze(-1).expand(-1, -1, flat.shape[-1])
        return torch.gather(flat, 1, idx)

    return ((1 - a) * (1 - b) * take(i0, j0) + a * (1 - b) * take(i0 + 1, j0)
            + (1 - a) * b * take(i0, j0 + 1) + a * b * take(i0 + 1, j0 + 1))


def _surface_height(top: torch.Tensor, mp: MapCfg, clay: ClayCfg, iters: int) -> torch.Tensor:
    """Height of the bilinear surface through ``top`` ``(B, vx, vy, 3)`` at the pixel centres."""
    b, nx, ny = top.shape[0], top.shape[1], top.shape[2]
    ax = torch.as_tensor(mp.axis, dtype=top.dtype, device=top.device)
    px, py = torch.meshgrid(ax, ax, indexing="ij")
    px, py = px.reshape(1, -1).expand(b, -1), py.reshape(1, -1).expand(b, -1)
    sx, sy = clay.size[0] / (nx - 1), clay.size[1] / (ny - 1)        # reference vertex spacing
    u = (px + clay.size[0] / 2) / sx                                 # initial guess: undeformed grid
    v = (py + clay.size[1] / 2) / sy
    for _ in range(iters):
        p = _bilinear(top, u, v)
        u = u + (px - p[..., 0]) / sx
        v = v + (py - p[..., 1]) / sy
    return _bilinear(top, u, v)[..., 2]


# ---- metrics (evaluation; all in metres) ---------------------------------------
def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance between point sets ``(n, 3)`` and ``(m, 3)``:
    mean nearest-neighbour distance a -> b plus b -> a, divided by two (unsquared)."""
    from scipy.spatial import cKDTree

    return 0.5 * float(cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


def emd(a: np.ndarray, b: np.ndarray) -> float:
    """Earth mover's distance between two point sets of equal size: mean distance
    under the optimal one-to-one assignment (Hungarian algorithm)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    assert len(a) == len(b)
    c = cdist(a, b)
    r, col = linear_sum_assignment(c)
    return float(c[r, col].mean())


def map_rmse(h1, h2):
    """Root-mean-square height difference over the map (last two axes)."""
    d = h1 - h2
    if isinstance(d, torch.Tensor):
        return d.pow(2).mean(dim=(-1, -2)).sqrt()
    return np.sqrt((d ** 2).mean(axis=(-1, -2)))
