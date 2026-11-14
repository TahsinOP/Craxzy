"""Learned one-primitive dynamics models: (shape, tool, action) -> next shape.

Three models, one interface (``DynamicsModel``):

* ``gnn``         pure learning on the cage nodes: an encode-process-decode graph
                  network (Sanchez-Gonzalez et al. 2020; RoboCraft) predicts the
                  displacement of every free node.
* ``structured``  hybrid: the swept-volume prior (``learn.prior``) gives a first
                  guess of the next nodes and the same graph network predicts the
                  residual (spring-back, bulge, everything the prior leaves out).
* ``unet``        pure learning on the 2D height map: a U-Net predicts the change
                  of the map.

What every learned model is told about the action is the same and is purely
geometric: where each node (or pixel) sits in the tool frame, how far it is above
the commanded tool-bottom plane, the travel length and the tool identity.  None of
the pure models is given the tool's shape; the structured model gets it through
the prior.

Units inside the networks: lengths are divided by ``L0 = 10 mm`` so displacements
are O(1).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from clay_sculptor.envs.shape import Geometry
from clay_sculptor.learn.prior import SweptVolumePrior, tool_frame
from clay_sculptor.models.scene import TOOLS

L0 = 0.010          # length unit of the network inputs and outputs [m]
XY0 = 0.05          # scale of tool-frame coordinates [m]


def mlp(n_in: int, n_hidden: int, n_out: int, norm: bool = True) -> nn.Sequential:
    layers = [nn.Linear(n_in, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_out)]
    if norm:
        layers.append(nn.LayerNorm(n_out))
    return nn.Sequential(*layers)


def action_features(xy: torch.Tensor, height: torch.Tensor, tool: torch.Tensor, action: torch.Tensor, top: float) -> torch.Tensor:
    """Per-point action features ``(B, ..., 9)`` from point positions ``xy`` ``(B, ..., 2)``
    and surface heights ``height`` ``(B, ...)`` (vertex centre + radius, or map value)."""
    shape = (-1,) + (1,) * (xy.dim() - 2)
    lx, ly = tool_frame(xy, action)
    half_travel = 0.5 * action[:, 4].reshape(shape)
    ly_out = torch.sign(ly) * (ly.abs() - half_travel).clamp(min=0.0)      # distance beyond the travel segment
    gap = (height - (top - action[:, 3].reshape(shape))) / L0             # > 0: above the tool-bottom plane
    onehot = torch.nn.functional.one_hot(tool.long(), len(TOOLS)).to(xy.dtype).reshape(-1, *(1,) * (xy.dim() - 2), len(TOOLS))
    feats = [(lx / XY0).clamp(-3, 3), (ly / XY0).clamp(-3, 3), (ly_out / XY0).clamp(-3, 3),
             (action[:, 4].reshape(shape) / 0.1).expand_as(lx), gap, (action[:, 3].reshape(shape) / L0).expand_as(lx)]
    return torch.cat([torch.stack(feats, dim=-1), onehot.expand(*lx.shape, len(TOOLS))], dim=-1)


class GraphNet(nn.Module):
    """Encode-process-decode network on the cage-node grid graph.

    Nodes: all cage nodes (pinned ones included, flagged).  Edges: both directions
    between grid neighbours.  ``reach = 1`` connects face neighbours (6 per interior
    node), ``reach = 2`` adds the edge diagonals of a cell (18), ``reach = 3`` the body
    diagonals (26).  The default is sized for CPU training: memory and time grow with
    the number of edges."""

    def __init__(self, geom: Geometry, n_extra: int = 0, hidden: int = 64, steps: int = 5, reach: int = 1):
        super().__init__()
        self.geom = geom
        nn_ = geom.nn
        idx = np.arange(np.prod(nn_)).reshape(nn_)
        send, recv = [], []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if not 0 < abs(dx) + abs(dy) + abs(dz) <= reach:
                        continue
                    src = idx[max(0, -dx):nn_[0] - max(0, dx), max(0, -dy):nn_[1] - max(0, dy), max(0, -dz):nn_[2] - max(0, dz)]
                    dst = idx[max(0, dx):nn_[0] - max(0, -dx), max(0, dy):nn_[1] - max(0, -dy), max(0, dz):nn_[2] - max(0, -dz)]
                    send.append(src.ravel())
                    recv.append(dst.ravel())
        self.register_buffer("send", torch.from_numpy(np.concatenate(send)))
        self.register_buffer("recv", torch.from_numpy(np.concatenate(recv)))
        node0 = torch.from_numpy(geom.node0.reshape(-1, 3).astype(np.float32))
        self.register_buffer("node0", node0)
        self.register_buffer("free", torch.from_numpy(geom.free.reshape(-1).astype(np.float32)))
        half = torch.tensor([geom.clay.size[0] / 2, geom.clay.size[1] / 2, geom.clay.size[2]], dtype=torch.float32)
        self.register_buffer("ref_feat", node0 / half)
        self.register_buffer("edge_ref", (node0[self.recv] - node0[self.send]) / L0)
        self.n_node_in = 3 + 3 + 1 + 9 + n_extra
        self.enc_node = mlp(self.n_node_in, hidden, hidden)
        self.enc_edge = mlp(6, hidden, hidden)
        self.edge_fn = nn.ModuleList([mlp(3 * hidden, hidden, hidden) for _ in range(steps)])
        self.node_fn = nn.ModuleList([mlp(2 * hidden, hidden, hidden) for _ in range(steps)])
        self.dec = mlp(hidden, hidden, 3, norm=False)

    def forward(self, nodes: torch.Tensor, tool: torch.Tensor, action: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """``nodes`` ``(B, N, 3)`` [m] -> displacement ``(B, N, 3)`` [m] (zero on pinned nodes)."""
        b, n = nodes.shape[0], nodes.shape[1]
        clay = self.geom.clay
        disp = (nodes - self.node0) / L0
        act = action_features(nodes[..., :2], nodes[..., 2] + clay.radius, tool, action, clay.top)
        feats = [self.ref_feat.expand(b, -1, -1), disp, (1 - self.free).reshape(1, n, 1).expand(b, -1, -1), act]
        if extra is not None:
            feats.append(extra)
        h = self.enc_node(torch.cat(feats, dim=-1))
        rel = (nodes[:, self.recv] - nodes[:, self.send]) / L0
        e = self.enc_edge(torch.cat([self.edge_ref.expand(b, -1, -1), rel - self.edge_ref], dim=-1))
        for edge_fn, node_fn in zip(self.edge_fn, self.node_fn):
            e = e + edge_fn(torch.cat([e, h[:, self.send], h[:, self.recv]], dim=-1))
            agg = torch.zeros_like(h).index_add_(1, self.recv, e)
            h = h + node_fn(torch.cat([h, agg], dim=-1))
        return self.dec(h) * L0 * self.free.reshape(1, n, 1)


class UNet(nn.Module):
    """U-Net on the height map: 32 -> 16 -> 8 and back, with skip connections."""

    def __init__(self, n_in: int, base: int = 32):
        super().__init__()

        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU())

        self.e1, self.e2, self.e3 = block(n_in, base), block(base, 2 * base), block(2 * base, 4 * base)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
        self.d2 = block(4 * base, 2 * base)
        self.up1 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
        self.d1 = block(2 * base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1).squeeze(1)


class DynamicsModel(nn.Module):
    """Common interface.  ``kind`` is ``gnn``, ``structured``, ``unet`` or ``prior``
    (the swept-volume prior alone: no parameters, the classical baseline).

    ``state`` is nodes ``(B, nx, ny, nz, 3)`` for the node models and a height map
    ``(B, G, G)`` for the U-Net (``state_kind`` says which).  ``forward`` returns the
    next state in the same representation; ``heightmap`` converts a state to a map,
    so every model can be scored and planned with in the same currency."""

    def __init__(self, kind: str, geom: Geometry, hidden: int = 64, steps: int = 5, base: int = 32, reach: int = 1):
        super().__init__()
        self.kind, self.geom = kind, geom
        self.state_kind = "map" if kind == "unet" else "nodes"
        self.prior = SweptVolumePrior(geom) if kind in ("structured", "prior") else None
        if kind == "gnn":
            self.net = GraphNet(geom, 0, hidden, steps, reach)
        elif kind == "structured":
            self.net = GraphNet(geom, 3, hidden, steps, reach)
        elif kind == "unet":
            self.net = UNet(1 + 9 + 2, base)
            ax = torch.from_numpy(geom.map.axis.astype(np.float32))
            gx, gy = torch.meshgrid(ax, ax, indexing="ij")
            self.register_buffer("pix_xy", torch.stack([gx, gy], dim=-1))
        elif kind == "prior":
            self.net = None
        else:
            raise ValueError(kind)

    def forward(self, state: torch.Tensor, tool: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        clay = self.geom.clay
        if self.kind == "unet":
            b = state.shape[0]
            xy = self.pix_xy.expand(b, -1, -1, -1)
            act = action_features(xy, state, tool, action, clay.top)
            x = torch.cat([((state - clay.top) / L0).unsqueeze(-1), act, xy / XY0], dim=-1).permute(0, 3, 1, 2)
            return state + self.net(x) * L0
        shape = state.shape
        flat = state.reshape(shape[0], -1, 3)
        if self.kind == "gnn":
            return (flat + self.net(flat, tool, action)).reshape(shape)
        prior = self.prior(state, tool, action).reshape(shape[0], -1, 3)
        if self.kind == "prior":
            return prior.reshape(shape)
        return (prior + self.net(flat, tool, action, extra=(prior - flat) / L0)).reshape(shape)

    def heightmap(self, state: torch.Tensor) -> torch.Tensor:
        return state if self.state_kind == "map" else self.geom.heightmap(state)

    def encode(self, nodes: torch.Tensor) -> torch.Tensor:
        """True nodes -> this model's state representation."""
        return self.geom.heightmap(nodes) if self.state_kind == "map" else nodes

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Ensemble:
    """Several independently trained members of one kind (deep ensemble).

    ``predict`` returns the member predictions stacked on a new first axis; the
    spread across that axis is the disagreement signal of M4."""

    def __init__(self, members: list[DynamicsModel]):
        self.members = members
        self.kind, self.geom = members[0].kind, members[0].geom
        self.state_kind = members[0].state_kind

    @torch.no_grad()
    def predict(self, state: torch.Tensor, tool: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(state, tool, action) for m in self.members])

    def heightmap(self, state: torch.Tensor) -> torch.Tensor:
        return self.members[0].heightmap(state)

    def encode(self, nodes: torch.Tensor) -> torch.Tensor:
        return self.members[0].encode(nodes)
