"""Geometric prior: the tool sweeps out a volume and the clay gets out of its way.

Known exactly (structure): the tool head geometry and where the arm takes it.  The
lowest surface the tool reaches over the whole primitive is, at a point ``(X, Y)``
of the table plane,

    z_env(X, Y) = z_bottom + r - sqrt(r^2 - d^2)        for d < r,   +inf otherwise,

where ``z_bottom = top - depth`` is the commanded height of the tool bottom, ``r``
the edge radius of the head and ``d`` the horizontal distance from ``(X, Y)`` to the
head's flat footprint, swept along the travel segment for the roller.  This is the
lower envelope of a "stadium" solid (flat footprint Minkowski-summed with a ball of
radius ``r``), which is what the tool heads are (``models.scene.TOOL_GEOMS``).

A flex vertex is a sphere of radius ``rho``, so its centre must stay below the
envelope of the tool inflated by ``rho``: the same formula with ``r + rho`` and
``z_bottom - rho``.

Prior update of the shape:

1. every top-layer vertex is clipped to the inflated envelope: ``z' = min(z, z_env)``;
2. the column under it is compressed in proportion (the bottom is pinned):
   ``z'_k = z_0 + (z_k - z_0) (z'_top - z_0) / (z_top - z_0)``;
3. the vertex displacement is carried over to the cage nodes, because the nodes are
   the state: each node takes the weighted mean of the displacements of the vertices
   it influences (``projection="lumped"``; least squares is available for comparison).

The prior moves nothing sideways, so it has no bulge, and it has no spring-back.
Those two are what the learned residual is for.
"""
from __future__ import annotations

import numpy as np
import torch

from clay_sculptor.envs.shape import Geometry
from clay_sculptor.models.scene import TOOL_GEOMS, TOOLS


def tool_frame(xy: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Coordinates of table-plane points in the tool frame.

    ``xy`` ``(B, ..., 2)``, ``action`` ``(B, 5)`` -> ``(lx, ly)`` with ``lx`` along the
    tool x axis (the roller axis) and ``ly`` along the travel direction."""
    shape = (-1,) + (1,) * (xy.dim() - 2)
    psi = action[:, 2].reshape(shape)
    rx = xy[..., 0] - action[:, 0].reshape(shape)
    ry = xy[..., 1] - action[:, 1].reshape(shape)
    c, s = torch.cos(psi), torch.sin(psi)
    return c * rx + s * ry, s * rx - c * ry


def envelope(xy: torch.Tensor, tool: torch.Tensor, action: torch.Tensor, top: float, inflate: float = 0.0) -> torch.Tensor:
    """Lowest height reached by the tool (inflated by ``inflate``) above each point.

    ``xy`` ``(B, ..., 2)`` in clay coordinates, ``tool`` ``(B,)`` ids, ``action`` ``(B, 5)``.
    Returns ``(B, ...)`` heights; ``+inf`` where the tool never passes over the point."""
    lx, ly = tool_frame(xy, action)
    shape = (-1,) + (1,) * (xy.dim() - 2)
    depth = action[:, 3].reshape(shape)
    half_travel = 0.5 * action[:, 4].reshape(shape)
    out = torch.full_like(lx, float("inf"))
    for tid, name in enumerate(TOOLS):
        sel = (tool == tid).reshape(shape).expand_as(lx)
        if not bool(sel.any()):
            continue
        z = torch.full_like(lx, float("inf"))
        for kind, hl, hw, r, yaw in TOOL_GEOMS[name].parts:
            # the two stamp bars are at yaw 0 and pi/2; swap the axes for the second
            px, py = (lx, ly) if abs(yaw) < 1e-9 else (ly, lx)
            if kind == "capsule":      # flat footprint: the axis segment, swept along the travel
                fx, fy = hl, half_travel
            else:                      # rounded box: the flat rectangle (no travel for these tools)
                fx, fy = hl - r, hw - r
            dx = (px.abs() - fx).clamp(min=0.0)
            dy = (py.abs() - fy).clamp(min=0.0)
            d2 = dx * dx + dy * dy
            rr = r + inflate
            zz = (top - depth - inflate) + rr - torch.sqrt((rr * rr - d2).clamp(min=0.0))
            z = torch.minimum(z, torch.where(d2 < rr * rr, zz, torch.full_like(zz, float("inf"))))
        out = torch.where(sel, z, out)
    return out


class SweptVolumePrior:
    """``projection``: how the vertex displacement becomes a node displacement.

    * ``"lsq"``     least squares (pseudo-inverse of the interpolation matrix): the
                    closest cage shape to the clipped vertices, but a sharp dent on a
                    coarse cage rings, so nodes next to the dent rise a little;
    * ``"lumped"``  each node takes the weighted mean of the displacements of the
                    vertices it influences (weights = interpolation weights): never
                    overshoots, but rounds the dent off.

    The default is ``"lumped"``: on held-out transitions its height-map error is
    0.61 mm against 0.80 mm for least squares (and 1.71 mm for predicting no change).
    The cage cannot hold a sharp dent anyway, so rounding it off is closer to what
    the simulator does than the nearest sharp shape is.
    """

    def __init__(self, geom: Geometry, projection: str = "lumped"):
        self.geom = geom
        self.projection = projection
        w_ = geom.W
        self.W_lump = torch.from_numpy((w_.T / w_.sum(axis=0)[:, None]).astype(np.float32))    # (nnode, nvert)
        self.top = geom.clay.top
        self.rho = geom.clay.radius
        self.W = torch.from_numpy(geom.W.astype(np.float32))                       # (nvert, nnode)
        self.W_pinv = torch.from_numpy(np.linalg.pinv(geom.W).astype(np.float32))  # (nnode, nvert)
        self.free = torch.from_numpy(geom.free.reshape(-1))

    def vertices(self, verts: torch.Tensor, tool: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Prior on the vertices ``(B, vx, vy, vz, 3)`` (steps 1 and 2)."""
        z = verts[..., 2]
        z_top, z_bot = z[..., -1], z[..., 0]
        env = envelope(verts[..., -1, :2], tool, action, self.top, inflate=self.rho)
        new_top = torch.minimum(z_top, env).clamp(min=z_bot + 1e-4)
        scale = ((new_top - z_bot) / (z_top - z_bot).clamp(min=1e-6)).unsqueeze(-1)
        new_z = z_bot.unsqueeze(-1) + (z - z_bot.unsqueeze(-1)) * scale
        return torch.cat([verts[..., :2], new_z.unsqueeze(-1)], dim=-1)

    def __call__(self, nodes: torch.Tensor, tool: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Prior next nodes ``(B, nx, ny, nz, 3)`` from current nodes, tool ids and actions."""
        b = nodes.shape[0]
        flat = nodes.reshape(b, -1, 3)
        w = self.W.to(nodes)
        verts = (w @ flat).reshape(b, *self.geom.nv, 3)
        dv = (self.vertices(verts, tool, action) - verts).reshape(b, -1, 3)
        dn = (self.W_pinv if self.projection == "lsq" else self.W_lump).to(nodes) @ dv
        dn = dn * self.free.to(nodes).reshape(1, -1, 1)          # pinned nodes do not move
        return (flat + dn).reshape(nodes.shape)
