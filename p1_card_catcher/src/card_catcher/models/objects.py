"""Thrown objects: geometry, mass and the fluid (aerodynamic) parameters.

All three objects use MuJoCo's ellipsoid fluid model (``fluidshape="ellipsoid"``),
which acts on the geom's equivalent ellipsoid with semi-axes ``(a, b, c)``:

* sphere of radius r        -> (r, r, r)
* cylinder (r, half-height) -> (r, r, h)
* box (half-sizes)          -> (sx, sy, sz)

Translational drag of that model (MuJoCo docs, computation/fluid.html, and
``engine_passive.c``), for body-frame velocity v:

    f = -[ beta * 6 pi r_eq                                (Stokes, viscosity beta)
           + rho |v| ( C_blunt A_proj(v) + C_slender (A_max - A_proj(v)) ) ] v

with ``A_proj`` the ellipsoid's projected area normal to v, ``A_max`` its largest
projected area and r_eq = (a+b+c)/3.  For a sphere A_proj = A_max = pi r^2, so the
quadratic part is *exactly* isotropic,

    a_drag = -k_d |v| v,      k_d = rho C_blunt pi r^2 / m    [1/m].

That is the "same drag form" the physics predictor uses (TASK.md pitfall 3).  For
the plate and the card the true drag is anisotropic and coupled to lift, Magnus
and added-mass (Munk) moments; the predictor keeps the isotropic form with a
fitted coefficient, and everything else is the *residual* that M2 learns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MuJoCo defaults for geom/fluidcoef: blunt drag, slender drag, angular drag, Kutta lift, Magnus lift.
FLUIDCOEF_DEFAULT = (0.5, 0.25, 1.5, 1.0, 1.0)


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    gtype: str                      # MJCF geom type
    size: tuple[float, ...]         # MJCF geom size
    mass: float                     # kg
    rgba: tuple[float, float, float, float]
    fluidcoef: tuple[float, ...] = FLUIDCOEF_DEFAULT
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    condim: int = 3

    @property
    def semi_axes(self) -> np.ndarray:
        if self.gtype == "sphere":
            return np.full(3, self.size[0])
        if self.gtype == "cylinder":
            return np.array([self.size[0], self.size[0], self.size[1]])
        if self.gtype == "box":
            return np.array(self.size, float)
        raise ValueError(self.gtype)

    @property
    def bound_radius(self) -> float:
        """Radius of the bounding sphere of the geom."""
        if self.gtype == "sphere":
            return self.size[0]
        if self.gtype == "cylinder":
            return float(np.hypot(self.size[0], self.size[1]))
        return float(np.linalg.norm(self.size))

    def projected_area(self, direction: np.ndarray) -> float:
        """Projected area of the equivalent ellipsoid normal to a body-frame direction."""
        a, b, c = self.semi_axes
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        return float(np.pi * np.sqrt((b * c * d[0]) ** 2 + (c * a * d[1]) ** 2 + (a * b * d[2]) ** 2))

    def kd_axis(self, direction: np.ndarray, rho: float = 1.2) -> float:
        """Quadratic drag coefficient k_d [1/m] for motion along a body-frame direction."""
        a, b, c = self.semi_axes
        a_max = np.pi * max(a * b, b * c, c * a)
        a_proj = self.projected_area(direction)
        cb, cs = self.fluidcoef[0], self.fluidcoef[1]
        return rho * (cb * a_proj + cs * (a_max - a_proj)) / self.mass

    def geom_xml(self, name: str = "obj") -> str:
        size = " ".join(f"{s:g}" for s in self.size)
        rgba = " ".join(f"{s:g}" for s in self.rgba)
        fc = " ".join(f"{s:g}" for s in self.fluidcoef)
        fr = " ".join(f"{s:g}" for s in self.friction)
        return (
            f'<geom name="{name}" type="{self.gtype}" size="{size}" mass="{self.mass:g}" rgba="{rgba}" '
            f'fluidshape="ellipsoid" fluidcoef="{fc}" friction="{fr}" condim="{self.condim}"/>'
        )


OBJECTS: dict[str, ObjectSpec] = {
    # soft juggling ball: 5 cm diameter, 40 g
    "ball": ObjectSpec("ball", "sphere", (0.025,), 0.040, (0.95, 0.45, 0.10, 1.0)),
    # light plastic plate / mini frisbee: 16 cm diameter, 6 mm thick, 60 g
    "plate": ObjectSpec("plate", "cylinder", (0.08, 0.003), 0.060, (0.20, 0.55, 0.90, 1.0)),
    # poker-size playing card: 63 x 88 x 0.7 mm, 1.8 g
    "card": ObjectSpec("card", "box", (0.0315, 0.044, 0.00035), 0.0018, (0.92, 0.92, 0.95, 1.0)),
}
