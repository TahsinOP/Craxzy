"""Target shapes.

Every target is produced by a hidden scripted demonstration in the simulator, so
it is reachable by construction: a perfect planner can reach zero error with the
demonstration's actions, and the error of a planner is measured on an absolute
scale.  The planner is given the target shape and the tool, never the actions.

* ``slab``  flatten the top to a lower level: four overlapping roller passes;
* ``dent``  one off-centre, rotated pocket made with the flat press;
* ``x``     an X pressed with the cross stamp (the cross at 45 degrees).
* ``t``     (stretch) the letter T pressed with the flat press: three presses for the
            bar, two for the stem;
* ``combo`` (stretch) a rolled band, a stamped cross and a pressed pocket: the tool
            is not given, the planner chooses it at every step.
* ``steps``  (unseen demo) a staircase of three square pockets along the diagonal,
            pressed 6, 9 and 12 mm deep and turned 45 degrees; added after every
            model and planner setting was fixed, for ``scripts/demo_unseen.py``.
* ``smiley`` (demo, big block only) two eyes pressed 12 mm and a smile of five 10 mm
            presses on an arc, on a 245 mm block (``scripts/demo_unseen.py --target
            smiley`` sets the block up; the presses lie outside the default block).
* ``house``  (demo, big block only, tool not given) a house: the roof is two rolls at
            +-45 degrees meeting at the top, the window a cross stamp, the door a
            square press; the planner chooses the tool at every step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clay_sculptor.envs.clay_env import ClayEnv
from clay_sculptor.envs.shape import chamfer, emd, map_rmse

DEMOS = {
    "slab": dict(tool="roller", max_steps=6, demo=[("roller", [x, 0.0, 0.0, 0.008, 0.10]) for x in (-0.045, -0.015, 0.015, 0.045)]),
    "dent": dict(tool="press", max_steps=3, demo=[("press", [0.015, -0.010, 0.35, 0.011, 0.0])]),
    "x": dict(tool="stamp", max_steps=3, demo=[("stamp", [-0.005, 0.005, np.pi / 4 - 1e-6, 0.012, 0.0])]),
    "t": dict(tool="press", max_steps=7, demo=[("press", [0.035, -0.030, 0.0, 0.010, 0.0]), ("press", [0.035, 0.0, 0.0, 0.010, 0.0]),
                                               ("press", [0.035, 0.030, 0.0, 0.010, 0.0]), ("press", [0.000, 0.0, 0.0, 0.010, 0.0]),
                                               ("press", [-0.035, 0.0, 0.0, 0.010, 0.0])]),
    "combo": dict(tool=None, max_steps=5, demo=[("roller", [0.035, 0.0, 0.0, 0.007, 0.09]),
                                                ("stamp", [-0.020, -0.022, 0.3, 0.011, 0.0]),
                                                ("press", [-0.025, 0.038, 0.0, 0.010, 0.0])]),
    "steps": dict(tool="press", max_steps=5, demo=[("press", [-0.035, -0.035, np.pi / 4, 0.006, 0.0]),
                                                   ("press", [0.000, 0.000, np.pi / 4, 0.009, 0.0]),
                                                   ("press", [0.035, 0.035, np.pi / 4, 0.012, 0.0])]),
    "smiley": dict(tool="press", max_steps=10,
                   demo=[("press", [-0.050, y, 0.0, 0.012, 0.0]) for y in (-0.035, 0.035)]
                   + [("press", [-0.010 + 0.060 * np.cos(a), 0.060 * np.sin(a), a, 0.010, 0.0]) for a in np.radians([-60, -30, 0, 30, 60])]),
    "house": dict(tool=None, max_steps=8, demo=[("roller", [-0.055, -0.040, np.pi / 4, 0.006, 0.10]),
                                                ("roller", [-0.055, 0.040, -np.pi / 4, 0.006, 0.10]),
                                                ("stamp", [0.045, -0.045, 0.0, 0.010, 0.0]),
                                                ("press", [0.070, 0.055, 0.0, 0.012, 0.0])]),
}
MAIN = ("slab", "dent", "x")


@dataclass
class Target:
    name: str
    tool: str | None
    max_steps: int
    demo: list
    state: np.ndarray          # full simulator state of the demonstration's result
    nodes: np.ndarray
    hmap: np.ndarray           # (G, G) [m]
    surface: np.ndarray        # (ns, 3) surface vertex cloud
    top: np.ndarray            # (vx * vy, 3) top vertex layer
    initial_cost: float        # height-map RMSE of the untouched block against the target [m]


def make_target(env: ClayEnv, name: str) -> Target:
    spec = DEMOS[name]
    state = env.state0
    for tool, a in spec["demo"]:
        state = env.step(state, tool, np.array(a))
        env.oracle_calls -= 1
    nodes = env.nodes(state)
    h = env.geom.heightmap(nodes)
    h0 = env.geom.heightmap(env.nodes(env.state0))
    return Target(name, spec["tool"], spec["max_steps"], spec["demo"], state, nodes, h, env.geom.surface_points(nodes),
                  env.geom.top(nodes).reshape(-1, 3), float(map_rmse(h0, h)))


def shape_errors(env: ClayEnv, state: np.ndarray, target: Target) -> dict:
    """All shape metrics of a true simulator state against a target, in millimetres."""
    nodes = env.nodes(state)
    return dict(
        map_rmse_mm=1e3 * float(map_rmse(env.geom.heightmap(nodes), target.hmap)),
        chamfer_mm=1e3 * chamfer(env.geom.surface_points(nodes), target.surface),
        emd_mm=1e3 * emd(env.geom.top(nodes).reshape(-1, 3), target.top),
    )
