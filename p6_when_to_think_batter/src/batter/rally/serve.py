"""The feeder: serves that bounce once on arm A's half and reach A's strike zone.

A serve is drawn as a release point at arm B's end, a landing point on A's half and a spin.
Its release velocity is solved through the flight model (aiming at the table surface) for a
grid of times to the bounce; a time is kept if the serve clears the net and, after its bounce,
crosses A's strike plane inside A's zone.  Serves with no such time are redrawn.
"""
from __future__ import annotations

import numpy as np

from batter.control.planner import Aimer
from batter.rally.flight import TableFlight

T_GRID = np.linspace(0.55, 1.00, 16)


def generate(tf: TableFlight, cfg: dict, n: int, seed: int) -> np.ndarray:
    """``n`` serve states (n, 9) in the world frame."""
    sc, rc, t = cfg["serve"], cfg["rally"], cfg["table"]
    rng = np.random.default_rng(seed)
    aimer = Aimer(tf.fm, ground_z=t["height"] + tf.radius)
    x_net = t["gap"] + t["length"] / 2
    z_net = t["height"] + t["net_height"] + tf.radius + rc["net_clearance"]
    x_plane = float(np.mean(rc["strike_x"]))
    out = []
    while len(out) < n:
        m = 4 * (n - len(out))
        p0 = np.stack([t["L"] - rng.uniform(*sc["release_back"], m), rng.uniform(*sc["release_y"], m),
                       rng.uniform(*sc["release_z"], m)], 1)
        land = np.stack([t["gap"] + rng.uniform(*sc["land_x"], m), rng.uniform(*sc["land_y"], m)], 1)
        mag = rng.uniform(*sc["spin"], m)
        axis = rng.normal(size=(m, 3))
        w = mag[:, None] * axis / np.linalg.norm(axis, axis=1, keepdims=True)
        order = rng.permuted(np.tile(np.arange(len(T_GRID)), (m, 1)), axis=1)
        for i in range(m):
            for T in T_GRID[order[i]]:
                v = aimer.solve(p0[i:i + 1], w[i:i + 1], land[i:i + 1], np.array([T]))[0]
                x0 = np.concatenate([p0[i], v, w[i]])
                path = tf.rollout(x0, 120)
                k_net = int(np.argmax(path[:, 0] <= x_net))
                if path[k_net, 2] < z_net:
                    continue
                k = np.flatnonzero(path[:, 0] <= x_plane)
                if k.size == 0:
                    continue
                c = path[k[0]]
                if rc["zone_z"][0] <= c[2] <= rc["zone_z"][1] and abs(c[1]) <= rc["zone_y"] and (np.diff(np.sign(path[: k[0], 5])) > 0).sum() == 1:
                    out.append(x0)
                    break
            if len(out) == n:
                break
    return np.array(out)
