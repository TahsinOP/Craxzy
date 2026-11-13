"""The pitching machine: random speed, spin, release height and aim; and the target zone.

A pitch is drawn as (release point, speed, spin vector, aim point).  The machine is zeroed
in: the release direction is solved so that the ball, *with* its drag and spin, crosses
the nominal hitting plane x = strike_x at the aim point.  Spin therefore does not move
where the ball arrives, it changes how it gets there: backspin floats, topspin dives, and
a batter who has seen the first tenth of the flight cannot tell the two apart yet.

Surprises (M4) are stored with the pitch so that every controller sees the same ones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from batter.control.planner import crossing
from batter.estim.flight import FlightModel


@dataclass
class Pitches:
    x0: np.ndarray          # (n, 9) ball state at release
    speed: np.ndarray       # (n,) release speed
    aim: np.ndarray         # (n, 3) crossing point on the nominal hitting plane (without surprises)
    cross_t: np.ndarray     # (n,) time of that crossing [s]
    target: np.ndarray      # (n, 2) target centre on the ground
    # surprises: kind 0 = none, 1 = spin change, 2 = gust
    s_kind: np.ndarray      # (n,) int
    s_step: np.ndarray      # (n,) simulator step at which the surprise starts
    s_spin: np.ndarray      # (n, 3) new spin [rad/s] (kind 1)
    s_force: np.ndarray     # (n, 3) gust force [N] (kind 2), applied for ``s_len`` steps
    s_len: np.ndarray       # (n,) int

    def __len__(self) -> int:
        return len(self.speed)

    def take(self, idx: np.ndarray) -> "Pitches":
        return Pitches(**{k: v[idx] for k, v in self.__dict__.items()})


def _spin(rng: np.random.Generator, n: int, mag_range, tilt_deg: float) -> np.ndarray:
    mag = rng.uniform(*mag_range, n)
    sign = rng.choice([-1.0, 1.0], n)                     # backspin (+y for a ball moving to -x) or topspin
    tilt = np.deg2rad(tilt_deg) * np.sqrt(rng.uniform(0, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    axis = np.stack([np.sin(tilt) * np.cos(phi), np.cos(tilt), np.sin(tilt) * np.sin(phi)], axis=1)
    return (mag * sign)[:, None] * axis


def solve_release(fm: FlightModel, p0: np.ndarray, speed: np.ndarray, w: np.ndarray, aim: np.ndarray,
                  iters: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Release velocity (n, 3) of magnitude ``speed`` whose flight crosses x = aim_x at ``aim``.

    Fixed-point iteration on a virtual aim point; returns (v0, crossing time).
    """
    n = len(speed)
    virtual = aim.copy()
    t_cross = np.zeros(n)
    v0 = np.zeros((n, 3))
    for _ in range(iters):
        d = virtual - p0
        T = np.linalg.norm(d, axis=1) / speed
        v0 = d / T[:, None] - 0.5 * fm.g * T[:, None]
        v0 *= (speed / np.linalg.norm(v0, axis=1))[:, None]
        steps = int(np.ceil(1.6 * T.max() / fm.h)) + 5
        traj = fm.rollout(np.concatenate([p0, v0, w], axis=1), steps)
        for i in range(n):
            c = crossing(traj[:, i], aim[i, 0])
            if c is None:
                raise RuntimeError("pitch does not reach the hitting plane")
            t_cross[i] = c[0] * fm.h
            virtual[i, 1:] += aim[i, 1:] - c[1][1:]
    return v0, t_cross


def generate(fm: FlightModel, cfg: dict, n: int, seed: int, surprise: str = "none", speed: float | None = None,
             aim: np.ndarray | None = None) -> Pitches:
    """Draw ``n`` pitches.  ``surprise``: "none", "spin", "gust" or "mixed" (half and half).

    ``speed`` fixes the release speed (for the hit-rate-against-speed curves); ``aim`` (n, 3)
    overrides the aim points (M0 aims at a static bat).
    """
    pc, tc = cfg["pitch"], cfg["target"]
    rng = np.random.default_rng(seed)
    p0 = np.stack([np.full(n, pc["release_x"]), rng.uniform(*pc["release_y"], n), rng.uniform(*pc["release_z"], n)], axis=1)
    spd = rng.uniform(*pc["speed"], n) if speed is None else np.full(n, float(speed))
    w = _spin(rng, n, pc["spin"], pc["spin_tilt_deg"])
    aim_default = np.stack([np.full(n, pc["strike_x"]), rng.uniform(*pc["aim_y"], n), rng.uniform(*pc["aim_z"], n)], axis=1)
    aim = aim_default if aim is None else np.asarray(aim, float)
    v0, t_cross = solve_release(fm, p0, spd, w, aim)
    r = rng.uniform(*tc["range"], n)
    az = np.deg2rad(rng.uniform(*tc["azimuth_deg"], n))
    target = np.stack([r * np.cos(az), r * np.sin(az)], axis=1)

    # surprises happen between 25 % and 60 % of the flight: late enough that a plan exists,
    # early enough that the arm can still do something about it
    s_kind = np.zeros(n, int)
    if surprise == "spin":
        s_kind[:] = 1
    elif surprise == "gust":
        s_kind[:] = 2
    elif surprise == "mixed":
        s_kind[:] = rng.integers(1, 3, n)
    elif surprise != "none":
        raise ValueError(surprise)
    rs = np.random.default_rng(seed + 7919)                # separate stream: same pitches with and without surprises
    s_step = np.round(rs.uniform(0.25, 0.60, n) * t_cross / fm.h).astype(int)
    sc = cfg.get("surprise", {})
    s_spin = _spin(rs, n, sc.get("spin", [100.0, 220.0]), pc["spin_tilt_deg"])
    flip = np.sign(s_spin[:, 1]) == np.sign(w[:, 1])
    s_spin[flip] *= -1.0                                   # the new spin opposes the old one: float <-> dive
    fmag = rs.uniform(*sc.get("gust_force", [0.5, 1.2]), n)
    ang = rs.uniform(0, 2 * np.pi, n)
    s_force = fmag[:, None] * np.stack([np.zeros(n), np.cos(ang), np.sin(ang)], axis=1)    # across the flight path
    s_len = np.full(n, int(round(sc.get("gust_time", 0.10) / fm.h)))
    return Pitches(np.concatenate([p0, v0, w], axis=1), spd, aim, t_cross, target, s_kind, s_step, s_spin, s_force, s_len)
