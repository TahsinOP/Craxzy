"""Free-flight datasets, simulated with ``mujoco.batch`` (multi-world CPU).

A dataset holds N *accepted* throws.  A throw is accepted when its simulated
trajectory crosses the catch plane inside the reachable window, not too early
(the arm needs time) and transversally enough for the crossing to be
well-conditioned.  Rejection is on the true trajectory, so the accepted set is
exactly "throws the arm could in principle catch".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from mujoco import batch as mjbatch

from card_catcher.config import Plane
from card_catcher.estim.physics import plane_crossing
from card_catcher.models.scene import build_flight_model
from card_catcher.models.throws import sample_throws


@dataclass
class FlightData:
    """N trajectories of K+1 samples at spacing dt."""

    obj: str
    dt: float
    p: np.ndarray        # (N, K+1, 3) position (world)
    quat: np.ndarray     # (N, K+1, 4) attitude
    v: np.ndarray        # (N, K+1, 3) linear velocity (world)
    w: np.ndarray        # (N, K+1, 3) angular velocity (body)
    cross_t: np.ndarray  # (N,) time of the plane crossing [s]
    cross_p: np.ndarray  # (N, 3) crossing point
    cross_v: np.ndarray  # (N, 3) velocity at the crossing
    cross_k: np.ndarray  # (N,) index of the last sample before the crossing

    def __len__(self) -> int:
        return self.p.shape[0]

    def subset(self, idx) -> "FlightData":
        f = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return FlightData(**f)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.__dict__)

    @staticmethod
    def load(path: Path) -> "FlightData":
        z = np.load(path, allow_pickle=False)
        return FlightData(**{k: (str(z[k]) if k == "obj" else (float(z[k]) if k == "dt" else z[k])) for k in z.files})


def simulate_flights(model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray, nstep: int,
                     nthread: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Roll N free flights forward.  Returns qpos (N, nstep+1, 7), qvel (N, nstep+1, 6)."""
    n = qpos.shape[0]
    b = mjbatch.Batch(model, nsim=n, nthread=nthread)
    b.reset(stage=int(mujoco.mjtStage.mjSTAGE_NONE))
    b.bind("qpos")[:] = qpos
    b.bind("qvel")[:] = qvel
    rec = b.rollout(nstep, record=("qpos", "qvel"))
    Q = np.concatenate([qpos[:, None], rec["qpos"]], axis=1)
    V = np.concatenate([qvel[:, None], rec["qvel"]], axis=1)
    return Q, V


def generate(obj: str, n: int, cfg: dict, seed: int, t_max: float = 1.2, min_cross_t: float = 0.35,
             min_normal_speed: float = 0.5, chunk: int = 2048, verbose: bool = False) -> FlightData:
    """Generate n accepted throws for ``obj`` (rejection sampling on the true flight)."""
    plane = Plane.from_cfg(cfg)
    model = build_flight_model(obj, cfg["sim"])
    dt = model.opt.timestep
    nstep = int(round(t_max / dt))
    rng = np.random.default_rng(seed)
    keep: list[dict[str, np.ndarray]] = []
    total, got = 0, 0
    while got < n:
        th = sample_throws(obj, chunk, plane, dt, rng)
        Q, V = simulate_flights(model, th["qpos"], th["qvel"], nstep)
        P, VL = Q[:, :, :3], V[:, :, :3]
        c = plane_crossing(P, plane.c0, plane.n, dt, VL)
        ok = c["found"] & plane.inside(c["p"]) & (c["t"] >= min_cross_t) & (-(c["v"] @ plane.n) >= min_normal_speed)
        ok &= np.isfinite(Q).all(axis=(1, 2))
        total += chunk
        idx = np.flatnonzero(ok)[: n - got]
        keep.append(dict(p=P[idx], quat=Q[idx, :, 3:], v=VL[idx], w=V[idx, :, 3:], cross_t=c["t"][idx],
                         cross_p=c["p"][idx], cross_v=c["v"][idx], cross_k=c["k"][idx]))
        got += len(idx)
        if verbose:
            print(f"  {obj}: accepted {got}/{n} (acceptance so far {ok.mean():.2f})")
    cat = {k: np.concatenate([d[k] for d in keep]) for k in keep[0]}
    return FlightData(obj=obj, dt=dt, **cat)
