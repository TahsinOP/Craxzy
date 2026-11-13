"""M0: throw datasets (mujoco.batch, fluid forces on) and evidence that the card flutters.

Writes data/flight_<obj>_seed<S>.npz (N accepted throws each), results/m0_dataset.json
and results/m0_flight.png.
"""
from __future__ import annotations

import argparse
import json
import time

import mujoco
import numpy as np

from card_catcher import DATA, RESULTS, config
from card_catcher.envs.flight import generate, simulate_flights
from card_catcher.estim.physics import plane_crossing, residual_accel, rollout_nominal
from card_catcher.eval import plotstyle as ps
from card_catcher.models.scene import build_flight_model
from card_catcher.models.throws import THROWS


def body_z(quat: np.ndarray) -> np.ndarray:
    """World-frame body z-axis (the card/plate normal) for quaternions (..., 4)."""
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=-1)


def timestep_convergence(obj: str, cfg: dict, D, n: int = 64) -> dict:
    """Position difference between the working timestep and a 0.5 ms reference, same launches."""
    out = {}
    q0 = np.concatenate([D.p[:n, 0], D.quat[:n, 0]], axis=1)
    v0 = np.concatenate([D.v[:n, 0], D.w[:n, 0]], axis=1)
    T, T1 = 0.64, 0.192       # both are multiples of 16 ms
    ref_dt = 0.0005
    mref = build_flight_model(obj, dict(cfg["sim"], timestep=ref_dt))
    Qr, _ = simulate_flights(mref, q0, v0, int(round(T / ref_dt)))
    for dt in (0.008, 0.016):
        m = build_flight_model(obj, dict(cfg["sim"], timestep=dt))
        Q, _ = simulate_flights(m, q0, v0, int(round(T / dt)))
        k = int(round(dt / ref_dt))
        err = np.linalg.norm(Q[:, :, :3] - Qr[:, ::k][:, : Q.shape[1], :3], axis=2)
        # the O(h) part of the difference is the known symplectic-Euler offset g h t / 2 (identical for
        # every throw and reproduced exactly by the predictor); remove it to expose the aerodynamic part
        t = np.arange(Q.shape[1]) * dt
        off = np.zeros((Q.shape[1], 3))
        off[:, 2] = -0.5 * 9.81 * (dt - ref_dt) * t
        err_aero = np.linalg.norm(Q[:, :, :3] - off - Qr[:, ::k][:, : Q.shape[1], :3], axis=2)
        out[f"{dt*1e3:.0f}ms"] = {
            "median_cm_at_0.19s": float(np.median(err[:, int(round(T1 / dt))]) * 100),
            "median_cm_at_0.64s": float(np.median(err[:, -1]) * 100),
            "median_cm_at_0.19s_minus_gravity_offset": float(np.median(err_aero[:, int(round(T1 / dt))]) * 100),
            "median_cm_at_0.64s_minus_gravity_offset": float(np.median(err_aero[:, -1]) * 100),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = config.load(overrides=args.set)
    plane = config.Plane.from_cfg(cfg)
    summary = {"n": args.n, "objects": {}}
    data0 = {}
    for obj in ("ball", "plate", "card"):
        rows = []
        for seed in cfg["seeds"]:
            t0 = time.time()
            D = generate(obj, args.n, cfg, seed=seed)
            D.save(DATA / f"flight_{obj}_seed{seed}.npz")
            el = time.time() - t0
            if seed == cfg["seeds"][0]:
                data0[obj] = D
            kd = THROWS[obj].kd_nominal
            Pn, _ = rollout_nominal(D.p[:, 0], D.v[:, 0], kd, D.dt, D.p.shape[1] - 1)
            cn = plane_crossing(Pn, plane.c0, plane.n, D.dt)
            miss = np.linalg.norm(cn["p"] - D.cross_p, axis=1)[cn["found"]]
            r = residual_accel(D.v, kd, D.dt)
            rr = np.concatenate([r[i, : D.cross_k[i]] for i in range(len(D))])
            nz = body_z(D.quat)[:, :, 2]
            flips = np.mean([(np.diff(np.sign(nz[i, : D.cross_k[i]])) != 0).sum() for i in range(len(D))])
            rows.append({
                "seed": seed, "gen_seconds": el, "launch_speed": float(np.linalg.norm(D.v[:, 0], axis=1).mean()),
                "cross_speed": float(np.linalg.norm(D.cross_v, axis=1).mean()), "flight_time": float(D.cross_t.mean()),
                "nominal_miss_cm_median": float(np.median(miss) * 100), "residual_rms": float(np.sqrt((rr**2).sum(1).mean())),
                "normal_flips_per_throw": float(flips), "max_ang_speed": float(np.linalg.norm(D.w, axis=2).max(axis=1).mean()),
            })
        agg = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in rows[0] if k != "seed"}
        summary["objects"][obj] = {"per_seed": rows, "mean_std": agg, "kd_nominal": THROWS[obj].kd_nominal,
                                   "timestep_convergence": timestep_convergence(obj, cfg, data0[obj])}
        print(obj, json.dumps(agg), json.dumps(summary["objects"][obj]["timestep_convergence"]))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "m0_dataset.json").write_text(json.dumps(summary, indent=2))

    # ---- figure: trajectories, card attitude (flutter) and aerodynamic acceleration
    ps.apply()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.2))
    a = ax[0, 0]
    for obj in ("ball", "plate", "card"):
        D = data0[obj]
        for i in range(40):
            k = D.cross_k[i] + 2
            a.plot(D.p[i, :k, 0], D.p[i, :k, 2], color=ps.OBJECT[obj], lw=0.7, alpha=0.45, label=obj if i == 0 else None)
    w = np.array(plane.w_range)
    seg = plane.c0[None] + w[:, None] * plane.ew[None]
    a.plot(seg[:, 0], seg[:, 2], color=ps.INK, lw=2.5, label="catch plane")
    a.set(xlabel="x [m]", ylabel="z [m]", title="40 throws per object (side view)")
    a.legend(loc="upper left")
    a.set_aspect("equal")

    D = data0["card"]
    a = ax[0, 1]
    idx = np.argsort(D.cross_t)[-4:]
    cols = [ps.BLUE, ps.ORANGE, ps.AQUA, ps.YELLOW]
    for c, i in zip(cols, idx):
        k = D.cross_k[i] + 1
        a.plot(np.arange(k) * D.dt, body_z(D.quat[i, :k])[:, 2], color=c, label=f"throw {i}")
    a.axhline(0, color=ps.AXIS, lw=0.8)
    a.set(xlabel="time since launch [s]", ylabel="card normal, world-z component", title="Card attitude: tumbling and flutter", ylim=(-1.05, 1.05))
    a.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.08), columnspacing=1.0, handlelength=1.2)

    a = ax[1, 0]
    kd = THROWS["card"].kd_nominal
    r = residual_accel(D.v, kd, D.dt)
    for c, i in zip(cols, idx):
        k = D.cross_k[i]
        a.plot(np.arange(k) * D.dt, np.linalg.norm(r[i, :k], axis=1), color=c, label=f"throw {i}")
    a.set(xlabel="time since launch [s]", ylabel="|residual acceleration| [m/s²]",
          title="Card: acceleration the drag-ballistic model misses")
    a.legend(ncol=4, loc="upper right", columnspacing=1.0, handlelength=1.2)

    a = ax[1, 1]
    for obj in ("ball", "plate", "card"):
        Dd = data0[obj]
        Pn, _ = rollout_nominal(Dd.p[:, 0], Dd.v[:, 0], THROWS[obj].kd_nominal, Dd.dt, Dd.p.shape[1] - 1)
        cn = plane_crossing(Pn, plane.c0, plane.n, Dd.dt)
        miss = np.sort(np.linalg.norm(cn["p"] - Dd.cross_p, axis=1)[cn["found"]]) * 100
        a.plot(miss, np.linspace(0, 1, len(miss)), color=ps.OBJECT[obj], label=obj)
    a.set(xscale="log", xlabel="catch-point miss of the nominal model from the true launch state [cm]",
          ylabel="fraction of throws", title="How wrong pure physics is, with perfect initial state")
    a.legend(loc="upper left")
    fig.tight_layout()
    ps.save(fig, RESULTS / "m0_flight.png")
    print("figure:", RESULTS / "m0_flight.png")


if __name__ == "__main__":
    main()
