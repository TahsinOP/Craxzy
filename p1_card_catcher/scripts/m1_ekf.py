"""M1: physics predictor + EKF.  Catch-point prediction error against time-to-catch.

Protocol
  * tuning set  : 500 throws per object (seed 100), used to fit the drag coefficient
                  (least squares on the simulator's own step map) and to tune the two
                  process-noise parameters by maximum likelihood of the innovations.
  * test sets   : the three M0 datasets (seeds 0, 1, 2; 1000 throws each); numbers are
                  mean +- std over the seeds.
Writes results/m1_params.json (used by M2-M4), results/m1_ekf.json, results/m1_error.png.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import chi2

from card_catcher import DATA, RESULTS, config
from card_catcher.envs.flight import FlightData, generate
from card_catcher.estim.ekf import EKFParams, fit_drag
from card_catcher.estim.runner import forecast_errors, innovation_nll, run_filter, simulate_measurements
from card_catcher.eval import plotstyle as ps

TAUS = np.round(np.arange(0.05, 0.651, 0.05), 3)
SIGMA_GRID = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
QKD_GRID = [0.0, 0.003, 0.01, 0.03]       # random-walk std of k_d per step, as a fraction of k_d


def before_crossing(D: FlightData) -> np.ndarray:
    K = D.p.shape[1] - 1
    return np.arange(K)[None, :] < D.cross_k[:, None]


def tune(obj: str, cfg: dict) -> dict:
    D = generate(obj, 500, cfg, seed=100)
    mask = before_crossing(D)
    kd = fit_drag(D.v, mask, D.dt)
    # spread of the per-throw best-fit coefficient -> prior std
    kd_i = np.array([fit_drag(D.v[i:i + 1], mask[i:i + 1], D.dt) for i in range(0, len(D), 5)])
    kd_std = float(max(np.std(kd_i), 0.05 * kd))
    z, valid = simulate_measurements(D.p, cfg["measurement"], np.random.default_rng(100))
    best = None
    table = []
    for sa in SIGMA_GRID:
        for qf in QKD_GRID:
            prm = EKFParams(sigma_a=sa, kd0=kd, kd0_std=kd_std, sigma_meas=cfg["measurement"]["sigma"], q_kd=qf * kd)
            nll = innovation_nll(run_filter(z, valid, D.dt, prm), valid & (np.arange(z.shape[1])[None] <= D.cross_k[:, None]))
            table.append((sa, qf, nll))
            if best is None or nll < best[0]:
                best = (nll, prm)
    prm = best[1]
    return {"kd0": prm.kd0, "kd0_std": prm.kd0_std, "sigma_a": prm.sigma_a, "q_kd": prm.q_kd,
            "sigma_meas": prm.sigma_meas, "innovation_nll": best[0], "kd_per_throw_std": float(np.std(kd_i)),
            "nll_table": table}


def evaluate(obj: str, cfg: dict, prm: EKFParams, seed: int, stride: int) -> dict:
    plane = config.Plane.from_cfg(cfg)
    D = FlightData.load(DATA / f"flight_{obj}_seed{seed}.npz")
    z, valid = simulate_measurements(D.p, cfg["measurement"], np.random.default_rng(1000 + seed))
    res = run_filter(z, valid, D.dt, prm)
    fe = forecast_errors(res, D.cross_k, D.cross_t, D.cross_p, D.dt, plane, prm.sigma_a, stride=stride, valid=valid)
    d2 = np.einsum("bi,bi->b", fe["err_uw"], np.linalg.solve(fe["cov_uw"], fe["err_uw"][:, :, None])[:, :, 0])
    rows = {}
    for tau in TAUS:
        sel = np.abs(fe["tau"] - tau) <= 0.5 * D.dt * stride + 1e-9
        if sel.sum() < 30:
            continue
        e = fe["err"][sel]
        rows[float(tau)] = {
            "n": int(sel.sum()), "rms_cm": float(np.sqrt(np.mean(e**2)) * 100), "median_cm": float(np.median(e) * 100),
            "p90_cm": float(np.quantile(e, 0.9) * 100), "rms_time_ms": float(np.sqrt(np.mean(fe["err_t"][sel] ** 2)) * 1e3),
            "cover50": float(np.mean(d2[sel] <= chi2.ppf(0.5, 2))), "cover90": float(np.mean(d2[sel] <= chi2.ppf(0.9, 2))),
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = config.load(overrides=args.set)
    params, results = {}, {}
    for obj in ("ball", "plate", "card"):
        t = tune(obj, cfg)
        params[obj] = {k: v for k, v in t.items() if k != "nll_table"}
        print(f"{obj}: fitted k_d = {t['kd0']:.4f} 1/m (per-throw std {t['kd_per_throw_std']:.4f}), "
              f"ML sigma_a = {t['sigma_a']} m/s^2, q_kd = {t['q_kd']:.2e}")
        prm = EKFParams(sigma_a=t["sigma_a"], kd0=t["kd0"], kd0_std=t["kd0_std"], sigma_meas=t["sigma_meas"], q_kd=t["q_kd"])
        per_seed = [evaluate(obj, cfg, prm, s, args.stride) for s in cfg["seeds"]]
        agg = {}
        for tau in per_seed[0]:
            if all(tau in r for r in per_seed):
                agg[tau] = {k: [float(np.mean([r[tau][k] for r in per_seed])), float(np.std([r[tau][k] for r in per_seed]))]
                            for k in per_seed[0][tau]}
        results[obj] = agg
        for tau in (0.4, 0.2, 0.1):
            if tau in agg:
                a = agg[tau]
                print(f"   {int(tau*1e3)} ms before catch: RMS {a['rms_cm'][0]:.2f} +- {a['rms_cm'][1]:.2f} cm, median "
                      f"{a['median_cm'][0]:.2f} cm, p90 {a['p90_cm'][0]:.2f} cm, timing RMS {a['rms_time_ms'][0]:.1f} ms, "
                      f"90% ellipse covers {a['cover90'][0]*100:.0f}%")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "m1_params.json").write_text(json.dumps(params, indent=2))
    (RESULTS / "m1_ekf.json").write_text(json.dumps({"measurement": cfg["measurement"], "taus": TAUS.tolist(), "results": results}, indent=2))

    ps.apply()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for obj in ("ball", "plate", "card"):
        taus = sorted(results[obj])
        x = np.array(taus) * 1e3
        m = np.array([results[obj][t]["rms_cm"][0] for t in taus])
        sd = np.array([results[obj][t]["rms_cm"][1] for t in taus])
        ax[0].plot(x, m, "-o", color=ps.OBJECT[obj], label=obj)
        ax[0].fill_between(x, m - sd, m + sd, color=ps.OBJECT[obj], alpha=0.2, lw=0)
        ax[0].annotate(f"{m[taus.index(0.2)]:.2f} cm", (200, m[taus.index(0.2)]), textcoords="offset points", xytext=(6, 6),
                       fontsize=8.5, color=ps.INK2)
        c = np.array([results[obj][t]["cover90"][0] for t in taus])
        ax[1].plot(x, c * 100, "-o", color=ps.OBJECT[obj], label=obj)
    ax[0].axhline(1.0, color=ps.INK2, lw=1, ls="--")
    ax[0].text(640, 1.08, "1 cm", color=ps.INK2, fontsize=8.5, ha="right")
    ax[0].axvline(200, color=ps.AXIS, lw=1)
    ax[0].set(yscale="log", xlabel="time before the catch [ms]", ylabel="catch-point error, RMS [cm]",
              title="Physics predictor + EKF: error vs time-to-catch")
    ax[0].invert_xaxis()
    ax[0].legend(loc="upper right")
    ax[1].axhline(90, color=ps.INK2, lw=1, ls="--")
    ax[1].set(xlabel="time before the catch [ms]", ylabel="errors inside the predicted 90% ellipse [%]", ylim=(0, 100),
              title="Does the EKF know how wrong it is?")
    ax[1].invert_xaxis()
    ax[1].legend(loc="lower left")
    fig.tight_layout()
    ps.save(fig, RESULTS / "m1_error.png")
    print("figure:", RESULTS / "m1_error.png")


if __name__ == "__main__":
    main()
