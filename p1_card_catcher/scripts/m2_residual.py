"""M2: learned aerodynamic residual (mean + variance) vs physics-only vs pure learning.

Predictors compared on the catch-point forecast (same measurements for all):
  physics : M1 EKF, drag-ballistic forecast, white process noise.
  hybrid  : EKF whose process model is physics + learned residual mean, with the learned
            variance (times one inflation factor kappa) as process noise; forecast = physics
            + decoded future residuals, with the decoded variances and an AR(1) error
            correlation phi.  kappa and phi are the only two numbers fitted after training,
            both on validation throws: kappa by maximum likelihood of the filter innovations
            (the filter treats the nowcast error as white, which it is not), phi by
            Yule-Walker on the standardised forecast errors.
  pure    : GRU on raw fixes -> future positions directly (no gravity, drag or integrator).

Training data: fresh throws (seed 200+), validation (seed 300); test = M0 datasets (seeds 0,1,2).
Writes data/models/*.pt, results/m2_residual.json, results/m2_residual.png.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy.stats import chi2, norm

from card_catcher import DATA, RESULTS, config
from card_catcher.envs.flight import FlightData, generate
from card_catcher.estim.ekf import EKFParams
from card_catcher.estim.physics import plane_crossing
from card_catcher.estim.runner import forecast_errors, innovation_nll, run_filter
from card_catcher.eval import plotstyle as ps
from card_catcher.learn import data as ld
from card_catcher.learn import train as lt

HORIZON = 112
TAUS = np.round(np.arange(0.05, 0.651, 0.05), 3)
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
PAD = 200


def ekf_params(obj: str) -> EKFParams:
    P = json.loads((RESULTS / "m1_params.json").read_text())[obj]
    return EKFParams(sigma_a=P["sigma_a"], kd0=P["kd0"], kd0_std=P["kd0_std"], sigma_meas=P["sigma_meas"], q_kd=P["q_kd"])


def get_model(obj, kind, n_train, seed, cfg, prm, steps):
    path = DATA / "models" / f"{obj}_{kind}_n{n_train}_s{seed}.pt"
    if path.exists():
        return lt.load(path)
    tr = generate(obj, n_train, cfg, seed=200 + seed)
    va = generate(obj, 500, cfg, seed=300)
    epochs = max(int(round(steps * 64 / n_train)), 1)
    model, info = lt.fit(tr, va, prm, cfg["measurement"], kind, HORIZON, seed=seed, epochs=epochs)
    meta = {"obj": obj, "kind": kind, "n_train": n_train, "seed": seed, "best_val": info["best_val"], "seconds": info["seconds"]}
    lt.save(model, path, meta)
    return model, meta


def pad_hold(a: np.ndarray, n: int) -> np.ndarray:
    """Extend (N, H, 3) to (N, n, 3) by holding the last value."""
    return np.concatenate([a, np.repeat(a[:, -1:], n - a.shape[1], axis=1)], axis=1)


def summarise(fe: dict, dt: float, stride: int) -> dict:
    d2 = np.einsum("bi,bi->b", fe["err_uw"], np.linalg.solve(fe["cov_uw"], fe["err_uw"][:, :, None])[:, :, 0])
    rows = {}
    for tau in TAUS:
        sel = np.abs(fe["tau"] - tau) <= 0.5 * dt * stride + 1e-9
        if sel.sum() < 30:
            continue
        e = fe["err"][sel]
        rows[float(tau)] = {"n": int(sel.sum()), "rms_cm": float(np.sqrt(np.mean(e**2)) * 100), "median_cm": float(np.median(e) * 100),
                            "cover90": float(np.mean(d2[sel] <= chi2.ppf(0.9, 2)))}
    return rows


def hybrid_outputs(model, sd: ld.SeqData):
    """Hidden states and the nowcast (j = 0) residual mean/std for every frame."""
    with torch.no_grad():
        H, _ = model.encode(torch.from_numpy(sd.feats))
        mu, sig = model.decode_physical(H)
    return H, mu.numpy(), sig.numpy()


def fit_phi(D: FlightData, sd: ld.SeqData, mu: np.ndarray, sig: np.ndarray) -> float:
    """AR(1) coefficient (Yule-Walker, lag 1) of the standardised forecast errors along the horizon."""
    N, K1 = sd.k_mask.shape
    num = den = 0.0
    for k in range(0, K1, 4):
        act = np.flatnonzero(sd.k_mask[:, k])
        if act.size == 0:
            continue
        nj = min(HORIZON, sd.r.shape[1] - k)
        e = (sd.r[act, k:k + nj] - mu[act, k, :nj]) / sig[act, k, :nj]
        num += (e[:, 1:] * e[:, :-1]).sum()
        den += (e[:, :-1] ** 2).sum()
    return float(np.clip(num / den, 0.0, 0.999))


KAPPAS = [1.0, 2.0, 4.0, 8.0, 16.0]


def fit_kappa(D: FlightData, sd: ld.SeqData, mu: np.ndarray, sig: np.ndarray, prm_h: EKFParams) -> float:
    """Inflation of the learned process noise that maximises the innovation likelihood."""
    mask = sd.valid & (np.arange(sd.z.shape[1])[None] <= D.cross_k[:, None])
    nll = [innovation_nll(run_filter(sd.z, sd.valid, D.dt, prm_h, r=mu[:, :-1, 0], r_var=k * sig[:, :-1, 0] ** 2), mask) for k in KAPPAS]
    return float(KAPPAS[int(np.argmin(nll))])


def residual_calibration(sd: ld.SeqData, mu, sig) -> dict:
    """NLL and central-interval coverage of the residual forecast, by horizon band."""
    N, K1 = sd.k_mask.shape
    bands = {"0-80 ms": (0, 10), "80-240 ms": (10, 30), "240-480 ms": (30, 60)}
    zs = {b: [] for b in bands}
    nll = {b: [] for b in bands}
    for k in range(0, K1, 3):
        act = np.flatnonzero(sd.k_mask[:, k])
        if act.size == 0:
            continue
        for b, (j0, j1) in bands.items():
            j1c = min(j1, sd.r.shape[1] - k)
            if j1c <= j0:
                continue
            s = sig[act, k, j0:j1c]
            zz = (sd.r[act, k + j0:k + j1c] - mu[act, k, j0:j1c]) / s
            zs[b].append(zz.ravel())
            nll[b].append((0.5 * zz**2 + np.log(s) + 0.5 * np.log(2 * np.pi)).ravel())
    out = {}
    for b in bands:
        zz = np.concatenate(zs[b])
        out[b] = {"nll": float(np.concatenate(nll[b]).mean()), "z_std": float(zz.std()),
                  "coverage": [float(np.mean(np.abs(zz) <= norm.ppf(0.5 + lv / 2))) for lv in LEVELS]}
    return out


def eval_hybrid(model, obj, cfg, prm, seed, stride):
    plane = config.Plane.from_cfg(cfg)
    D = FlightData.load(DATA / f"flight_{obj}_seed{seed}.npz")
    sd = ld.build(D, prm, cfg["measurement"], np.random.default_rng(1000 + seed), "residual")
    H, mu, sig = hybrid_outputs(model, sd)
    # hybrid EKF: drag fixed at the fitted value (the residual owns every deviation from it)
    prm_h = EKFParams(sigma_a=prm.sigma_a, kd0=prm.kd0, kd0_std=0.0, sigma_meas=prm.sigma_meas, q_kd=0.0)
    Dv = generate(obj, 500, cfg, seed=300)
    sv = ld.build(Dv, prm, cfg["measurement"], np.random.default_rng(77), "residual")
    _, muv, sigv = hybrid_outputs(model, sv)
    phi = fit_phi(Dv, sv, muv, sigv)
    kappa = fit_kappa(Dv, sv, muv, sigv, prm_h)
    res = run_filter(sd.z, sd.valid, D.dt, prm_h, r=mu[:, :-1, 0], r_var=kappa * sig[:, :-1, 0] ** 2)
    kw = dict(stride=stride, valid=sd.valid, r_future=lambda k: pad_hold(mu[:, k], PAD), sigma_future=lambda k: pad_hold(sig[:, k], PAD))
    fe = forecast_errors(res, D.cross_k, D.cross_t, D.cross_p, D.dt, plane, None, phi=phi, **kw)
    fe_white = forecast_errors(res, D.cross_k, D.cross_t, D.cross_p, D.dt, plane, None, phi=0.0, **kw)
    return {"rows": summarise(fe, D.dt, stride), "rows_white": summarise(fe_white, D.dt, stride), "phi": phi, "kappa": kappa,
            "calibration": residual_calibration(sd, mu, sig)}


def eval_physics(obj, cfg, prm, seed, stride):
    plane = config.Plane.from_cfg(cfg)
    D = FlightData.load(DATA / f"flight_{obj}_seed{seed}.npz")
    sd = ld.build(D, prm, cfg["measurement"], np.random.default_rng(1000 + seed), "residual")
    res = run_filter(sd.z, sd.valid, D.dt, prm)
    fe = forecast_errors(res, D.cross_k, D.cross_t, D.cross_p, D.dt, plane, prm.sigma_a, stride=stride, valid=sd.valid)
    return {"rows": summarise(fe, D.dt, stride)}


def eval_pure(model, obj, cfg, prm, seed, stride):
    plane = config.Plane.from_cfg(cfg)
    D = FlightData.load(DATA / f"flight_{obj}_seed{seed}.npz")
    sd = ld.build(D, prm, cfg["measurement"], np.random.default_rng(1000 + seed), "pure")
    with torch.no_grad():
        H, _ = model.encode(torch.from_numpy(sd.feats))
    out = {k: [] for k in ("tau", "err", "err_uw", "cov_uw")}
    for k in range(0, int(D.cross_k.max()) + 1, stride):
        act = np.flatnonzero(sd.k_mask[:, k])
        if act.size == 0:
            continue
        with torch.no_grad():
            mu, sig = model.decode_physical(H[act, k])
        mu, sig = mu.numpy(), sig.numpy()
        path = np.concatenate([sd.anchor[act, k][:, None], sd.anchor[act, k][:, None] + mu], axis=1)   # (n, H+1, 3)
        c = plane_crossing(path, plane.c0, plane.n, D.dt)
        ok = c["found"]
        idx = act[ok]
        j = c["k"][ok]                                   # crossing between path samples j and j+1
        vel = (path[ok][np.arange(ok.sum()), j + 1] - path[ok][np.arange(ok.sum()), j]) / D.dt
        var = sig[ok][np.arange(ok.sum()), np.minimum(j, HORIZON - 1)] ** 2      # position variance at the crossing step
        nv = vel @ plane.n
        J = np.eye(3)[None] - vel[:, :, None] * plane.n[None, None, :] / nv[:, None, None]
        cov = J @ (var[:, :, None] * np.eye(3)[None]) @ J.transpose(0, 2, 1)
        d = c["p"][ok] - D.cross_p[idx]
        out["tau"].append(D.cross_t[idx] - k * D.dt)
        out["err"].append(np.linalg.norm(d, axis=1))
        out["err_uw"].append(d @ plane.basis.T)
        out["cov_uw"].append(plane.basis[None] @ cov @ plane.basis.T[None])
    fe = {k: np.concatenate(v) for k, v in out.items()}
    return {"rows": summarise(fe, D.dt, stride)}


def agg(per_seed: list[dict]) -> dict:
    out = {}
    for tau in per_seed[0]:
        if all(tau in r for r in per_seed):
            out[tau] = {k: [float(np.mean([r[tau][k] for r in per_seed])), float(np.std([r[tau][k] for r in per_seed]))]
                        for k in per_seed[0][tau]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="*", default=["card", "plate"])
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=1200, help="gradient steps per model (batch 64)")
    ap.add_argument("--sweep", nargs="*", type=int, default=[250, 1000], help="extra training-set sizes for the card (seed 0)")
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(8)
    cfg = config.load()
    out = {"n_train": args.n_train, "steps": args.steps, "objects": {}}
    for obj in args.objects:
        prm = ekf_params(obj)
        seeds = cfg["seeds"] if obj == "card" else cfg["seeds"][:1]
        R = {"physics": [], "hybrid": [], "hybrid_white": [], "pure": []}
        calib, phis, kappas = [], [], []
        for s in seeds:
            mh, _ = get_model(obj, "residual", args.n_train, s, cfg, prm, args.steps)
            mp, _ = get_model(obj, "pure", args.n_train, s, cfg, prm, args.steps)
            eh = eval_hybrid(mh, obj, cfg, prm, s, args.stride)
            R["physics"].append(eval_physics(obj, cfg, prm, s, args.stride)["rows"])
            R["hybrid"].append(eh["rows"])
            R["hybrid_white"].append(eh["rows_white"])
            R["pure"].append(eval_pure(mp, obj, cfg, prm, s, args.stride)["rows"])
            calib.append(eh["calibration"])
            phis.append(eh["phi"])
            kappas.append(eh["kappa"])
            print(f"{obj} seed {s}: phi = {eh['phi']:.3f}, kappa = {eh['kappa']:g}; RMS @200 ms  physics {R['physics'][-1][0.2]['rms_cm']:.2f}  "
                  f"pure {R['pure'][-1][0.2]['rms_cm']:.2f}  hybrid {R['hybrid'][-1][0.2]['rms_cm']:.2f} cm", flush=True)
        res = {m: agg(v) for m, v in R.items()}
        res["phi"] = [float(np.mean(phis)), float(np.std(phis))]
        res["phi_per_seed"], res["kappa_per_seed"] = phis, kappas
        res["calibration"] = {b: {"nll": [float(np.mean([c[b]["nll"] for c in calib])), float(np.std([c[b]["nll"] for c in calib]))],
                                  "z_std": float(np.mean([c[b]["z_std"] for c in calib])),
                                  "coverage": np.mean([c[b]["coverage"] for c in calib], axis=0).tolist()} for b in calib[0]}
        res["seeds"] = seeds
        if obj == "card" and args.sweep:
            sweep = {}
            for n in args.sweep:
                mh, _ = get_model(obj, "residual", n, 0, cfg, prm, args.steps)
                mp, _ = get_model(obj, "pure", n, 0, cfg, prm, args.steps)
                sweep[n] = {"hybrid": eval_hybrid(mh, obj, cfg, prm, 0, args.stride)["rows"][0.2]["rms_cm"],
                            "pure": eval_pure(mp, obj, cfg, prm, 0, args.stride)["rows"][0.2]["rms_cm"]}
                print(f"card sweep n={n}: {sweep[n]}", flush=True)
            sweep[args.n_train] = {"hybrid": R["hybrid"][0][0.2]["rms_cm"], "pure": R["pure"][0][0.2]["rms_cm"]}
            res["sweep_rms_cm_at_200ms"] = sweep
        out["objects"][obj] = res
        (RESULTS / "m2_residual.json").write_text(json.dumps(out, indent=2))
    plot(out)


def plot(out: dict) -> None:
    ps.apply()
    import matplotlib.pyplot as plt

    res = out["objects"]["card"]
    names = {"physics": "physics only (EKF)", "pure": "pure learning (GRU)", "hybrid": "hybrid (physics + residual)"}
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))
    a = ax[0, 0]
    for m in ("physics", "pure", "hybrid"):
        taus = sorted(float(t) for t in res[m])
        y = np.array([res[m][t]["rms_cm"][0] for t in taus])
        sd = np.array([res[m][t]["rms_cm"][1] for t in taus])
        a.plot(np.array(taus) * 1e3, y, "-o", color=ps.METHOD[m], label=names[m])
        a.fill_between(np.array(taus) * 1e3, y - sd, y + sd, color=ps.METHOD[m], alpha=0.2, lw=0)
    a.set(xlabel="time before the catch [ms]", ylabel="catch-point error, RMS [cm]", title="Card: prediction error vs time-to-catch", ylim=(0, None))
    a.invert_xaxis()
    a.legend(loc="upper right")

    a = ax[0, 1]
    a.plot([0, 100], [0, 100], color=ps.INK2, lw=1, ls="--", label="ideal")
    for (b, c), col in zip(res["calibration"].items(), (ps.AQUA, ps.VIOLET, ps.MAGENTA)):
        a.plot(LEVELS * 100, np.array(c["coverage"]) * 100, "-o", color=col, label=f"horizon {b}  (NLL {c['nll'][0]:.2f})")
    a.set(xlabel="nominal central interval [%]", ylabel="residuals inside the interval [%]",
          title="Residual model: is the predicted variance calibrated?", xlim=(0, 100), ylim=(0, 100))
    a.legend(loc="upper left")

    a = ax[1, 0]
    if "sweep_rms_cm_at_200ms" in res:
        sw = res["sweep_rms_cm_at_200ms"]
        ns = sorted(int(n) for n in sw)
        get = lambda n, m: sw[n][m] if n in sw else sw[str(n)][m]
        for m in ("pure", "hybrid"):
            a.plot(ns, [get(n, m) for n in ns], "-o", color=ps.METHOD[m], label=names[m])
        a.axhline(res["physics"][0.2 if 0.2 in res["physics"] else "0.2"]["rms_cm"][0], color=ps.METHOD["physics"], label=names["physics"])
        a.set(xscale="log", xlabel="training throws", ylabel="RMS error 200 ms before the catch [cm]",
              title="Card: data efficiency (seed 0)", ylim=(0, None))
        a.legend(loc="center right")

    a = ax[1, 1]
    a.axhline(90, color=ps.INK2, lw=1, ls="--")
    for m, lab in (("physics", names["physics"]), ("pure", names["pure"]), ("hybrid", names["hybrid"])):
        taus = sorted(float(t) for t in res[m])
        a.plot(np.array(taus) * 1e3, [res[m][t]["cover90"][0] * 100 for t in taus], "-o", color=ps.METHOD[m], label=lab)
    taus = sorted(float(t) for t in res["hybrid_white"])
    a.plot(np.array(taus) * 1e3, [res["hybrid_white"][t]["cover90"][0] * 100 for t in taus], "--", color=ps.METHOD["hybrid"], lw=1.5,
           label="hybrid, white-noise forecast (no AR(1))")
    a.set(xlabel="time before the catch [ms]", ylabel="catch points inside the 90% ellipse [%]", ylim=(0, 100),
          title="Card: catch-point covariance coverage")
    a.invert_xaxis()
    a.legend(loc="lower left")
    fig.tight_layout()
    ps.save(fig, RESULTS / "m2_residual.png")
    print("figure:", RESULTS / "m2_residual.png")


if __name__ == "__main__":
    main()
