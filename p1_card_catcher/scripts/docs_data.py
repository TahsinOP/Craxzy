"""Extra measurements used only by the documentation figures (results/docs_data.json).

    env -u PYTHONPATH .venv/bin/python scripts/docs_data.py

1. When along the flight does each replanning trigger fire?  (ball, 128 throws)
2. One card throw: the predicted catch point and its 90 % ellipse at several times
   before the catch, for the physics-only and the hybrid predictor.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from card_catcher import DATA, RESULTS, config
from card_catcher.config import Plane
from card_catcher.control.triggers import Drift, Learned, Periodic, Variance
from card_catcher.envs.catch import DEFAULT_PARAMS, DEFAULT_PREDICTOR, CatchEnv
from card_catcher.envs.flight import generate
from card_catcher.estim.predictor import make_predictor
from card_catcher.estim.runner import simulate_measurements
from card_catcher.learn import ppo


def trigger_times(cfg, n=128, seed=0) -> dict:
    obj = "ball"
    policy = ppo.TriggerPolicy()
    policy.load_state_dict(torch.load(DATA / "models" / f"trigger_{obj}.pt", map_location="cpu")["state"])
    trig = {"periodic 12.5 Hz": Periodic(12.5), "variance, δ = 15 mm": Variance(0.015), "drift, δ = 15 mm": Drift(0.015),
            "learned, λ = 0.003": Learned(policy, 0.003, np.random.default_rng(900 + seed))}
    D = generate(obj, n, cfg, seed=600 + seed)
    out = {}
    for name, t in trig.items():
        env = CatchEnv(obj, cfg, n, make_predictor(DEFAULT_PREDICTOR[obj], obj, n, cfg), DEFAULT_PARAMS[obj], seed=seed)
        obs = env.reset(D, np.arange(n))
        t.reset(n)
        taus, done = [], False
        while not done:
            fire = t(obs) & obs["active"] & obs["has_plan"]
            taus += obs["tau"][fire].tolist()
            obs, done = env.step(fire)
        o = env.outcome()
        out[name] = {"tau": taus, "catch": float(o["caught"].mean()), "plans": float(o["n_plans"].mean())}
        print(f"{name:22s} catch {out[name]['catch']*100:5.1f} %  plans/throw {out[name]['plans']:.1f}  "
              f"replans in the last 150 ms: {np.mean(np.array(taus) < 0.15) * 100:.0f} % of replans, "
              f"{np.sum(np.array(taus) < 0.15) / n:.2f} per throw", flush=True)
    return out


def ellipses(cfg, seed=0, n=64) -> dict:
    obj = "card"
    plane, h = Plane.from_cfg(cfg), cfg["sim"]["timestep"]
    D = generate(obj, n, cfg, seed=700 + seed)
    z, valid = simulate_measurements(D.p, cfg["measurement"], np.random.default_rng(seed))
    preds = {k: make_predictor(k, obj, n, cfg) for k in ("physics", "hybrid")}
    want = [0.4, 0.3, 0.2, 0.1]
    rec = {k: {} for k in preds}
    for k in range(z.shape[1]):
        for name, p in preds.items():
            p.step(z[:, k], valid[:, k])
        tau_true = D.cross_t - k * h
        for w in want:
            sel = np.abs(tau_true - w) < h / 2
            if sel.any():
                for name, p in preds.items():
                    fc = p.forecast()
                    for i in np.flatnonzero(sel & fc["found"]):
                        rec[name].setdefault(int(i), {})[str(w)] = {"uw": plane.coords(fc["p"][i]).tolist(), "cov": fc["cov_uw"][i].tolist()}
    # a throw with a typical hybrid error at 200 ms and all four snapshots for both predictors
    truth = plane.coords(D.cross_p)
    full = [i for i in rec["hybrid"] if all(str(w) in rec[m].get(i, {}) for w in want for m in rec)]
    err = np.array([np.linalg.norm(np.array(rec["hybrid"][i]["0.2"]["uw"]) - truth[i]) for i in full])
    i = full[int(np.argsort(err)[len(err) // 2])]
    return {"throw": int(i), "truth_uw": truth[i].tolist(), "physics": rec["physics"][i], "hybrid": rec["hybrid"][i],
            "u_range": list(plane.u_range), "w_range": list(plane.w_range)}


def main():
    cfg = config.load()
    out = {"trigger_times": trigger_times(cfg), "ellipses": ellipses(cfg)}
    (RESULTS / "docs_data.json").write_text(json.dumps(out))
    print("wrote", RESULTS / "docs_data.json")


if __name__ == "__main__":
    main()
