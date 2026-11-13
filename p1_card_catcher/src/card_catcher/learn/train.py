"""Training loop for the sequence predictors (CPU, a few minutes)."""
from __future__ import annotations

import time

import numpy as np
import torch

from card_catcher.envs.flight import FlightData
from card_catcher.estim.ekf import EKFParams
from card_catcher.learn import data as ld
from card_catcher.learn.models import SeqPredictor, gaussian_nll


def targets(sd_r: torch.Tensor, sd_p: torch.Tensor, anchor: torch.Tensor, kind: str, horizon: int, scale: torch.Tensor):
    """Normalised targets y (B, K1, horizon, 3) and their validity (B, K1, horizon, 1).

    residual: y[k, j] = r_{k+j} / scale          (needs k + j <= K - 1)
    pure    : y[k, j] = (p_{k+j+1} - anchor_k) / scale[j]   (needs k + j + 1 <= K)
    """
    B, K1 = sd_p.shape[:2]
    src = sd_r if kind == "residual" else sd_p
    off = 0 if kind == "residual" else 1
    idx = torch.arange(K1)[:, None] + torch.arange(horizon)[None, :] + off          # (K1, horizon)
    ok = idx < src.shape[1]
    idx = idx.clamp(max=src.shape[1] - 1)
    y = src[:, idx]                                                                  # (B, K1, horizon, 3)
    if kind == "pure":
        y = y - anchor[:, :, None, :]
    return y / scale[None, None, :, None], ok[None, :, :, None]


def fit(train: FlightData, val: FlightData, prm: EKFParams, meas: dict, kind: str, horizon: int, seed: int,
        epochs: int = 30, batch: int = 64, lr: float = 2e-3, hidden: int = 128, n_noise: int = 3,
        verbose: bool = True) -> tuple[SeqPredictor, dict]:
    """Train one predictor.  ``n_noise`` independent measurement-noise realisations of
    the training tracks are used (noise augmentation); validation uses one."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sets = [ld.build(train, prm, meas, np.random.default_rng(seed * 1000 + i), kind) for i in range(n_noise)]
    sv = ld.build(val, prm, meas, np.random.default_rng(seed * 1000 + 999), kind)
    n_in = ld.N_FEAT_HYBRID if kind == "residual" else ld.N_FEAT_PURE
    model = SeqPredictor(n_in, horizon, hidden, kind)
    if kind == "residual":
        r = sets[0].r
        mask = np.arange(r.shape[1])[None] < train.cross_k[:, None]
        model.scale[:] = float(np.sqrt((r[mask] ** 2).mean()))
    else:
        model.scale[:] = torch.tensor(0.02 + 4.0 * train.dt * np.arange(1, horizon + 1), dtype=torch.float32)

    def tens(sd: ld.SeqData):
        return (torch.from_numpy(sd.feats), torch.from_numpy(sd.k_mask), torch.from_numpy(sd.r.astype(np.float32)),
                torch.from_numpy(sd.p.astype(np.float32)), torch.from_numpy(sd.anchor.astype(np.float32)))

    T = [tens(s) for s in sets]
    V = tens(sv)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = len(train)
    steps = epochs * int(np.ceil(N / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)

    def loss_on(t, ids):
        feats, kmask, r, p, anchor = (x[ids] for x in t)
        kmax = int(kmask.any(dim=0).nonzero().max()) + 1          # frames after the last forecast are not needed
        H, _ = model.encode(feats[:, :kmax])
        sel = kmask[:, :kmax]
        mu, lv = model.decode(H[sel])                             # decode only frames that issue a forecast
        y, ok = targets(r, p, anchor, kind, horizon, model.scale)
        return gaussian_nll(mu, lv, y[:, :kmax][sel], ok.expand(len(ids), -1, -1, -1)[:, :kmax][sel])

    hist, best, best_state, t0 = [], np.inf, None, time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(N)
        tot = 0.0
        for i in range(0, N, batch):
            ids = torch.from_numpy(perm[i:i + batch])
            loss = loss_on(T[rng.integers(n_noise)], ids)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(ids)
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_on(V, torch.arange(i, min(i + 128, len(val)))).item() for i in range(0, len(val), 128)]))
        hist.append((tot / N, vl))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"    [{kind} n={N} seed={seed}] epoch {ep:3d}  train {tot / N:.4f}  val {vl:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": hist, "best_val": best, "seconds": time.time() - t0}


def save(model: SeqPredictor, path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "kind": model.kind, "horizon": model.horizon, "hidden": model.hidden,
                "n_in": model.inp[0].in_features, "meta": meta}, path)


def load(path) -> tuple[SeqPredictor, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SeqPredictor(ck["n_in"], ck["horizon"], ck["hidden"], ck["kind"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck["meta"]
