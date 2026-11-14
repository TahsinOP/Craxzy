"""Training and evaluation of the dynamics models (CPU).

Loss: mean squared error of the predicted change, in units of ``L0`` (10 mm): over
the free nodes for the node models, over the pixels for the U-Net.  A small
Gaussian noise is added to the input shape during training (as in Sanchez-Gonzalez
et al. 2020) so that the model sees slightly-off inputs like its own multi-step
predictions; the target stays the true next shape.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from clay_sculptor import DATA
from clay_sculptor.envs.shape import Geometry, map_rmse
from clay_sculptor.learn.models import L0, DynamicsModel, Ensemble

MODEL_DIR = DATA / "models"
KINDS = ("gnn", "structured", "unet")
SIZES = {"gnn": dict(hidden=48, steps=4), "structured": dict(hidden=48, steps=4), "unet": dict(base=32), "prior": {}}


class Transitions:
    """Flat view of a dataset split: one row per valid transition, plus the episode
    structure for multi-step evaluation."""

    def __init__(self, data: dict, episodes: slice, geom: Geometry):
        nodes, ok = data["nodes"][episodes], data["ok"][episodes]
        self.ep_nodes = torch.from_numpy(nodes)                    # (E, K+1, nx, ny, nz, 3)
        self.ep_tool = torch.from_numpy(data["tool"][episodes].astype(np.int64))
        self.ep_action = torch.from_numpy(data["action"][episodes])
        self.ep_ok = torch.from_numpy(ok)
        e, k = np.nonzero(ok)
        self.before = self.ep_nodes[e, k]
        self.after = self.ep_nodes[e, k + 1]
        self.tool = self.ep_tool[e, k]
        self.action = self.ep_action[e, k]
        self.step = torch.from_numpy(k)
        self.geom = geom
        self._maps = None

    def __len__(self) -> int:
        return len(self.tool)

    @property
    def maps(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._maps is None:
            with torch.no_grad():
                f = lambda x: torch.cat([self.geom.heightmap(x[i:i + 256]) for i in range(0, len(x), 256)])
                self._maps = (f(self.before), f(self.after))
        return self._maps


def split(data: dict, n_val: int = 100, n_test: int = 100) -> tuple[slice, slice, slice]:
    e = len(data["nodes"])
    return slice(0, e - n_val - n_test), slice(e - n_val - n_test, e - n_test), slice(e - n_test, e)


def _free_mask(geom: Geometry) -> torch.Tensor:
    return torch.from_numpy(geom.free.astype(np.float32)).reshape(1, *geom.nn, 1)


def loss_fn(model: DynamicsModel, before, after, tool, action, noise: float = 0.0) -> torch.Tensor:
    """Squared error of the predicted next shape in units of L0 (mean over free nodes or pixels)."""
    if model.state_kind == "map":
        x = before + noise * torch.randn_like(before) if noise > 0 else before
        return ((model(x, tool, action) - after) / L0).pow(2).mean()
    free = _free_mask(model.geom)
    x = before + noise * torch.randn_like(before) * free if noise > 0 else before
    err = ((model(x, tool, action) - after) / L0).pow(2) * free
    return err.sum() / (free.sum() * before.shape[0] * 3)


@torch.no_grad()
def predict(model: DynamicsModel, state: torch.Tensor, tool: torch.Tensor, action: torch.Tensor, batch: int = 32) -> torch.Tensor:
    return torch.cat([model(state[i:i + batch], tool[i:i + batch], action[i:i + batch]) for i in range(0, len(state), batch)])


@torch.no_grad()
def one_step_metrics(model: DynamicsModel, tr: Transitions, batch: int = 32) -> dict:
    """Per-sample errors of one-step predictions, in metres."""
    hb, ha = tr.maps
    state = hb if model.state_kind == "map" else tr.before
    pred = predict(model, state, tr.tool, tr.action, batch)
    hp = pred if model.state_kind == "map" else torch.cat([model.geom.heightmap(pred[i:i + 256]) for i in range(0, len(pred), 256)])
    out = dict(map_rmse=map_rmse(hp, ha).numpy(), tool=tr.tool.numpy(), step=tr.step.numpy(),
               change_rmse=map_rmse(hb, ha).numpy())
    if model.state_kind == "nodes":
        free = torch.from_numpy(model.geom.free)
        d = (pred - tr.after)[:, free]                                   # (n, nfree, 3)
        out["node_rmse"] = d.pow(2).sum(-1).mean(-1).sqrt().numpy()        # RMS node position error
        top_p, top_t = model.geom.top(pred), model.geom.top(tr.after)
        out["top_rmse"] = (top_p - top_t).pow(2).sum(-1).mean(dim=(-1, -2)).sqrt().numpy()
    return out


@torch.no_grad()
def multi_step_metrics(model: DynamicsModel, tr: Transitions, batch: int = 32) -> np.ndarray:
    """Open-loop rollouts from the first shape of every fully valid episode: the model
    is fed its own predictions.  Returns the height-map RMSE ``(E, K)`` [m] after 1..K steps."""
    good = tr.ep_ok.all(dim=1)
    nodes, tool, action = tr.ep_nodes[good], tr.ep_tool[good], tr.ep_action[good]
    k = tool.shape[1]
    hmap = lambda x: torch.cat([model.geom.heightmap(x[i:i + 256]) for i in range(0, len(x), 256)])
    state = hmap(nodes[:, 0]) if model.state_kind == "map" else nodes[:, 0]
    out = np.zeros((len(nodes), k))
    for j in range(k):
        state = predict(model, state, tool[:, j], action[:, j], batch)
        hp = state if model.state_kind == "map" else hmap(state)
        out[:, j] = map_rmse(hp, hmap(nodes[:, j + 1])).numpy()
    return out


def train(kind: str, geom: Geometry, tr: Transitions, va: Transitions, seed: int, epochs: int = 6, batch: int = 16,
          lr: float = 2e-3, noise: float = 2e-4, log=print, **size) -> tuple[DynamicsModel, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = {**SIZES[kind], **size}
    model = DynamicsModel(kind, geom, **kw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = len(tr)
    steps = epochs * (n // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.08)
    if model.state_kind == "map":
        xb, xa = tr.maps
    else:
        xb, xa = tr.before, tr.after
    hist, t0 = [], time.time()
    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for i in range(0, n - batch + 1, batch):
            idx = perm[i:i + batch]
            loss = loss_fn(model, xb[idx], xa[idx], tr.tool[idx], tr.action[idx], noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        val = float(np.mean(one_step_metrics(model, va)["map_rmse"]))
        hist.append(dict(epoch=ep + 1, train_loss=tot / (n // batch), val_map_rmse_mm=1e3 * val, wall_s=time.time() - t0))
        log(f"  [{kind} seed {seed}] epoch {ep + 1}/{epochs}  train loss {hist[-1]['train_loss']:.5f}  "
            f"val map RMSE {1e3 * val:.3f} mm  {time.time() - t0:.0f} s")
    info = dict(kind=kind, seed=seed, size=kw, epochs=epochs, batch=batch, lr=lr, noise=noise, n_train=n,
                n_params=model.n_params(), wall_s=time.time() - t0, history=hist)
    return model, info


def save(model: DynamicsModel, info: dict, name: str | None = None) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name or info['kind']}_seed{info['seed']}.pt"
    torch.save(dict(state=model.state_dict(), info=info), path)
    path.with_suffix(".json").write_text(json.dumps(info, indent=1))
    return path


def load(kind: str, seed: int, geom: Geometry, name: str | None = None) -> DynamicsModel:
    if kind == "prior":
        return DynamicsModel("prior", geom).eval()
    ck = torch.load(MODEL_DIR / f"{name or kind}_seed{seed}.pt", weights_only=False)
    model = DynamicsModel(kind, geom, **ck["info"]["size"])
    model.load_state_dict(ck["state"])
    return model.eval()


def load_ensemble(kind: str, seeds, geom: Geometry, name: str | None = None) -> Ensemble:
    return Ensemble([load(kind, s, geom, name) for s in seeds])
