"""M2: prediction error of the dynamics models, one step and multi-step, and speed.

    env -u PYTHONPATH .venv/bin/python scripts/m2_eval.py

Held-out episodes (never seen in training or validation).  All errors are
height-map RMSE in mm unless noted, mean +- std over model seeds.  Baselines next
to every learned model: "no change" (predict the current shape) and the geometric
prior alone.
"""
import argparse
import json
import time

import numpy as np
import torch

from clay_sculptor import DATA, RESULTS, config
from clay_sculptor.eval import plots
from clay_sculptor.learn import data as D
from clay_sculptor.learn import train as T
from clay_sculptor.models.scene import TOOLS

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
ap.add_argument("--data", default="main")
ap.add_argument("--shift", nargs="*", default=["young_lo", "young_hi", "yield_lo", "yield_hi"])
ap.add_argument("--threads", type=int, default=8)
args = ap.parse_args()
torch.set_num_threads(args.threads)

cfg = config.load()
env = cfg.env(nsim=2)
geom = env.geom
data = D.load(DATA / f"transitions_{args.data}.npz")
tr_s, va_s, te_s = T.split(data)
te = T.Transitions(data, te_s, geom)
print(f"test: {len(te)} transitions from {te_s.stop - te_s.start} episodes", flush=True)
ms = lambda a: [float(np.mean(a)), float(np.std(a))]
out = dict(n_test=len(te), seeds=args.seeds, one_step={}, multi_step={}, speed={}, shift={}, training={})

# baseline: predict no change
hb, ha = te.maps
no_change = T.map_rmse(hb, ha).numpy()
out["one_step"]["no_change"] = dict(map_rmse_mm=[1e3 * float(no_change.mean()), 0.0],
                                    by_tool={t: 1e3 * float(no_change[te.tool.numpy() == i].mean()) for i, t in enumerate(TOOLS)})
good = te.ep_ok.all(dim=1)
hmap = lambda x: torch.cat([geom.heightmap(x[i:i + 256]) for i in range(0, len(x), 256)])
h_first = hmap(te.ep_nodes[good][:, 0])
out["multi_step"]["no_change"] = [[1e3 * float(T.map_rmse(h_first, hmap(te.ep_nodes[good][:, j + 1])).mean()), 0.0]
                                  for j in range(te.ep_tool.shape[1])]

examples = {}
for kind in ("prior", "gnn", "structured", "unet"):
    seeds = [0] if kind == "prior" else args.seeds
    one, multi, by_tool, node, top, speed = [], [], {t: [] for t in TOOLS}, [], [], []
    for seed in seeds:
        model = T.load(kind, seed, geom)
        m1 = T.one_step_metrics(model, te)
        one.append(1e3 * m1["map_rmse"].mean())
        for i, t in enumerate(TOOLS):
            by_tool[t].append(1e3 * m1["map_rmse"][m1["tool"] == i].mean())
        if "node_rmse" in m1:
            node.append(1e3 * m1["node_rmse"].mean())
            top.append(1e3 * m1["top_rmse"].mean())
        multi.append(1e3 * T.multi_step_metrics(model, te).mean(axis=0))
        # speed: one batch of 32 candidates, as the planner uses it
        state = (hb if model.state_kind == "map" else te.before)[:32]
        with torch.no_grad():
            model(state, te.tool[:32], te.action[:32])
            t0 = time.time()
            for _ in range(5):
                model.heightmap(model(state, te.tool[:32], te.action[:32]))
            speed.append((time.time() - t0) / 5 / 32)
        if seed == seeds[0]:
            idx = torch.arange(0, len(te), max(1, len(te) // 6))[:6]
            st = (hb if model.state_kind == "map" else te.before)[idx]
            with torch.no_grad():
                examples[kind] = model.heightmap(model(st, te.tool[idx], te.action[idx])).numpy()
            examples["before"], examples["after"] = hb[idx].numpy(), ha[idx].numpy()
            examples["tool"] = te.tool[idx].numpy()
        if kind != "prior":
            info = json.loads((T.MODEL_DIR / f"{kind}_seed{seed}.json").read_text())
            out["training"].setdefault(kind, []).append(dict(seed=seed, wall_s=info["wall_s"], n_params=info["n_params"],
                                                             epochs=info["epochs"], n_train=info["n_train"],
                                                             history=info["history"]))
    multi = np.array(multi)
    out["one_step"][kind] = dict(map_rmse_mm=ms(one), by_tool={t: ms(v) for t, v in by_tool.items()},
                                 node_rmse_mm=ms(node) if node else None, top_vertex_rmse_mm=ms(top) if top else None)
    out["multi_step"][kind] = [ms(multi[:, j]) for j in range(multi.shape[1])]
    out["speed"][kind] = dict(ms_per_prediction=1e3 * float(np.mean(speed)), n_params=int(model.n_params()))
    print(f"{kind:11s} one-step {ms(one)[0]:.3f} +- {ms(one)[1]:.3f} mm | 5-step {ms(multi[:, 4])[0]:.3f} +- {ms(multi[:, 4])[1]:.3f} mm | "
          f"{out['speed'][kind]['ms_per_prediction']:.2f} ms per prediction", flush=True)
print(f"no change   one-step {out['one_step']['no_change']['map_rmse_mm'][0]:.3f} mm | 5-step {out['multi_step']['no_change'][4][0]:.3f} mm", flush=True)

# the two ways of carrying the prior from vertices to nodes (learn/prior.py)
from clay_sculptor.learn.prior import SweptVolumePrior  # noqa: E402

out["prior_projection"] = {}
for proj in ("lumped", "lsq"):
    pr = SweptVolumePrior(geom, proj)
    with torch.no_grad():
        hp = torch.cat([geom.heightmap(pr(te.before[i:i + 256], te.tool[i:i + 256], te.action[i:i + 256])) for i in range(0, len(te), 256)])
    out["prior_projection"][proj] = 1e3 * float(T.map_rmse(hp, ha).mean())
print("prior projection:", out["prior_projection"], flush=True)
# what the prior misses: lateral against vertical motion of the free nodes
free = torch.from_numpy(geom.free)
dn = (te.after - te.before)[:, free]
out["true_motion_mm"] = dict(lateral_rms=1e3 * float(dn[..., :2].pow(2).sum(-1).mean().sqrt()), vertical_rms=1e3 * float(dn[..., 2].pow(2).mean().sqrt()))

# oracle speed (measured by m0_bench.py)
m0 = RESULTS / "m0_bench.json"
if m0.exists():
    sp = json.loads(m0.read_text())["speed"]
    out["speed"]["oracle"] = dict(ms_per_prediction_1thread=1e3 * sp["press_s_per_rollout_1thread"],
                                  ms_per_prediction_16thread=1e3 * sp["press_s_per_rollout_16thread"],
                                  roller_ms_16thread=1e3 * sp["roller_s_per_rollout_16thread"])

# domain shift: the same models on clay with a different Young's modulus or yield strain
for name in args.shift:
    path = DATA / f"transitions_{name}.npz"
    if not path.exists():
        continue
    ds = D.load(path)
    sh = T.Transitions(ds, slice(0, len(ds["nodes"])), geom)
    sb, sa = sh.maps
    res = dict(overrides=json.loads(str(ds["meta_overrides"])), n=len(sh), no_change=[1e3 * float(T.map_rmse(sb, sa).mean()), 0.0])
    for kind in ("prior", "gnn", "structured", "unet"):
        seeds = [0] if kind == "prior" else args.seeds
        res[kind] = ms([1e3 * T.one_step_metrics(T.load(kind, s, geom), sh)["map_rmse"].mean() for s in seeds])
    out["shift"][name] = res
    print(f"shift {name} {res['overrides']}: " + ", ".join(f"{k} {res[k][0]:.3f}" for k in ("no_change", "prior", "gnn", "structured", "unet")), flush=True)

(RESULTS / "m2_eval.json").write_text(json.dumps(out, indent=1))
np.savez_compressed(RESULTS / "m2_examples.npz", **examples)
plots.m2_errors(out, RESULTS / "m2_errors.png")
plots.m2_examples(examples, float(geom.clay.top), RESULTS / "m2_examples.png")
if out["shift"]:
    plots.m2_shift(out, RESULTS / "m2_shift.png")
print("wrote results/m2_eval.json", flush=True)
