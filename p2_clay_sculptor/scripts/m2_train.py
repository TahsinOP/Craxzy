"""M2: train the dynamics models on CPU.

    env -u PYTHONPATH .venv/bin/python scripts/m2_train.py --kinds gnn structured unet --seeds 0 1 2

One checkpoint per (kind, seed) in ``data/models``.  The structured model is
trained with five seeds by default: they are the deep ensemble of M4.
"""
import argparse
import json
import time

import torch

from clay_sculptor import DATA, RESULTS, config
from clay_sculptor.envs.clay_env import quiet_build
from clay_sculptor.envs.shape import Geometry
from clay_sculptor.learn import data as D
from clay_sculptor.learn import train as T

ap = argparse.ArgumentParser()
ap.add_argument("--kinds", nargs="*", default=["structured", "gnn", "unet"])
ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
ap.add_argument("--ensemble-seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4], help="seeds of the structured model")
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--unet-epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--data", default="main")
ap.add_argument("--max-train", type=int, default=0, help="use only the first N training episodes (0: all)")
ap.add_argument("--tag", default="", help="suffix for the checkpoint names (data-size ablation)")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--skip-existing", action="store_true")
args = ap.parse_args()
torch.set_num_threads(args.threads)

cfg = config.load()
# only the geometry is needed here; a full ClayEnv (three models, their data, the batches) costs 400 MB
geom = Geometry(quiet_build("press", cfg.clay, cfg.sim), cfg.clay, cfg.map)
data = D.load(DATA / f"transitions_{args.data}.npz")
tr_s, va_s, te_s = T.split(data)
if args.max_train:
    tr_s = slice(0, args.max_train)
tr, va = T.Transitions(data, tr_s, geom), T.Transitions(data, va_s, geom)
print(f"train {len(tr)} transitions, val {len(va)}", flush=True)
for kind in args.kinds:
    seeds = args.ensemble_seeds if kind == "structured" else args.seeds
    for seed in seeds:
        name = kind + args.tag
        if args.skip_existing and (T.MODEL_DIR / f"{name}_seed{seed}.pt").exists():
            continue
        t0 = time.time()
        epochs = args.unet_epochs if kind == "unet" else args.epochs
        model, info = T.train(kind, geom, tr, va, seed, epochs=epochs, batch=args.batch, lr=args.lr,
                              log=lambda s: print(s, flush=True))
        T.save(model, info, name)
        print(f"saved {name}_seed{seed}: {info['n_params']} params, {time.time() - t0:.0f} s, "
              f"val map RMSE {info['history'][-1]['val_map_rmse_mm']:.3f} mm", flush=True)
