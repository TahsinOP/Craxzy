"""M2 data: random-primitive transitions on CPU with ``mujoco.batch``.

    env -u PYTHONPATH .venv/bin/python scripts/m2_dataset.py --episodes 1100 --steps 6

Writes ``data/transitions_<name>.npz``.  ``--set clay.young=...`` generates a
shifted-material set (used for the domain-shift evaluation).
"""
import argparse
import json
import time

from clay_sculptor import DATA, RESULTS, config
from clay_sculptor.learn import data as D

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=1100)
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--nthread", type=int, default=0)
ap.add_argument("--name", default="main")
ap.add_argument("--set", nargs="*", default=[])
args = ap.parse_args()

cfg = config.load(overrides=args.set)
env = cfg.env(nthread=args.nthread, nsim=args.batch)
print(f"dataset '{args.name}': {args.episodes} episodes x {args.steps} steps, seed {args.seed}, overrides {args.set}", flush=True)
t0 = time.time()
data = D.generate(env, args.episodes, args.steps, args.seed, log=lambda s: print(s, flush=True))
out = DATA / f"transitions_{args.name}.npz"
D.save(out, data, seed=args.seed, overrides=json.dumps(args.set))
summary = dict(name=args.name, episodes=args.episodes, steps=args.steps, seed=args.seed, overrides=args.set,
               transitions=int(data["ok"].sum()), failed=int((~data["ok"]).sum()), wall_s=time.time() - t0,
               s_per_transition=(time.time() - t0) / (args.episodes * args.steps))
RESULTS.mkdir(exist_ok=True)
(RESULTS / f"m2_dataset_{args.name}.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary), flush=True)
