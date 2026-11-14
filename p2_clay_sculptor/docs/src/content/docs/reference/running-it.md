---
title: "Running It"
description: "Install, watch the arm sculpt in the live viewer, reproduce every milestone, run the tests, and rebuild this site."
sidebar:
  order: 1
---

Everything runs on the CPU. The clay needs MuJoCo's native flex and plasticity, which are not available on the GPU backends in 4.0, and the learned models are small enough to train without one.

## Install

```bash
cd p2_clay_sculptor
uv venv --python 3.12
uv pip install -e ".[dev]"        # versions are pinned in pyproject.toml
```

:::caution[On this machine, unset PYTHONPATH]
The shell exports ROS Humble's Python 3.10 site-packages. Every command below is therefore written with `env -u PYTHONPATH`. On a clean machine the prefix does nothing and can be dropped.
:::

## Watch

The live viewer replays scripted sequences, or plans towards a target and replays what it executed.

```bash
# scripted: three presses, three roller passes, one stamp
env -u PYTHONPATH .venv/bin/python scripts/view.py --demo press
env -u PYTHONPATH .venv/bin/python scripts/view.py --demo roll
env -u PYTHONPATH .venv/bin/python scripts/view.py --demo stamp

# plan towards a target with a rollout model, then replay
env -u PYTHONPATH .venv/bin/python scripts/view.py --target x --plan oracle
env -u PYTHONPATH .venv/bin/python scripts/view.py --target slab --plan structured --slow 2
env -u PYTHONPATH .venv/bin/python scripts/view.py --target dent --plan demo      # the hidden demonstration itself
```

`--plan` is one of `oracle`, `prior`, `gnn`, `structured`, `unet` or `demo`; the learned ones need the trained models of milestone 2. `--target` is one of `slab`, `dent`, `x`, `t`, `combo`. Planning with the oracle takes from one to four minutes before anything moves, because it simulates about a hundred primitives per decision.

The scene also opens in MuJoCo Studio:

```bash
env -u PYTHONPATH .venv/bin/python -c "from clay_sculptor.models.scene import export_xml; export_xml('assets/clay_press.xml', 'press')"
env -u PYTHONPATH .venv/bin/python -m mujoco.studio.viewer --model=assets/clay_press.xml --hot_reload=all
```

The exported file holds absolute mesh paths for the machine it was written on.

## Reproduce

```bash
scripts/run_all.sh          # every milestone in order
scripts/run_all.sh 3        # from milestone 3 on
```

or one script at a time:

| script | milestone | what it does | main arguments |
|---|---|---|---|
| `scripts/m0_bench.py` | M0 | scripted sequences, permanence, save/restore, tracking, speed, video | `--no-video` |
| `scripts/m0_discretisation.py` | M0 | trilinear cage against full-vertex flex | |
| `scripts/m1_oracle.py` | M1 | CEM with MuJoCo as the rollout model, three targets | `--targets --seeds --pop --iters` |
| `scripts/m2_dataset.py` | M2 | random-primitive transitions with `mujoco.batch` | `--episodes --steps --seed --name --set` |
| `scripts/m2_train.py` | M2 | trains the GNN, the structured model (five seeds) and the U-Net | `--kinds --seeds --epochs --max-train` |
| `scripts/m2_eval.py` | M2 | one-step and multi-step error, speed, domain shift | `--seeds --shift` |
| `scripts/m3_mpc.py` | M3 | the same planner with each learned model | `--kinds --targets --seeds` |
| `scripts/m4_fallback.py` | M4 | ensemble signals, oracle re-ranking under three triggers, paired gains plot | `--triggers --quantiles --top-m --members --resume` |
| `scripts/stretch.py` | stretch | the letter T, and a target where the planner picks the tool | `--targets --planners --seeds` |
| `scripts/demo_unseen.py` | demo | every planner on an unseen target (a staircase, or a smiley on a 245 mm block), one side-by-side video | `--target --seeds --planners --video-only` |
| `scripts/demo_toolchange.py` | demo | the three-tool episode replayed with the tool changes done by the arm from a rack | `--planner --seed` |
| `scripts/make_reports.py` | all | writes `results/M0.md` to `M4.md` from the JSON results | |

Every script reads `configs/default.yaml` and accepts `--set key.sub=value` where it makes sense, for example a softer clay:

```bash
env -u PYTHONPATH .venv/bin/python scripts/m2_dataset.py --episodes 48 --name young_lo --set clay.young=0.75e+5
```

Outputs go to `results/` (JSON, PNG, MP4 and the `Mx.md` reports) and `data/` (datasets and trained models).

### How long it takes

On the 8-core laptop with no GPU that produced the numbers in this site:

| step | time |
|---|---|
| M0 benchmark | 5 min |
| M1 oracle MPC, 30 episodes (3 targets × 10 seeds) | 55 min |
| M2 dataset, 6,600 transitions | 32 min |
| M2 training, 11 models, two at a time | about 1.5 h (17 min per graph network, 12 min per U-Net) |
| M3, 120 episodes (4 learned models) | 12 min |
| M4, 330 episodes (11 planners) | 2.4 h |
| stretch, learned planners (3 seeds) and the oracle (1 seed) | 25 min, about 16 of them for the oracle |

Two habits keep long runs safe on a laptop. `scripts/run_all.sh` wraps Python in `systemd-inhibit` so the machine does not suspend mid-run. And anything that trains should run under a memory cap, so that a mistake ends the job and not the desktop session:

```bash
systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 \
  env -u PYTHONPATH .venv/bin/python scripts/m2_train.py
```

On a server, raise `--episodes`, `--epochs` and the CEM population (`--pop`); nothing else needs to change.

## Test

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -q
```

[The Tests](/reference/tests/) lists what each one establishes.

## Rebuild this site

```bash
env -u PYTHONPATH .venv/bin/python docs/figures/make_figures.py     # figures from results/, light and dark
env -u PYTHONPATH .venv/bin/python scripts/docs_videos.py           # clips for the site
cd docs
pnpm install
pnpm dev                # live preview on localhost:4321
pnpm build              # static site in docs/dist
pnpm build:offline      # docs/dist-offline, opens straight from the file system
pnpm check:widgets      # runs every interactive figure through its slider ranges
```
