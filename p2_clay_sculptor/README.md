# P2: Clay Sculptor

A Franka Panda presses, rolls and stamps a block of plastic clay into a target shape in MuJoCo 4.0. The stack is hybrid at three levels: a geometric prior (the volume the tool sweeps) with a learned residual for what the material does, a sampling-based planner whose proposals use that same geometry, and an ensemble meant to ask the simulator only when its members disagree. The brief is in [`TASK.md`](TASK.md).

Everything runs on CPU. The plastic flex is not available on the GPU backends in 4.0, so all multi-world work uses `mujoco.batch`, and the learned models are sized to train without a GPU.

## Results at a glance

Three targets (slab, dent, X), ten planner seeds each; "error left" is the final height-map RMSE as a share of the untouched block's, averaged over the targets (mean ± std over seeds). Full tables in `results/M0.md` to `M4.md`; the explanations are in the docs site.

| planner (rollout model) | error left | planning time per decision | simulated primitives per episode |
|---|---|---|---|
| oracle: MuJoCo for every candidate (M1) | 13.9 ± 1.9 % | 24.4 s | 371 |
| U-Net on the height map (pure learning) | 14.6 ± 2.0 % | 0.06 s | 0 |
| prior only (no learning) | 15.0 ± 1.8 % | 0.10 s | 0 |
| structured: prior + GNN residual (hybrid) | 15.8 ± 1.9 % | 0.76 s | 0 |
| GNN on the cage nodes (pure learning) | 17.3 ± 2.3 % | 0.79 s | 0 |
| ensemble of 5 structured, never asks (M4) | 15.8 ± 1.3 % | 4.6 s | 0 |
| ensemble, re-rank best 8 with MuJoCo at every decision | 14.8 ± 2.0 % | 6.6 s | 30 |
| ensemble, re-rank when the members' maps disagree (median threshold) | 16.0 ± 1.7 % | 5.6 s | 14 |
| ensemble, re-rank when the members doubt the ranking (median threshold) | 15.1 ± 1.3 % | 5.3 s | 15 |
| ensemble, re-rank at random (50 %) | 14.9 ± 1.6 % | 5.4 s | 16 |

One-step prediction error on 600 held-out transitions (M2): no change 1.74 mm, prior 0.61, GNN 0.32, structured 0.27, U-Net 0.15 mm.

**Demo on an unseen target** (`scripts/demo_unseen.py` → `results/demo_unseen.mp4`): a staircase of three pockets, 6, 9 and 12 mm deep, added after all training and tuning. Every planner finds the three pockets in order and ends at 0.40 to 0.52 mm of the initial 3.78 mm (U-Net 0.40, hybrid 0.44, structured and GNN 0.47, prior 0.49, oracle 0.52 ± 0.17; three seeds, the spreads overlap).

**Smiley on a 1.75x bigger block** (`scripts/demo_unseen.py --target smiley` → `results/demo_smiley.mp4`): the trained models reused without retraining. Every planner draws the face; after ten presses GNN 0.88, oracle 0.89, hybrid 0.93, prior 1.04, structured 1.10, U-Net 1.26 mm of the initial 2.48 (one seed). The U-Net, which takes absolute pixel coordinates as input, transfers worst.

**Tool changes from a rack** (`scripts/demo_toolchange.py` → `results/demo_toolchange.mp4`): the planner picks the tool at every step; this replays its three-tool episode with the arm returning and fetching tools from a rack beside the clay (three changes, same 1.34 mm result as planned).

**A house from three tools** (`scripts/demo_unseen.py --target house` → `results/demo_house.mp4`; `scripts/demo_toolchange.py --source house --planner hybrid` → `results/demo_toolchange_house.mp4`). On the 245 mm block, the target is a roof of two rolls, a stamped window and a pressed door, and the planner picks the tool at each of eight steps. Every planner uses all three tools in the right places. After eight actions, of the initial 3.17 mm: hybrid 0.84, oracle 0.86, prior 0.97, structured 0.99, GNN 1.07, U-Net 1.35 mm (one seed). The hybrid matches the oracle with a twelfth of its simulations.

What held up (paired comparisons by target and seed):

- **The prior helps the graph network**: 17 % lower one-step error, and 1.5 ± 1.3 points less error left in the planner (p = 0.03).
- **Learned planners come close to the oracle** at 30 to 400 times less planning time; the U-Net planner is not separated from it (−0.8 ± 0.9 points, p = 0.10).
- **Asking MuJoCo to re-rank the model's best eight candidates pays**: −1.0 ± 0.9 points against the ensemble alone (p = 0.04), half of the oracle's advantage at 1/12 of its simulations.

What did not work:

- **The ensemble's disagreement is a bad trigger.** It predicts its own error well (Spearman 0.79), but re-ranking when it is high is *worse* than re-ranking at random decisions at the same rate (+1.1 ± 0.6 points, p = 0.0004). It fires by target (every slab decision, no dent decision), not by decision. A trigger on the members' doubt about the ranking ties with random (p ≥ 0.3).
- **A better model is not a much better planner**: a factor of two in prediction error became one or two points of shape error. On the slab, the prior alone, the worst predictor, planned better than MuJoCo itself, because its error made it press shallower.
- **No target was reached**, though all are reachable exactly: the one-step greedy search digs too deep early and clay cannot be raised again. The guided proposal also ignores spring-back, so first strokes start too shallow (left in place so every planner shares one search).
- **Not attempted**: folding the paper sheet, and a second arm. The graph networks are undertrained next to the U-Net (6 epochs on a CPU).

## Run it

The shell on this machine exports ROS Humble's Python path, so every command needs `env -u PYTHONPATH`.

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"     # once; versions are pinned in pyproject.toml

# watch the arm sculpt, live in the MuJoCo viewer
env -u PYTHONPATH .venv/bin/python scripts/view.py --demo press
env -u PYTHONPATH .venv/bin/python scripts/view.py --target x --plan oracle
env -u PYTHONPATH .venv/bin/python scripts/view.py --target slab --plan structured --slow 2

# reproduce every milestone, or start from milestone N
scripts/run_all.sh
scripts/run_all.sh 3

# regenerate results/M0.md ... M4.md from the JSON results
env -u PYTHONPATH .venv/bin/python scripts/make_reports.py

# tests (33)
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -q

# the scene in MuJoCo Studio (assets/clay_{press,roller,stamp}.xml hold absolute mesh paths for this machine;
# clay_sculptor.models.scene.export_xml regenerates them elsewhere)
env -u PYTHONPATH .venv/bin/python -m mujoco.studio.viewer --model=assets/clay_press.xml --hot_reload=all
```

| script | what it does | main arguments |
|---|---|---|
| `scripts/m0_bench.py` | scripted press, roll and stamp; permanence; save/restore; tracking; speed; video | `--no-video` |
| `scripts/m0_discretisation.py` | trilinear cage against the full-vertex flex | |
| `scripts/m1_oracle.py` | CEM with MuJoCo as the rollout model on three targets | `--targets --seeds --pop --iters` |
| `scripts/m2_dataset.py` | random-primitive transitions with `mujoco.batch` | `--episodes --steps --seed --name --set` |
| `scripts/train_all.sh` | trains all eleven models, two capped processes | epochs |
| `scripts/m2_train.py` | trains chosen models | `--kinds --seeds --ensemble-seeds --epochs --max-train` |
| `scripts/m2_eval.py` | one-step and multi-step error, speed, domain shift | `--seeds --shift` |
| `scripts/m3_mpc.py` | the same planner with each learned model | `--kinds --targets --seeds` |
| `scripts/m4_fallback.py` | ensemble signals, oracle re-ranking under three triggers, paired gains plot | `--triggers --quantiles --top-m --members --resume` |
| `scripts/stretch.py` | the letter T, and a target where the planner picks the tool | `--targets --planners` |
| `scripts/demo_unseen.py` | every planner on an unseen target (a staircase, a smiley or a three-tool house on a 245 mm block), one side-by-side video | `--target --seeds --planners --video-only` |
| `scripts/demo_toolchange.py` | a three-tool episode (stretch combo, or the house) replayed with the tool changes done by the arm from a rack | `--source --planner --seed` |
| `scripts/view.py` | live viewer | `--demo --target --plan --slow` |

Outputs go to `results/` (JSON, PNG, MP4 and the `Mx.md` reports) and `data/` (datasets, trained models). Every script reads `configs/default.yaml`; the dataset and benchmark scripts take `--set key.sub=value`.

Two habits for long runs on a laptop: `scripts/run_all.sh` wraps Python in `systemd-inhibit` so the machine does not suspend, and training runs under a memory cap (`scripts/train_all.sh`), so a mistake ends the job and not the desktop session.

## How it works

```
target height map, current shape (cage nodes -> vertices -> 32 x 32 height map)        envs/shape.py
   └─ CEM over one primitive a = (x, y, yaw, depth, travel), 96 candidates             control/mpc.py
        └─ rollout model: what does this action do to this shape?
             ├─ oracle       MuJoCo itself, mujoco.batch, exact state restore           envs/clay_env.py
             ├─ prior        the volume the tool sweeps, one formula for three tools    learn/prior.py
             ├─ gnn          graph network on the cage nodes (pure learning)            learn/models.py
             ├─ structured   prior + graph-network residual (hybrid)
             └─ unet         U-Net on the height map (pure learning)
        └─ fallback: five structured models disagree about the chosen action?
             └─ yes: simulate the best 8 candidates and execute the truly best          control/mpc.py (HybridEval)
   └─ execute: IK -> minimum-jerk joint references -> pid servos -> plastic flex        control/arm.py, primitives.py
```

Design decisions that the rest depends on:

- **The clay is a trilinear cage** (`flexcomp dof="trilinear"`, 12 x 12 x 3 cells, `<elasticity yield="0.02">`). The full-vertex flex keeps more of a press but creeps for seconds after unloading and inverts elements at 14 mm; the cage does neither (`results/M0.md`).
- **A state is the whole `mjSTATE_INTEGRATION` vector**, plastic strain included. Restoring it replays a rollout bit for bit; leaving the plastic strain out costs 3.4 mm.
- **Three tools, one state layout.** Each tool is its own compiled model and differs only in geoms welded to the flange, so a tool change is a change of model with the state carried over.
- **`pid` servos instead of Menagerie's position servos**, with gravity compensation: the press reaches 10.0 of a commanded 10.0 mm, against 8.4 mm with plain PD.
- **One cost for every rollout model**: height-map RMSE to the target. Chamfer and EMD are computed on true simulator states only.
- **Targets come from hidden scripted demonstrations**, so zero error is reachable and errors are on an absolute scale.

## Layout

```
src/clay_sculptor/
  models/scene.py        MjSpec scene: clay flexcomp, Menagerie Panda, three tools, pid servos
  envs/clay_env.py       state layout, batched primitive rollouts, continuous execution
  envs/shape.py          nodes <-> vertices <-> height map, Chamfer, EMD
  envs/targets.py        targets from hidden demonstrations
  control/arm.py         tool-down inverse kinematics
  control/primitives.py  the five-parameter primitive and its joint references
  control/mpc.py         CEM, guided proposal, evaluators (oracle, model, hybrid), episode loop
  learn/prior.py         swept-volume envelope and prior
  learn/models.py        graph network, U-Net, common interface, ensemble
  learn/train.py         datasets, training, one-step and multi-step metrics
  learn/data.py          random-primitive data generation
  eval/                  plots (light and dark), rendering, episode bookkeeping
scripts/                 one or two scripts per milestone, view.py, run_all.sh
configs/default.yaml     every physical and planning parameter
tests/                   33 tests: kinematics, state, shape maps, prior geometry, planner pieces, models
docs/                    Astro/Starlight site (pnpm install --frozen-lockfile; pnpm dev)
```

## Documentation site

`docs/` holds a site with the derivations, interactive figures, plots and clips: the clay model and why the cage was chosen, the action primitive and its kinematics, the state, the shape maps, the planner, the prior, the three models, and every result with its limits.

```bash
env -u PYTHONPATH .venv/bin/python docs/figures/make_figures.py   # figures from results/, light and dark
env -u PYTHONPATH .venv/bin/python scripts/docs_videos.py         # clips
cd docs && pnpm install --frozen-lockfile && pnpm dev             # localhost:4321
pnpm build && pnpm check:widgets                                   # static build, slider sweep of the widgets
pnpm build:offline                                                 # docs/dist-offline, opens from the file system
```
