# P2 clay sculptor: working notes for Claude Code

Read `TASK.md` for the brief and `README.md` for what exists and how to reproduce it.

## Environment
- Python 3.12 venv in `.venv` (uv). `mujoco==4.0.1`, `mujoco-menagerie`, CPU `torch`. Versions are pinned in `pyproject.toml`.
- **Always run with `env -u PYTHONPATH`**: the shell exports ROS Humble's Python 3.10 site-packages, which breaks imports and pytest plugin loading.
  - scripts: `env -u PYTHONPATH .venv/bin/python scripts/<name>.py`
  - tests: `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -q`
- No CUDA build of torch on this machine and about 5 GB of free RAM: everything is sized for CPU. Run anything that trains under a memory cap so a mistake kills the job and not the desktop: `systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env -u PYTHONPATH .venv/bin/python ...`.
- Long runs: prefix with `systemd-inhibit --what=idle:sleep` (the laptop suspends when idle), use `python -u`, and `grep --line-buffered` when filtering into a log.
- Offscreen rendering uses EGL (`MUJOCO_GL=egl` is set by `clay_sculptor.eval.render`).

## Layout
- `src/clay_sculptor/{models,envs,control,learn,eval}`, `scripts/` (one or two scripts per milestone), `configs/default.yaml`, `tests/`.
- `data/` (datasets, trained models) and `results/` (json, png, mp4) are git-ignored, except `results/*.md`.
- `docs/` is the Astro/Starlight site; `docs/figures/make_figures.py` regenerates its figures from `results/`.

## Conventions that matter here
- **State = MuJoCo's `mjSTATE_INTEGRATION` vector** (flat float64, 11,834 numbers), plastic strain included. Save = copy, restore = write back. Never restore from `qpos`/`qvel` alone: the replay drifts by millimetres (`tests/test_bench.py`).
- **The three tool models share one state layout**, so a state moves between tools unchanged. Keep it that way: tools may differ only in geoms welded to the flange.
- **Shape = cage nodes** `(13, 13, 4, 3)` in clay coordinates (origin at the block centre on the board). Vertices are `W @ nodes` (trilinear, exact), the height map is derived from the top vertex layer (`envs/shape.py`). Models, metrics and plots all go through `Geometry`.
- **One planning cost for every rollout model**: height-map RMSE to the target. Chamfer and EMD are evaluation metrics, computed on true simulator states only.
- **Depth is absolute**: the tool bottom goes to `clay.top - depth`, whatever the current shape.
- Rollouts set the arm directly to the hover pose of the next primitive (the free-space move is not simulated). `ClayEnv.run_continuous` simulates it; `scripts/m0_bench.py` reports the difference.
- Every milestone: a script that reproduces it, `results/Mx.md` with plots, the classical baseline (prior only) and the pure-learning baselines (GNN, U-Net) next to the hybrid (structured). Closed-loop runs use the ten planner seeds of `configs/default.yaml` (planner seed i uses model seed i mod 3) and are compared *paired* by target and seed (`make_reports.paired`); prediction errors use three training seeds. Three seeds were not enough to separate any M4 configuration; the three-seed first pass is kept in `results/three_seeds/`.
- Plots use `eval/plotstyle.py`: fixed colour, marker and dash per method, light and dark themes.

## MuJoCo 4.0 facts already checked against 4.0.1
- `flexcomp dof="trilinear"` with `<elasticity yield=...>` stores 6 numbers per cage cell in `mjData.plastic`. The full-vertex flex (`dof` omitted) creeps for seconds after unloading and inverts elements under a sharp tool; the trilinear cage does neither. Do not switch back without redoing the checks in `results/M0.md`.
- `mujoco.batch.Batch.rollout(nstep, control, record=...)`: `record` takes MjData array names or `mjtState` items, not `"time"`. Results are bit-identical to `mj_step` on one `MjData` at any thread count.
- `pid` actuators take `[q_ref, qd_ref]` per joint, so `nu = 14` for the arm and `na = 7` (the integrators).
- `MjSpec.attach(child, frame=..., prefix=...)`; set the child's `option.integrator` and `impratio` to the parent's first, or MuJoCo prints a conflict warning on every build.
- PyYAML reads `1.5e5` as a string; write `1.5e+5` (the config loader also casts).

## Gotchas found the hard way
- Thread scaling of the clay rollouts stops at about 3.5x on 8 cores (memory bound). Budget 0.2 s per press or stamp rollout and 0.43 s per roll with all cores.
- A graph network with diagonal edges or 96 hidden units at batch 32 needs more than 5 GB to train: keep `reach=1`, 48 hidden, 4 steps unless there is a GPU.
- Training memory: one process peaks at 2.5 GB at batch 32 and 1.4 GB at batch 16 (measure with `/usr/bin/time -v`). Three processes at batch 32 got two of themselves killed by their caps and, before that, made the kernel kill two Chrome renderers on the desktop. `scripts/train_all.sh` therefore runs two processes at batch 16 under 2 GB caps. Do not raise the process count or the batch size without measuring first.
- `scripts/view.py` uses the passive viewer; a tool change closes and reopens the window because each tool is its own compiled model. Keep the short sleep after `close()`.
- Docs: this machine has Node 22.23.3 and newer Astro releases refuse it. `docs/pnpm-lock.yaml` is P1's lockfile (Astro 7.3.1); install with `pnpm install --frozen-lockfile`, never a bare `pnpm install`.
- Docs widgets: no blank line inside a `<figure>` or `<script>` block (Markdown takes over after one), one widget per page, static `viewBox`, ids prefixed `w-`. `pnpm check:widgets` sweeps every slider.
- `scripts/m4_fallback.py --resume` (or `scripts/ten_seeds.sh resume`) keeps the configurations already in `results/m4_fallback.json` when seeds, members and top-m match; use it after an interrupted run instead of starting over (11 configurations take about 2.5 h).
- The guided proposal (`control/mpc.py`) sets the depth as if the clay kept the whole stroke; this clay gives back a third, so first strokes come out too shallow. Known and deliberately left in place so all planners share one search; correct it (depth x ~1.5) only together with a full rerun of M1, M3 and M4.
- M4 outcome to keep in mind before touching the fallback: the disagreement trigger fires by target (it scales with the size of the predicted change) and is worse than random re-ranking; the misranking trigger ties random. `results/M4.md` has the head-to-head table and per-target firing rates.
