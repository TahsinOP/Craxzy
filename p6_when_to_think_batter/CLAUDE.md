# P6 when-to-think batter: working notes for Claude Code

Read `TASK.md` for the brief and `README.md` for what exists and how to reproduce it.

## Environment
- Python 3.12 venv in `.venv` (uv). `mujoco==4.0.1`, `mujoco-menagerie`, CPU `torch`, `osqp`. Versions are pinned in `pyproject.toml`.
- **Always run with `env -u PYTHONPATH`**: the shell exports ROS Humble's Python 3.10 site-packages, which breaks imports and pytest plugin loading.
  - scripts: `env -u PYTHONPATH .venv/bin/python scripts/<name>.py`
  - tests: `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -q`
- Offscreen rendering uses EGL (`MUJOCO_GL=egl` is set in `batter/__init__.py`). Videos default to Filament (Vulkan, `eval/filament.py`); `--renderer classic` uses the classic renderer.
- Speculative contacts are CPU only in MuJoCo 4.0: every multi-world run goes through `mujoco.batch`, never MJX/Warp.

## Layout
- `src/batter/{models,envs,control,learn,estim,eval}`, `scripts/` (one script per milestone), `configs/default.yaml`, `tests/`.
- `data/` (batting data, trained models) and `results/` (json, png, mp4, logs) are git-ignored, except `results/*.md`.
- `results/Mx.md` are generated from the JSON by `scripts/make_reports.py`. Do not edit them by hand.

## Conventions that matter here
- World frame: robot base at the origin on a pedestal, pitches come down the +x axis towards -x, the field and the target are at +x.
- One discrete flight model for everything (`estim/flight.py`): MuJoCo's `Euler` step with explicit fluid forces, coefficients read off the engine by `FlightModel.identify`. The pitching machine, the EKF, the planner's forecast and the landing extrapolation all use it. Do not switch the integrator to `implicitfast` without redoing that model (its fluid derivatives couple linear and angular velocity).
- The planner interface is `plan_ctx(ctx) -> Plan | None` (analytic: `control/planner.py`; learned: `learn/wm_planner.py`). Planners time themselves (`wall`, `times`).
- A trigger is a callable on the environment's observation dict (`control/triggers.py`). Across processes it is a picklable spec `(kind, value)`, see `eval/runner.py`.
- Every milestone: a script that reproduces it, `results/Mx.md` with plots, 3 seeds (mean ± std), classical and pure-learning baselines next to the hybrid.
- Plots use `eval/plotstyle.py` (fixed colour per method: classical blue, pure learning orange, hybrid aqua).

## The rally (stretch, `src/batter/rally/`, `scripts/rally.py`, `configs/rally.yaml`)
- Two Pandas; arm B is the second copy attached with prefix `b_` at x = L, rotated by pi. Every per-arm computation happens in that arm's own frame (`rally.scene.to_arm_frame`), where it is arm A, so the batter's planner, IK and QP are reused unchanged. One single-arm model (`build_arm_model`) serves both arms' kinematics.
- `configs/rally.yaml` is merged over `default.yaml` by `batter.rally.load`; `table.L` is derived there.
- Table bounce: frictionless pair, exact law v_z' = -e v_z (start-of-step v_z) inside the step whose ballistic prediction crosses the surface (`rally/flight.py`). With friction, MuJoCo's one-step restitution contact can return the ball faster than it came: do not turn friction back on without redoing that study (`scripts/rally.py`, bounce study).
- The filter must be told about bounce timing (`TableFlight.extra_process_noise`), and the arm must not replan while the filter's v_z sd exceeds `rally.replan_vz_sd`: without that gate rallies are half as long.
- Within `rally.commit_time` of an impact only the same strike candidate may be re-planned: switching strike planes that late (they are 60 ms apart at ping-pong speeds) ruins the stroke.
- A midi table and a 0.30 m paddle handle: on an ITTF table a return needs about 3 m/s of paddle speed, the Panda gives 1.6.
- The paddle pose cache (`data/rally_poses.json`) is keyed on the paddle geometry and strike zone only; design it in the main process before starting a pool.

## Gotchas found the hard way
- The planner is single-threaded Python, so `mujoco.batch` alone does not use the machine. `eval/runner.py` splits pitches over a spawn process pool, one small batch per process, `nthread=1`. Functions sent to the pool must live in the package, not in a script.
- Wall-clock per planner call measured inside a loaded pool is inflated (about 2x). `time_planner` measures it alone on one core; that is the number to quote.
- The bat-ball pair is frictionless with a direct `solref = (0, -(1+e)/h)`: zero stiffness, as the 4.0 notes on speculative contacts and restitution ask. The effective restitution the planner needs (0.585) is lower than the pair's (0.8) because the wrist gives; `scripts/m1_hitter.py` fits it.
- `Batch.reset` zeroes the sensor history: the delayed camera sensor reads zeros for the first `delay/h` steps, so fixes are only used from step `nd` on.
- Do not render on the GPU from the process pool. 24 concurrent EGL contexts made the nouveau driver reject commands (`kernel rejected pushbuf`). `scripts/m5_events.py` calls `eventcam.use_software_rendering()` before the pool starts: it probes the EGL devices and selects Mesa llvmpipe through `MUJOCO_EGL_DEVICE_ID` (on this machine: 0 nouveau, 1 Intel, 2 llvmpipe). `LIBGL_ALWAYS_SOFTWARE=1` segfaults in `eglInitialize` with MuJoCo's device-based EGL. Videos render in the main process only.
- `MUJOCO_GL=egl` is set in `batter/__init__.py` because MuJoCo reads it at import; setting it later silently leaves the GLFW backend, which needs a display.
- Pool workers cache environments (`eval/runner.py`, `rally/run.py`). Keep the cache small: one rally environment per condition in 24 workers ran the 31 GB machine out of memory (the kernel's OOM killer ended the run). The rally runner keeps one per worker.
- Long sweeps save after each condition and resume (`scripts/rally.py`); launch them with `setsid nohup ... &` so they outlive the session.
- Stopping a run: `pkill -f "scripts/rally.py"` also matches the shell that runs the pkill (its command line contains the pattern) and kills it. Use a pattern it cannot match, e.g. `pkill -f "rally[.]py"`.
- The swing QP needs a horizon of at least 2 steps (`solveh_banded` fails on 1).
- Studio discovers message handlers by annotation: no `from __future__ import annotations` in `scripts/studio_batter.py`.
- Long runs: use `python -u`. The machine suspends when idle, which pauses jobs and pollutes wall-clock timings.
