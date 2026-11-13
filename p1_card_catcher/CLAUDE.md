# P1 card catcher: working notes for Claude Code

Read `TASK.md` for the brief and `README.md` for what exists and how to reproduce it.

## Environment
- Python 3.12 venv in `.venv` (uv). `mujoco==4.0.1`, `mujoco-menagerie`, CPU `torch`, `osqp`. Versions are pinned in `pyproject.toml`.
- **Always run with `env -u PYTHONPATH`**: the shell exports ROS Humble's Python 3.10 site-packages, which breaks imports and pytest plugin loading.
  - scripts: `env -u PYTHONPATH .venv/bin/python scripts/<name>.py`
  - tests: `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -q`
- Offscreen rendering uses EGL (`MUJOCO_GL=egl` is set by `card_catcher.eval.render`).
- Speculative contacts are CPU only in MuJoCo 4.0: everything multi-world goes through `mujoco.batch`, never MJX/Warp.

## Layout
- `src/card_catcher/{models,envs,control,learn,estim,eval}`, `scripts/` (one script per milestone), `configs/default.yaml`, `tests/`.
- `data/` (datasets, trained models) and `results/` (json, png, mp4) are git-ignored, except `results/*.md`.

## Conventions that matter here
- One discrete physics model for everything (`estim/physics.py`): MuJoCo's linearly-implicit step. Do not swap in closed-form ballistics or plain Euler; the tests show the error that causes (3 mm and 3 cm on the ball).
- Residual = simulated step minus that model, per step. Predictors, EKF and throw aiming all import the same functions.
- MuJoCo 4.0 API differences already hit: `mjData.M` replaces `qM`; `mj_fullM(m, d, dst)`; contact sensor via `MjSpec` uses `intprm = [data bits, reduce, num]`.
- Every milestone: a script that reproduces it, `results/Mx.md` with plots, 3 seeds (mean ± std), classical and pure-learning baselines next to the hybrid.
- Plots use `eval/plotstyle.py` (fixed colour per method: physics blue, pure learning orange, hybrid aqua).

## Gotchas found the hard way
- `scripts/view.py` is the live viewer (passive `mujoco.viewer`); keep the `viewer.lock()` and the short sleep after `close()`, it crashes at exit without them.
- "caught" is a contact-sensor test; "clean" also rejects interpenetration deeper than the object's thickness. Report both for the card, especially with speculative contacts off.
- Long runs: use `python -u` and `grep --line-buffered` when filtering OSQP's "Polishing" lines into a log. The laptop suspends when idle, which pauses jobs and pollutes wall-clock timings.
