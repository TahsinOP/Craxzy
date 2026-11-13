---
title: "Running It"
description: "Installing, watching the batter in MuJoCo Studio, reproducing every milestone, and rebuilding this site."
sidebar:
  order: 1
---

The shell on the development machine exports ROS Humble's Python path, so every command uses `env -u PYTHONPATH`. Everything runs on CPU.

## Install

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"      # versions are pinned in pyproject.toml
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -q   # 19 tests, about a minute
```

## Watch the batter think

```bash
env -u PYTHONPATH .venv/bin/python scripts/studio_batter.py               # native MuJoCo Studio window
env -u PYTHONPATH .venv/bin/python scripts/studio_batter.py --gfx web     # in the browser, http://localhost:8080
```

A Studio `SimPlugin` runs the batting cage (EKF, planner, trigger) and a `ViewerPlugin` draws what the controller believes: the ball's trail (white), the current forecast (green), the forecast the cached swing was built on (orange), the planned strike point (red) and a marker wherever the planner ran (yellow). A panel picks the trigger and its setting, adds surprises and sets the slow motion. Without the plugins, the scene alone:

```bash
env -u PYTHONPATH .venv/bin/python -m mujoco.studio.viewer --model=assets/batting_cage.xml --hot_reload=all
```

## Reproduce

```bash
scripts/run_all.sh          # every milestone, about 2 hours on a 28-thread desktop
scripts/run_all.sh 3        # from milestone 3 on
env -u PYTHONPATH .venv/bin/python scripts/make_reports.py   # results/M0.md ... M5.md and the README tables
```

| script | what it does | main arguments |
|---|---|---|
| `scripts/m0_cage.py` | pitch distribution; speculative contacts on/off × timestep; video | `--n` |
| `scripts/m1_hitter.py` | estimator accuracy; restitution fit; hit rate and accuracy against speed; planner timing; video | `--n` |
| `scripts/m2_worldmodel.py` | batting data, world-model training, learned against analytic planner; video | `--n-train --steps --n` |
| `scripts/m3_triggers.py` | PPO trigger training; sweeps of every trigger; Pareto plots | `--n --ppo-iters --ppo-batch` |
| `scripts/m4_surprises.py` | spin changes and gusts; reaction latency; video | `--n --ppo-iters` |
| `scripts/m5_events.py` | event-camera emulator and event trigger | `--n` |
| `scripts/design_ready.py` | the search behind the nominal strike posture and the ready pose | |

| `scripts/rally.py` | stretch: the two-arm ping-pong rally (feasibility, bounce study, calibration, rallies, video) | `--n` |

Every script takes `--set key.sub=value` to override `configs/default.yaml`, and `--video-only` where it makes a video.

## Videos

Videos are rendered offscreen with MuJoCo 4.0's **Filament** renderer (Vulkan; on the development machine, the integrated Intel GPU) through the same model scene MuJoCo Studio uses, with physically based materials and lighting set by `filament.*` custom flags (`eval/filament.py`). Pass `--renderer classic` for the classic renderer; if no Vulkan context can be made the scripts fall back to it.

```bash
env -u PYTHONPATH .venv/bin/python scripts/m1_hitter.py --video-only                      # Filament
env -u PYTHONPATH .venv/bin/python scripts/m1_hitter.py --video-only --renderer classic   # classic
``` Results go to `results/` (JSON, PNG, MP4, logs and reports), trained models and batting data to `data/`.

## This site

```bash
env -u PYTHONPATH .venv/bin/python docs/figures/make_figures.py   # figures from results/*.json, both themes
cp results/m0_tunnel.mp4 docs/public/video/tunnel.mp4              # and hitter, worldmodel, surprise
cd docs && pnpm install && pnpm dev                                 # http://localhost:4321
pnpm build                                                          # static site in docs/dist
pnpm build:offline                                                  # docs/dist-offline, opens from file://
```
