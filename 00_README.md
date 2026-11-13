# Hybrid Sim Projects: MuJoCo 4.0 + Playground 1.0

**Theme for every project:** *model what you can, learn the rest, and know when to trust each.*
Each folder-level `.md` is a self-contained brief for Claude Code. Open a fresh repo, drop the file in as `TASK.md`, and tell Claude Code: "Read TASK.md and execute milestone M0."

## The stack (as of November 2026)

- **MuJoCo 4.0.x** (released 6 Nov 2026; 4.0.1 on 11 Nov). New features these projects use:
  - **Speculative contacts:** `<flag speculative="enable"/>`. Fast or thin objects don't tunnel through each other at large timesteps. **CPU only in 4.0**; MJX, Warp, heightfields, SDFs and flexes come later.
  - **Mature deformables:**
    - implicit `integrator="discrete"`;
    - penetration-free IPC cloth (`<flag ipc="enable"/>`);
    - 1D elastic rods (`flexcomp dim="1"`);
    - **plasticity**: `springyield` on joints and tendons, `<elasticity yield=...>` on shells and solids; the plastic state lives in `mjData.plastic`.
  - **New actuators and sensors:** `pid`, datasheet `dcmotor`, SO(3) `orientation` actuator, **actuator/sensor delays**, **`tactile`**, `contact`, camera `rangefinder`.
  - **New contact features:** per-contact `surfacevel` (conveyors) and `adhesion` (suction, magnets).
  - **Tools:** `mujoco.sysid` (system identification; `pip install "mujoco[sysid]"`), `mujoco-menagerie` on pip (about 70 robots), and **MuJoCo Studio**: `python -m mujoco.studio.viewer --model=x.xml --hot_reload=all`, or `--gfx=web` for a browser view.
  - **Install:** `pip uninstall -y mujoco && pip install -U mujoco`. Python 3.11+ (3.10 support was dropped).
- **MuJoCo Playground 1.0**, released alongside 4.0. MJX and **MuJoCo Warp** backends, RSL-RL and Brax PPO trainers, play-back in the Studio web viewer.
  - Quickstart: `uvx --from "playground[rl]>=1.0" python -m mujoco_playground.rl.rsl_rl.train --env=CartpoleBalance --logdir=/tmp/x`
  - Older CLI: `train-jax-ppo --env_name <Env> --impl warp`
  - Use `impl="warp"` for contact-heavy scenes.

> **Rule of thumb:** anything that needs **speculative contacts or deformables** runs on **CPU MuJoCo** (`mujoco.batch` for multiple worlds) with a small RL budget, or uses model-based control or MPC. Anything that needs **millions of RL steps** (locomotion) runs in **Playground on GPU**.

## Projects

| # | File | One line | Robot | Hybrid pattern | Main 4.0 feature | Crazy level |
|---|---|---|---|---|---|---|
| 1 | `P1_card_catcher.md` | Catch a fluttering playing card mid-air | Panda / UR5e | Physics predictor + learned aero residual + event-triggered replanning | Speculative contacts, fluid model | 🔥🔥🔥 |
| 2 | `P2_clay_sculptor.md` | Press clay into a target shape | Panda + tool | Learned plastic dynamics + MPC | Plasticity (solids) | 🔥🔥🔥🔥 |
| 3 | `P3_chicken_head_duck.md` | Open Duck Mini keeps its head camera still while walking, like a bird, so visual odometry drifts less | Open Duck Mini v2 | Model-based gaze stabilisation (IMU + IK) + learned predictive residual + RL gait | `orientation` actuator, delays, `dcmotor`, offscreen cameras | 🔥🔥🔥 |
| 4 | `P4_humanoid_learn_to_fall.md` | A humanoid that predicts an inevitable fall and falls safely | Unitree G1 / Booster T1 | Capture-point fall prediction + learned switch + RL safe-fall policy | Plasticity as a damage metric, contact/tactile sensors | 🔥🔥🔥🔥 |
| 5 | `P5_holonomic_swarm.md` | 4→256 holonomic robots learn coverage, formation, sorting, gate passage and herding with one policy | Holonomic disc robots | Classical swarm controllers (Lloyd, consensus, CBF) + GNN residual + task embedding | `<replicate>`, MuJoCo Warp at scale, `surfacevel`, `adhesion` | 🔥🔥🔥🔥 |
| 6 | `P6_when_to_think_batter.md` | Hit a pitched ball; learn *when* to replan | Arm + thin bat | World model / MPC + learned compute trigger | Speculative contacts | 🔥🔥🔥🔥 |

**Suggested order:**
- Start with **P1** (a fast win, and shows off speculative contacts).
- Then **P3** or **P4** (locomotion RL on GPU).
- Then **P5** (scales to many robots, entirely on GPU).
- **P2** and **P6** are the "crazy" research-grade ones.

## Shared conventions (put in each repo's CLAUDE.md)

- Python ≥ 3.11, `uv` for environments, `mujoco>=4.0.1`, `playground>=1.0`.
- Code layout:
  - `src/<pkg>/{models,envs,control,learn,estim,eval}`
  - `assets/` (MJCF)
  - `scripts/`
  - `configs/` (YAML)
  - `results/` (git-ignored)
- Every milestone ends with:
  - a script that reproduces its result;
  - a short `results/Mx.md` with plots;
  - a Studio or video recording, for example `python -m mujoco.studio.viewer --gfx=web` or offscreen rendering to mp4.
- Seed everything; report mean ± std over at least 3 seeds.
- Always keep the **classical baseline** and the **pure-learning baseline** next to the hybrid in every plot.
- Pin versions in `pyproject.toml`. Check API names against the 4.0 docs (mujoco.readthedocs.io), because 4.0 has breaking changes: `<composite>` was removed, `mujoco.experimental.studio` became `mujoco.studio`, and 1D rods need `integrator="discrete"`, `Euler` or `RK4`.
