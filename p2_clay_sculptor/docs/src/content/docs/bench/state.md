---
title: "State, Save and Restore"
description: "What a complete state of the clay bench is, why the plastic strain belongs to it, and the measured cost of leaving each component out when a planner rewinds the simulator."
sidebar:
  order: 4
---

:::note[Guiding question]
A planner that uses the simulator as its model rewinds it thousands of times: copy the state, try an action, put the state back. What exactly has to be copied for the second try to start from the same world as the first?
:::

## A state is one flat vector

MuJoCo defines the state through a bit field, `mjtState`. The combination `mjSTATE_INTEGRATION` is everything the integrator reads when it takes a step. For this scene it is a vector of 11,834 numbers:

| component | size | what it is here |
|---|---|---|
| `time` | 1 | simulation time |
| `qpos` | 1,528 | 1,521 node displacements of the clay and 7 joint angles |
| `qvel` | 1,528 | their velocities |
| `act` | 7 | the integrators of the seven `pid` servos |
| `warmstart` | 1,528 | the previous acceleration, a starting point for the constraint solver |
| `ctrl` | 14 | position and velocity setpoints of the seven joints |
| `qfrc_applied`, `xfrc_applied` | 4,636 | externally applied forces (zero here) |
| **`plastic`** | **2,592** | the plastic strain: 6 numbers for each of the $12 \times 12 \times 3$ cage cells |

Saving is `mj_getState(model, data, vector, mjSTATE_INTEGRATION)`, restoring is `mj_setState` with the same arguments. In this project a state *is* that vector: the planner holds states as rows of a NumPy array, and `mujoco.batch` holds one row per parallel world.

## Why the plastic strain is state

Positions and velocities are enough for an elastic body: its stress is a function of its current shape. A plastic body remembers. Each cage cell carries a plastic strain $\varepsilon_p$, and the stress depends on the *elastic* part of the deformation, what is left after $\varepsilon_p$ is taken out ([The Clay](/bench/clay/)). Two blocks with identical node positions and different $\varepsilon_p$ are different objects: one is at rest in that shape, the other is a spring compressed into it.

Restoring `qpos` and `qvel` and forgetting `plastic` therefore produces the second object. The dent is still drawn on screen, but the simulator now believes the clay is an undeformed block that someone is holding in a dented pose, and it lets go.

## The experiment

Press once. Save the state. Press somewhere else and record the result. Restore, press again in the same way, and compare.

| restored state | second run identical, bit for bit | largest node position difference | largest height difference |
|---|---|---|---|
| full `mjSTATE_INTEGRATION` | **yes** | 0 | 0 |
| `plastic` reset to the undeformed block | no | 3.36 mm | 3.31 mm |
| `qvel` set to zero | no | $9 \times 10^{-8}$ mm | $9 \times 10^{-8}$ mm |
| `warmstart` set to zero | yes | 0 | 0 |

The same experiment through `mujoco.batch` on sixteen threads and through plain `mj_step` on a single `MjData` also agree bit for bit, and so do two independent environments, one on one thread and one on all of them.

Three things follow.

**The plastic strain is not optional.** Leaving it out costs 3.4 mm, more than the whole error of a good plan. Nothing crashes and nothing warns; the rollouts are simply rollouts of a different block. This is the pitfall the project brief warns about, measured.

**Velocities barely matter, because every primitive ends at rest.** The 0.1 s settling time at the end of a primitive leaves the clay moving at well under a millimetre per second, so zeroing the velocities changes the next primitive by a tenth of a nanometre. That is a property of how the primitives are built, not of the simulator.

**The warm start does not change the result here.** With these solver settings a replay from a zeroed warm start reproduces the original bit for bit. It is still part of the saved vector, because that is a statement about this model and these settings, and a restore that is exact by construction needs no such argument.

:::caution[What "bit for bit" does and does not promise]
It holds on one machine, one build of MuJoCo, and any number of threads. It is not a promise across CPUs or library versions, where the order of floating-point operations can differ. The tests assert exact equality because every comparison in them runs in one process.
:::

## One layout for three tools

Each tool is a separately compiled model ([Scene and Tools](/bench/scene/)). The three models differ only in geoms welded to the robot's flange. They have the same joints, the same actuators and the same flex, so their state vectors have the same components at the same offsets. A state saved with the press is a valid state of the roller model.

Changing the tool is therefore not an operation on the simulator. The planner hands the same vector to a different model. This is what makes the three-tool target of [the stretch goals](/evaluation/stretch/) cheap: at every step the planner evaluates candidates for all three tools from one saved state.

## What a rollout resets

Before a primitive is rolled out, four things in the state are overwritten:

- the seven joint angles, set to the hover pose of the primitive;
- the joint velocities and the servo integrators, set to zero;
- the time, set to zero.

The clay part is untouched. The result of a rollout then depends on (clay state, tool, action) and on nothing else, which is what a dynamics model of the form $s' = f(s, a)$ assumes. The free-space move between two primitives is not simulated. Running whole sequences continuously instead, with those moves simulated, changes the final height map by at most 0.13 mm (0.03 mm RMS) after three primitives: the arm arrives with a little residual motion, and the clay has half a second longer to settle.

:::tip[Where in the code]
`envs/clay_env.py`: `StateLayout` (offsets of every component), `ClayEnv.get_state` / `set_state`, `prepare` (what a rollout resets), `rollout` (`mujoco.batch`), `step_reference` (plain `mj_step`), `run_continuous`.
Tests in `tests/test_bench.py`: `test_state_layout_matches_mujoco`, `test_tool_models_share_one_state_layout`, `test_save_restore_replays_bit_for_bit`, `test_restore_without_plastic_state_diverges`, `test_batch_equals_single_thread_bit_for_bit`.
Numbers: `scripts/m0_bench.py`, sections "restore" and "continuous" of `results/m0_bench.json`.
:::
