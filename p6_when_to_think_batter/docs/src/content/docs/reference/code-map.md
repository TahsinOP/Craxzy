---
title: "Code Map"
description: "Where each idea lives in the source tree."
sidebar:
  order: 2
---

| module | contents |
|---|---|
| `models/scene.py` | the cage built with `MjSpec`: Panda with torque motors, bat, ball, contact pair, delayed camera sensor, contact sensors, target mocap; `export_xml` for Studio |
| `models/pitch.py` | the pitching machine: random pitches, release solved through the flight model, surprises stored with the pitch |
| `estim/flight.py` | the ball's flight model: identification from the engine, step, Jacobian, rollout, landing |
| `estim/ekf.py` | the EKF on delayed fixes, its surprise detector, the forecast and the crossing point with covariance |
| `control/arm.py` | kinematics, 5-dof strike IK, strike velocity, discrete computed torque, the nominal strike and ready poses |
| `control/traj.py` | the swing: joint QPs (closed form, OSQP, soft), constant-velocity strike, brake |
| `control/planner.py` | the analytic planner: crossing, aiming, impact inversion, candidate search |
| `control/triggers.py` | fixed-rate, innovation, variance, drift and learned triggers |
| `envs/cage.py` | the batched environment on `mujoco.batch`: sensing, plan caching, the trigger statistics, surprises, outcomes |
| `learn/ppo.py` | the trigger policy, its features, PPO rollouts and updates |
| `learn/wm.py` | the latent world model: encoder, dynamics, outcome head, prior, training |
| `learn/wm_planner.py` | swing decoder, data-collecting planner, MPPI world-model planner |
| `eval/runner.py` | the process pool, trigger specs, evaluation, uncontended planner timing |
| `eval/sweeps.py` | swept settings, PPO training, evaluation rows shared by M3 and M4 |
| `eval/eventcam.py` | the event-camera emulator, the event trigger, software-rendering selection |
| `eval/render.py` | offscreen rendering, the frame recorder with forecast overlays, video writing, `make_recorder` |
| `eval/filament.py` | Filament videos: the look model (materials, lighting flags) and a recorder with the same overlays |
| `rally/scene.py` | the rally scene: two Pandas with paddles, table, net, ball; mirroring into arm B's frame |
| `rally/flight.py` | the bounce model, identified from simulated bounces, and the bounce-aware flight model |
| `rally/planner.py` | the rally planner: strike candidates aimed at the far half, net and reach checks |
| `rally/env.py` | the batched rally, its referee, the filter restart after each hit, the confidence gate |
| `rally/serve.py`, `rally/poses.py`, `rally/run.py`, `rally/video.py` | the feeder, the paddle's poses, pool evaluation, the Filament rally video |
| `scripts/studio_batter.py` | MuJoCo Studio plugins |
