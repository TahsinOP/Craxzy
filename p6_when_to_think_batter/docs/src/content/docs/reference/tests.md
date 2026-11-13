---
title: "Tests"
description: "The 26 tests behind the claims on this site."
sidebar:
  order: 3
---

`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -q`

| test | claim |
|---|---|
| `test_flight.py::test_step_matches_mujoco_flight_model` | the flight model reproduces a simulated spinning pitch to 1e-9 m (velocity 1e-8, spin 1e-7) |
| `test_flight.py::test_same_flight_in_the_cage_model` | the same holds in the full scene with the arm present |
| `test_flight.py::test_spin_matters` | 200 rad/s of backspin moves the crossing point by more than 10 cm |
| `test_flight.py::test_jacobian_is_the_derivative_of_step` | the analytic Jacobian matches finite differences to 1e-7 |
| `test_flight.py::test_landing_matches_simulation` | the landing time and point of the ball-only model match the simulation to within one step |
| `test_ekf.py::test_crossing_point_converges` | the forecast is centimetres off early and under 8 mm in the last 50 ms |
| `test_ekf.py::test_filter_is_consistent` | the NIS has mean 3, the crossing-point ellipse covers 85 to 98 %, and the surprise detector fires on fewer than 3 % of ordinary pitches |
| `test_ekf.py::test_covariance_reset_on_a_gust` | the detector trips on most gusts and makes the covariance less over-confident |
| `test_planner.py::test_aimer_lands_on_the_target` | the aim solver lands within 5 cm of the target under the exact model |
| `test_planner.py::test_swing_reaches_the_strike_and_respects_limits` | the swing reaches the strike exactly, obeys the integrator and the limits, and brakes to rest |
| `test_planner.py::test_soft_swing_gets_as_close_as_the_limits_allow` | an unreachable strike gives the closest approach within the limits |
| `test_planner.py::test_strike_velocity_is_along_the_normal` | the strike velocity has the requested normal speed within the joint limits |
| `test_planner.py::test_planner_strike_geometry` | the planned pose puts the face at the ball, facing the impulse, and the impact model maps v_b to v_out |
| `test_planner.py::test_swing_parameterisation_round_trip` | the learned planner's swing parameters decode to the intended strike |
| `test_env.py::test_camera_sensor_is_delayed` | the MuJoCo delayed sensor shows the ball where it was 30 ms ago |
| `test_env.py::test_speculative_contacts_stop_the_ball_tunnelling` | every pitch bounces off a static bat with the flag on; fewer than 80 % without it |
| `test_env.py::test_batter_hits_and_landing_extrapolation_is_exact` | the oracle batter hits every pitch, and the continued simulation follows the extrapolated flight to 1e-8 down to 5 cm above the floor |
| `test_env.py::test_triggers` | fixed rates make the expected number of calls; replanning beats one plan; a gust adds calls to the innovation trigger and not to the variance trigger |
| `test_env.py::test_ppo_rollout_and_update` | PPO rollouts are well formed and an update changes the policy |
| `test_env.py::test_filament_look_model_matches_the_simulated_one` | the Filament look model has the simulated model's state layout and geometry, so a world's state can be copied across |
| `test_rally.py::test_bounce_model_is_the_pairs_restitution` | the identified bounce restitution is the table pair's |
| `test_rally.py::test_flight_through_a_bounce_matches_the_simulator` | the table-aware flight model follows the simulator through a bounce to 0.05 mm |
| `test_rally.py::test_jacobian_through_a_bounce` | its Jacobian through a bounce matches finite differences |
| `test_rally.py::test_arm_b_is_arm_a_in_its_own_frame` | arm B's paddle, in B's frame, is where A's kinematics put it, to 1e-9 m |
| `test_rally.py::test_serves_reach_arm_a` | every serve clears the net, bounces once on A's half and reaches A's strike zone |
| `test_rally.py::test_rally_with_the_true_ball_state` | two arms keep two rallies going for four returns each with the true ball state |
