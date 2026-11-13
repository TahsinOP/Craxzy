# P1: Card Catcher

A Franka Panda catches a thrown ball, plate and 0.7 mm playing card in MuJoCo 4.0, with a hybrid stack: a physics predictor and EKF, a learned aerodynamic residual with calibrated variance, a QP interception controller, and event-triggered replanning (periodic, variance-triggered, PPO-learned). The brief is in [`TASK.md`](TASK.md).

Everything runs on CPU. Speculative contacts are CPU-only in 4.0, so all multi-world work uses `mujoco.batch`.

## Results at a glance

| milestone | acceptance | result | report |
|---|---|---|---|
| M0 scene, throws, tunnelling | video shows tunnelling with the flag off and none with it on; 1,000 card trajectories | met: the card passes through a fingertip in 100 % of 600 throws with the flag off and 0 % with it on, at 8 and 16 ms; 3 × 1,000 throws per object | [`results/M0.md`](results/M0.md) |
| M1 physics + EKF | ball error < 1 cm at 200 ms before the catch | met: 0.35 ± 0.00 cm RMS (card 6.64 cm, plate 5.00 cm) | [`results/M1.md`](results/M1.md) |
| M2 learned residual | beats physics-only and pure learning on the card; variance calibrated | met: 3.22 ± 0.07 cm hybrid against 6.65 ± 0.06 (physics) and 5.10 ± 0.33 (pure learning) at 200 ms; residual interval coverage within 2.3 points of nominal | [`results/M2.md`](results/M2.md) |
| M3 interception | ball catch rate > 90 %; report the card | met: ball 97.3 ± 0.3 %; card 33.3 ± 2.1 % (hybrid), 10.0 ± 1.8 % (physics only), 47.0 ± 2.2 % (perfect predictor) | [`results/M3.md`](results/M3.md) |
| M4 replanning triggers | Pareto plot: catch rate against planner calls | met: on the ball the PPO trigger holds 96.9 ± 1.3 % with 4.6 planner calls per throw against 35.2 for the 50 Hz baseline (97.1 ± 0.7 %); slow periodic replanning is nearly as good (95.3 ± 1.7 % at 4.9 calls) and the learned trigger pulls ahead only below that | [`results/M4.md`](results/M4.md) |

All rates are mean ± std over 3 seeds. Every plot keeps the classical and the pure-learning baseline next to the hybrid.

What did not work, stated plainly:

- **Plate with the parallel gripper: 1 %.** A 16 cm disc does not fit an 8 cm jaw opening unless its rim is presented to the jaws. With the suction disc (stretch goal, 4.0 `adhesion`) it is retained in 87 % of throws.
- **The card tops out near 47 % even with a perfect predictor.** The sensor gives position only; the card arrives tumbling and only some attitudes slide between the jaws. Prediction quality moves the rate from 10 % to 33 %; closing the rest needs attitude sensing or a different effector.
- **Fixed-threshold event triggers (variance, drift) are worse than periodic replanning at the same budget on the ball.** At 10 calls per throw they catch 60 % and 74 % against 97 % for periodic. The likely reason is that they fire early and go quiet before the catch, when a fresh plan matters most.
- **On the card the learned trigger does not reach the 50 Hz catch rate**; it matches periodic replanning at the small budgets it chose to use.
- **The catch-point covariance of the hybrid is conservative far from the catch** (95–98 % inside the 90 % ellipse) and slightly optimistic in the last 50 ms (82 %).
- **Without speculative contacts the sensor-based catch test over-counts the card.** It reports 69 % caught at 8 ms with the flag off against 46 % with it on, but the "caught" cards sit a median 2.6 mm inside a finger (the card is 0.7 mm thick). Counting only catches without such overlap gives 43.8 ± 0.6 % on against 26.8 ± 3.9 % off at 8 ms, and 40.6 ± 1.1 % against 12.2 ± 1.6 % at 16 ms. For the ball the flag matters only at 16 ms: 99.2 % on, 1.8 % off. By the same stricter test the headline card rate is 30.5 % instead of 33.3 %.

## Run it

The shell on this machine exports ROS Humble's Python path, so every command needs `env -u PYTHONPATH`.

```bash
uv venv --python 3.12 && uv pip install -e .          # once; versions are pinned in pyproject.toml

# watch catches live in the MuJoCo viewer
env -u PYTHONPATH .venv/bin/python scripts/view.py --obj ball --throws 10 --slow 4
env -u PYTHONPATH .venv/bin/python scripts/view.py --obj card --predictor hybrid

# reproduce every milestone (about 2.5 h on a 16-thread laptop), or start from milestone N
scripts/run_all.sh
scripts/run_all.sh 3

# regenerate results/M0.md ... M4.md from the JSON results
env -u PYTHONPATH .venv/bin/python scripts/make_reports.py

# tests (18)
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -q

# the scene in MuJoCo Studio
env -u PYTHONPATH .venv/bin/python -m mujoco.studio.viewer --model=assets/catch_card.xml --hot_reload=all
```

`assets/catch_card.xml` holds absolute mesh paths for this machine; `card_catcher.models.scene.export_xml` regenerates it elsewhere.

| script | what it does | main arguments |
|---|---|---|
| `scripts/m0_mount.py` | upright against inverted mount: reachable palm-up poses | |
| `scripts/m0_dataset.py` | throw datasets, drag and flutter checks | |
| `scripts/m0_tunnel.py` | speculative contacts on/off × timestep, video | |
| `scripts/m1_ekf.py` | drag fit, EKF tuning, error against time to catch | |
| `scripts/m2_residual.py` | trains residual and pure-learning models, calibration, data sweep | `--n-train --steps --sweep --objects` |
| `scripts/m3_catch.py` | catch rates, predictor ablation, contact ablation, suction stretch, videos | `--n --n-ablation --objects` |
| `scripts/m4_triggers.py` | trigger sweeps, PPO training, Pareto plot | `--n --ppo-iters --ppo-batch --objects` |
| `scripts/view.py` | live viewer | `--obj --predictor --rate --slow` |

Outputs go to `results/` (JSON, PNG, MP4, and the `Mx.md` reports) and `data/` (datasets, trained models).

## How it works

```
position fixes (125 Hz, σ = 3 mm, 10 % dropout)
   └─ EKF on [p, v, k_d] with the simulator's own discrete step            estim/ekf.py, physics.py
        └─ GRU residual: mean and log-variance of the unmodelled
           acceleration for the next 0.9 s (Gaussian NLL)                   learn/
             └─ forecast to the catch plane: point, time, covariance        estim/predictor.py
                  └─ trigger: replan now?  (periodic / variance / drift / PPO)   control/triggers.py
                       └─ IK → trajectory QP → discrete computed torque     control/arm.py, planner.py
                            └─ mujoco.batch worlds                          envs/catch.py
```

- **One discrete physics model.** MuJoCo's `implicitfast` step is linearly implicit: v' = v + h (I + hK)⁻¹(g − k_d|v|v), with K = k_d|v|(I + v̂v̂ᵀ). `estim/physics.py` has its closed form and analytic Jacobian. It matches MuJoCo to 1e-9 m over a 0.8 s flight; plain Euler is off by 3 mm and continuous-time ballistics by 3 cm. The residual is defined per step against this model, so it contains no discretisation error.
- **Residual and its uncertainty.** The network outputs a mean and a variance per future step. The catch-point covariance treats the residual error as AR(1) along the horizon (φ ≈ 0.92, fitted) because treating it as white leaves the 90 % ellipse covering about half of the outcomes. The filter's learned process noise is inflated by κ = 4, chosen by maximum likelihood of the innovations.
- **Planner.** Per joint, a QP in the accelerations from the current state to rest at the IK goal at the arrival time, under position, velocity and acceleration limits, written in the simulator's integrator. Closed form when no limit is active, OSQP otherwise; an analytic bang-coast-bang time gives reachability. Validated against an LP solver in the tests.
- **Tracking.** τ = (M + hD)a + c − f_passive with gains from pole placement on the sampled error dynamics.
- **Triggers.** The variance trigger fires when tr P(t_plan) − tr P(t) > δ²: for a consistent estimator that difference is the expected squared distance between the planned and the current aim point. The learned trigger is one PPO policy conditioned on λ in reward = catch − λ · replans, so sweeping λ traces its whole curve.

## Decisions that differ from a literal reading of the brief

- **The Panda hangs inverted from a gantry.** Objects arrive from above, so the palm must face up and forward. An upright Panda has a usable pose for that in 13 % / 2 % / 0 % of window positions at 20° / 40° / 60° palm elevation, against 53 % / 64 % / 60 % inverted (`scripts/m0_mount.py`).
- **Torque motors** replace the Menagerie position servos, at the datasheet torque limits. The planner uses datasheet acceleration limits as a stand-in for torque limits; saturation occurs in fewer than 0.2 steps per throw.
- **The catch window is ±20 cm by ±14 cm on a plane tilted 30°.** A larger window lost 10 % of ball catches with a perfect predictor because its edges were not reachable in time.
- **The ball is small and soft** (r = 25 mm, 40 g, low restitution) so that a parallel gripper can hold it.
- **"Caught" is judged by contact sensors half a second after the catch**: both fingers touching, the object within 8 cm of the tool centre and off the ground. A second column, "clean", also requires that nothing overlaps the object by more than its own thickness, because a card can lodge inside a finger and pass the sensor test.
- **A drift trigger** is included next to the three required ones, as the realised-change counterpart of the variance trigger.

## How the referenced papers are used

- **Johannink et al., Residual RL** and **O'Connell et al., Neural-Fly**: the residual structure. Physics carries the bulk; the network learns only the unmodelled aerodynamic acceleration, and it enters the estimator as a known input.
- **Kendall & Gal**: the heteroscedastic Gaussian NLL head that gives the residual its variance.
- **Kim, Shukla, Billard, Catching Objects in Flight**: the formulation as predicting a catch point and time on a reachable set and re-aiming as the estimate improves.
- **Forrai et al., Event-based Agile Object Catching**: the setting of catching fast objects from low-latency position measurements, and reporting performance against the time left before impact.
- **Trimpe & D'Andrea**: triggering on covariance rather than on data, which is the variance trigger.

## What to scale up on a server

All budgets are arguments. Worth raising: `m2_residual.py --n-train` and `--steps` (the data sweep stops at 4,000 throws), 3 seeds for the plate, `m3_catch.py --n`, and `m4_triggers.py --ppo-iters --ppo-batch` (the card policy trained for 60 iterations of 96 throws). From the stretch list only the suction cup is done; two-arm juggling and the paper plane are not. Attitude sensing for the card is the most useful next step.
