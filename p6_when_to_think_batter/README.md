# P6: When-to-Think Batter

A Franka Panda swings a 1.2 cm thick bat at pitched baseballs in MuJoCo 4.0 and sends them towards a target on the ground. The planner that builds each swing is expensive, so the question is *when* to run it: at a fixed rate, when the camera contradicts the plan's forecast, when the estimate has become more certain, or when a learned policy says so. The brief is in [`TASK.md`](TASK.md).

Everything runs on CPU. Speculative contacts are CPU-only in 4.0, so all multi-world work uses `mujoco.batch`.

## Results at a glance

<!-- results:begin -->
| milestone | acceptance | result | report |
|---|---|---|---|
| M0 batting cage | video of the bat passing through the ball with speculative contacts off and hitting it with them on; pitch distribution | met: with the flag off the ball passes through a bat held still in 50 to 75 % of pitches (8, 10, 16 ms); with it on it bounces off in 100 % | [`results/M0.md`](results/M0.md) |
| M1 model-based hitter | hit rate and target accuracy against pitch speed | met: 100.0 ± 0.0 % hit, 97.8 ± 0.2 % inside the 0.75 m target zone at 100 Hz replanning; the same planner called once hits 11.8 ± 1.8 % | [`results/M1.md`](results/M1.md) |
| M2 learned world-model planner | compare with the analytic MPC on hit rate and planning time | done: the learned planner hits 96.9 ± 4.4 % but lands 54.9 ± 9.1 % on target against 96.9 ± 0.0 %; 8.7 ms per call against 9.3 ms | [`results/M2.md`](results/M2.md) |
| M3 compute triggers | Pareto curves of hit quality against planner calls | met: every step costs 48 calls per pitch for quality 0.98; the learned trigger holds 0.94 ± 0.00 with 4.0 calls (fixed rate: 0.91 ± 0.01 with 4.4) and 0.81 ± 0.01 with 2.9 (fixed rate: 0.58 ± 0.03 with 2.9) | [`results/M3.md`](results/M3.md) |
| M4 surprises | response latency to surprises for each trigger type | met: a gust makes the learned trigger spend +1.8 planner calls and the innovation trigger +1.8; fixed-rate and variance triggers do not change their schedule (+0.0, +0.0). Hit rate under gusts at about 4 calls: learned 92 %, fixed rate 79 %. No trigger replans sooner because of a spin change | [`results/M4.md`](results/M4.md) |
| Stretch: event camera | event-camera emulator; trigger on events instead of state innovation | done: within 150 ms of a gust the event trigger calls the planner in 48 % of pitches (placebo 30 %), the frame-based innovation trigger in 31 % (placebo 21 %) | [`results/M5.md`](results/M5.md) |
| Stretch: ping-pong rally | two-arm rally, thin paddles, speculative contacts | done: 10.0 returns per rally with the filter (63 % reach 12), 11.5 with the true ball state; confidence-gated replanning doubled it from 4.5; at 12.5 Hz 9.2 returns with 10 planner calls per return instead of 60; speculative contacts off: 0.6 | [`results/RALLY.md`](results/RALLY.md) |
| Stretch: Filament videos | use the Filament renderer for nice-looking videos | done: every milestone video is rendered offscreen with MuJoCo 4.0's Filament renderer (Vulkan), with physically based materials; `--renderer classic` for the classic one | `src/batter/eval/filament.py` |
<!-- results:end -->

All rates are mean ± std over 3 seeds. Every Pareto plot keeps the classical baseline (analytic planner at a fixed rate) and the pure-learning baseline (world-model planner) next to the hybrid (analytic planner with a learned trigger).

## What did not work, stated plainly

<!-- caveats:begin -->
- **The classical event triggers lose to a fixed rate on ordinary pitches.** At about 4 planner calls per pitch the innovation trigger reaches a hit quality of 0.45, the variance trigger 0.29 (with 3.0 calls) and the drift trigger 0.44 (3.4 calls), against 0.91 for a fixed rate. Variance and drift fire early, while the estimate still moves by centimetres, and are silent late, when millimetres matter; the innovation trigger fires on noise, because without a surprise the camera does not contradict the forecast, it only sharpens it.
- **The learned trigger does not go below about 2.5 calls per pitch.** At the highest price (λ = 0.2) it still spends 2.5 calls for a quality of 0.68; the corner of the curve below that was not reached.
- **Reacting to a surprise does not buy back the quality.** At about 4 calls per pitch a gust takes the fixed rate from 0.91 to 0.47, the learned trigger trained with surprises from 0.94 to 0.64, and the innovation trigger from 0.46 to 0.31. Even replanning at every step loses quality on surprised pitches (spin change: 0.95, gust: 0.78), because the estimator needs frames to see the change and the arm needs time to move. What the reaction buys is modest: with the 5.9 calls it then uses, the learned trigger's 0.64 compares with 0.58 for a fixed rate at 6.5 calls.
- **A spin change is hard to see.** It bends the path gradually, so in the 150 ms after it the innovation trigger replans in 28 % of pitches against 29 % on the same pitches without it (gust: 35 % against 29 %). The EKF's own surprise detector (a χ² gate on its innovations) trips on a majority of gusts and on few spin changes (`tests/test_ekf.py`); when it trips it makes the covariance honest, not the estimate better.
- **The learned planner hits but does not aim.** 96.9 ± 4.4 % hit rate, 54.9 ± 9.1 % on target. Its outcome model predicts the landing point to about 1.0 m, which is the size of the target. It also needed data from the analytic batter; with a sixteenth of the data it hits 64.8 ± 1.1 %.
- **The planner is not real-time at 100 Hz.** A call takes 9.1 ms (median) and 21.5 ms (90th percentile) on one core, against a 10 ms control period, and the simulation does not charge for that time: a plan is applied in the step it was asked for. `EnvParams.latency_steps` delays plans by whole steps; it is implemented and not evaluated.
- **Events trigger sooner, but that does not make a good trigger.** At about 4 calls per ordinary pitch the event trigger's quality is 0.47 (0.37 under gusts), the frame-based innovation trigger's 0.47 (0.30), a fixed rate's 0.89 (0.44). The planner the events wake up still runs on the EKF, which is fed by frames that are 30 ms old; the events would have to feed the estimator for the earlier trigger to pay.
- **The rally needs a midi table, a long handle and a frictionless table.** On an ITTF table a return needs 3.0 m/s of paddle speed and the Panda gives 1.6; the rally is played on a 1.8 m table with a 0.30 m handle. With table friction, MuJoCo's one-step bounce returned the ball faster than it arrived in 67 % of bounces, so the table is frictionless (no spin kick off the table).
- **The filter still costs the rally about 1.6 returns** (10.0 against 11.5 with the true ball state). Most rallies that end early end in the net or on the hitter's own half: the return is slightly slow or low. Both arms run the same controller; there is no adversarial play.
- **Simplified:** the bat and paddles do not collide with the arms, and joint limits during the follow-through are left to MuJoCo's limit constraints.
<!-- caveats:end -->

## Documentation site

`docs/` is an Astro Starlight site that explains the whole project chapter by chapter, with every figure regenerated from the result files: the problem, the cage, the estimator, the planner, the learned planner, the triggers, surprises and events, and a chapter of **lessons and shortcomings**. Build it with `cd docs && pnpm install && pnpm dev` (or `pnpm build:offline` for a copy that opens from `file://`); see `docs/src/content/docs/reference/running-it.md`.

## Run it

The shell on this machine exports ROS Humble's Python path, so every command needs `env -u PYTHONPATH`.

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"     # once; versions are pinned in pyproject.toml

# watch the batter think in MuJoCo Studio: forecast, cached plan, planner calls; pick the trigger in the panel
env -u PYTHONPATH .venv/bin/python scripts/studio_batter.py               # native window
env -u PYTHONPATH .venv/bin/python scripts/studio_batter.py --gfx web     # browser, http://localhost:8080

# reproduce every milestone (about 2 h on a 28-thread desktop), or start from milestone N
scripts/run_all.sh
scripts/run_all.sh 3

# regenerate results/M0.md ... M5.md and the tables in this file from the JSON results
env -u PYTHONPATH .venv/bin/python scripts/make_reports.py

# tests (26, about a minute and a half)
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -q
```

| script | what it does | main arguments |
|---|---|---|
| `scripts/m0_cage.py` | pitch distribution; speculative contacts on/off × timestep; video | `--n` |
| `scripts/m1_hitter.py` | EKF accuracy; impact calibration; hit rate and accuracy against pitch speed; planner timing; video | `--n` |
| `scripts/m2_worldmodel.py` | collects batting data, trains the world model, compares the two planners; video | `--n-train --steps --n` |
| `scripts/m3_triggers.py` | trains the PPO trigger, sweeps all triggers, Pareto plots | `--n --ppo-iters --ppo-batch` |
| `scripts/m4_surprises.py` | spin changes and gusts: Pareto, reaction latency; video | `--n --ppo-iters` |
| `scripts/m5_events.py` | stretch: event-camera emulator and event trigger | `--n` |
| `scripts/rally.py` | stretch: two-arm ping-pong rally (feasibility, bounce study, calibration, rallies, video) | `--n --fresh` |
| `scripts/studio_batter.py` | MuJoCo Studio `SimPlugin` + `ViewerPlugin` | `--gfx web` |
| `scripts/design_ready.py` | the search behind the ready pose | |

Every script takes `--set key.sub=value` to override `configs/default.yaml` (the rally: `configs/rally.yaml`), and `--video-only` and `--renderer {filament,classic}` where it makes a video. Videos are rendered with MuJoCo 4.0's Filament renderer by default. Outputs go to `results/` (JSON, PNG, MP4, logs, and the `Mx.md` reports) and `data/` (batting data, trained models).

## How it works

```
camera fix (100 Hz, 30 ms late through a MuJoCo sensor delay, noisy, 5 % dropped)
  └─ EKF on [position, velocity, spin], process model = the simulator's flight step     estim/ekf.py, flight.py
       └─ forecast to the plate with covariance                                          estim/ekf.py
            └─ trigger: run the planner now, or keep the cached swing?                   control/triggers.py, learn/ppo.py
                 └─ planner (expensive)                                                  control/planner.py  |  learn/wm_planner.py
                      aim → impact model → 5-dof IK → joint swing QPs                    control/arm.py, traj.py
                      └─ discrete computed torque on the cached swing                    control/arm.py
                           └─ mujoco.batch worlds, speculative contacts                  envs/cage.py
```

- **One flight model.** With the `Euler` integrator the ball's step has a closed form (drag, lift from spin, spin decay); the coefficients are read off the engine. It matches MuJoCo to 1e-9 m and is used by the pitching machine, the EKF, the forecast and the landing point.
- **Hidden spin.** The machine is zeroed in, so spin does not move where a pitch arrives; it changes the path. Early in the flight a floating and a diving ball look alike, which is why the first forecast is off by tens of centimetres and why replanning matters.
- **Impact.** The bat-ball pair is frictionless with a direct `solref` (zero stiffness, damping (1+e)/h), the form the 4.0 notes on speculative contacts ask for. The planner inverts v_out = v_b − (1+e)((v_b − v_c)·n)n for the face normal and bat speed that land the ball on the target; the effective e is fitted in M1.
- **Swing.** Per joint, a QP in the simulator's integrator: approach, constant velocity through the impact, brake. Closed form when no limit is active, OSQP otherwise, and a soft version that returns the closest approach when the strike cannot be reached in time.
- **Triggers.** Innovation: χ² of the camera against the forecast the plan was built on. Variance: tr C(plan time) − tr C(now) > δ². Drift: the crossing-point estimate moved by more than δ. Learned: one PPO policy conditioned on λ in reward = hit quality − λ · planner calls.
- **Learned planner.** Encoder, latent dynamics, outcome head and policy prior trained on batting data; MPPI in the latent space over six swing parameters. It sees raw camera fixes, not the EKF.

## Decisions that differ from a literal reading of the brief

- **The bat is a flat blade**, 1.2 cm thick, 9 cm wide, 38 cm long, not a round bat. Thin in the direction of the impact is what speculative contacts are for, and a flat face makes "bat pose chosen to send the ball toward the target" a well-posed problem: the face normal sets the direction.
- **Pitches are slow and from close by** (9 to 13 m/s from 6.5 m, 0.48 to 0.76 s of flight). The Panda's joint speed limits cap the bat at about 2 m/s; the scenario is sized to what the arm can do.
- **The bat-ball contact is frictionless and bouncier than a real one** (pair restitution 0.8, effective 0.585 once the wrist gives). With the bat speed available, that is what carries a ball 3.5 to 6 m.
- **Torque motors** replace the Menagerie position servos, at the datasheet torque limits. The bat collides with the ball only; it can pass through the arm.
- **"Replan at 100 Hz" means every simulator step** (10 ms). The Python planner takes about as long as one step, so 100 Hz is not real-time in this implementation; M1 to M3 report wall-clock next to call counts.
- **M2 is the "small latent model + MPPI" option, not full TD-MPC2.** Its actions are swing parameters, not joint torques, and it has no bootstrapped value function (the reward arrives once, at the end of a pitch). It is trained on data from a noisy analytic batter.
- **A drift trigger** is included next to the required ones, as the realised-change counterpart of the variance trigger.
- **A pitch ends a few steps after the impact**; the landing point is extrapolated with the flight model, which a test checks against the continued simulation.
- **The sensor delay is a MuJoCo 4.0 delayed `framepos` sensor**; noise and dropouts are added in Python.

- **The rally is played on a midi table (1.80 × 1.00 m) with a 0.30 m paddle handle.** On an ITTF table a return needs about 3 m/s of paddle speed; the Panda gives 1.6 m/s (`results/RALLY.md`).
- **The rally's table is frictionless**, like the bat and the paddles. With friction, MuJoCo's one-step restitution contact returned the ball faster than it arrived in two bounces out of three. Frictionless, the bounce is an exact restitution law, and the flight model stays exact through it.
- **The rally replans only when the filter is confident** (vertical-speed standard deviation under 0.08 m/s), and within 60 ms of an impact only the same stroke may be corrected. Both rules came out of measuring why rallies ended: the commit rule took true-state rallies from 1.3 to 12 returns (on 8 test rallies), and the gate took filter-based rallies from 4.5 to 10.0 (144 rallies).

## How the referenced papers are used

- **Hansen, Su, Wang, TD-MPC2:** the structure of the learned planner: encoder, latent dynamics trained by consistency, a reward-like outcome head, a policy prior that seeds MPPI in latent space.
- **Williams et al., MPPI:** the sampling planner inside it.
- **Sung, Kaelbling, Lozano-Pérez, Learning When to Quit:** the framing of M3 as meta-reasoning: a learned policy that decides whether more planning is worth its cost, trained with that cost in the reward.
- **Trimpe & D'Andrea, variance-based triggering:** the variance trigger, which fires on covariances and never on data.
- **Heemels, Johansson, Tabuada:** the send-on-delta (drift) trigger and the periodic baseline it is measured against.
- **Jayanth et al., Lie Events; Gehrig & Scaramuzza, low-latency vision with event cameras:** triggering on change instead of on a clock, and in M5 an emulated event camera whose events arrive before the frame camera's delayed fix.

## What to scale up on a server

The rally: `scripts/rally.py --n` (48 rallies per seed and condition) and a learned trigger for it, which was not trained; the confidence gate is the hand-made version of what M3's PPO trigger learned for the batter.


All budgets are arguments. Worth raising: `m2_worldmodel.py --n-train --steps` (the learned planner is still improving with data), `m3_triggers.py --ppo-iters --ppo-batch` and the number of evaluation pitches `--n` everywhere. Open ends: feed the event camera into the estimator, not only the trigger; a trigger policy that also decides when to make the *first* plan; a planner with a tunable effort (number of candidates) so that the trigger can choose how hard to think, not only whether to.
