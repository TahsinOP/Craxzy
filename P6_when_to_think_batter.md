# P6: When-to-Think Batter

**Hit a pitched ball with a thin bat, and learn *when* to replan.**

## Goal
An arm swings a **thin bat** (a 1.2 cm geom, the kind of object speculative contacts were built for) to hit pitched balls toward a target. The controller is a model-based planner (MPC over a learned or known model) that is expensive to run. The research core is a **compute trigger**: a learned decision of *when* to re-run the expensive planner versus executing the cached plan, rewarded for hitting well with as few planner calls as possible.

This is your Q1 ("what to compute, and when") and Gehrig's "world models … when to act" in one toy that's still hard.

## Why it's a hybrid research question
- **Model:** ballistic ball model + known arm dynamics, giving an interception MPC.
- **Learned:** (a) a small world model (TD-MPC2-style latent model) as an alternative planner; (b) the **trigger policy**, a learned event condition analogous to event cameras and Lie Events, but for compute.
- **Compare trigger rules:**
  - fixed rate;
  - innovation-based: replan when the ball observation deviates from prediction by more than a threshold, the χ² gate idea;
  - variance-based (Trimpe);
  - learned.

## MuJoCo 4.0 features used
- `<flag speculative="enable"/>` with timesteps of 8–16 ms (see `model/speculative/flipper.xml`, a baseball and 1.2 cm bat at 10 ms). **CPU only**, so use `mujoco.batch` for parallel worlds.
- Sensor delays on ball observations (camera latency).
- Studio for visualising predicted versus true ball paths, using a custom `ViewerPlugin` that draws the plan and the trigger events.

## Milestones

**M0: Batting cage**
- Arm (Panda / UR5e) + bat + pitching machine (random speed, spin, height); target zone in the field.
- Show the bat passing through the ball with speculative contacts off, and hitting it with them on.
- *Accept when:* a video of both, and a pitch distribution.

**M1: Model-based hitter**
- Ball EKF → predicted crossing point and time → swing trajectory via a QP or trajectory optimisation (bat pose and velocity at impact chosen to send the ball toward the target).
- Replan at a fixed 100 Hz.
- *Accept when:* hit rate and target accuracy against pitch speed.

**M2: Learned world model planner**
- TD-MPC2 (or a small latent model + MPPI) trained on batting data. Compare against the analytic MPC on hit rate and planning time.

**M3: Compute triggers**
- Implement fixed-rate, innovation-triggered (χ² on ball-prediction innovation), variance-triggered, and learned (PPO policy observing innovation, covariance and time-to-impact; reward = hit quality − λ·planner calls) triggers.
- *Accept when:* Pareto curves of hit quality against planner calls, swept over λ and thresholds.

**M4: Surprises**
- Add mid-flight surprises (a spin change, a gust via `xfrc_applied`).
- Show that innovation-triggered and learned triggers react, while a fixed low rate misses.
- *Accept when:* the response latency to surprises is measured for each trigger type.

**Stretch (crazy)**
- Two-arm ping-pong rally (thin paddles, speculative contacts).
- An event-camera emulator: generate log-brightness-change events from offscreen renders, and trigger on events instead of state innovation.
- Use the Filament renderer for nice-looking videos.

## Evaluation
- Hit rate, distance to target, planner calls per pitch, compute time, latency to surprises; Pareto fronts.

## Papers
- Hansen, Su, Wang, *TD-MPC2* (ICLR 2024): https://arxiv.org/abs/2310.16828
- Sung, Kaelbling, Lozano-Pérez, *Learning When to Quit: Meta-Reasoning for Motion Planning* (IROS 2021): https://arxiv.org/abs/2103.04374
- Jayanth, …, Gehrig, *Neural Inertial Odometry from Lie Events* (RSS 2025), on event-triggered sampling: https://arxiv.org/abs/2505.09780
- Gehrig & Scaramuzza, *Low-latency automotive vision with event cameras* (Nature 2024): https://doi.org/10.1038/s41586-024-07409-w
- Trimpe & D'Andrea, *Event-Based State Estimation with Variance-Based Triggering* (IEEE TAC 2014): https://scholar.google.com/scholar?q=Event-Based+State+Estimation+with+Variance-Based+Triggering
- Heemels, Johansson, Tabuada, *An Introduction to Event-triggered and Self-triggered Control* (CDC 2012): https://scholar.google.com/scholar?q=An+introduction+to+event-triggered+and+self-triggered+control
- Williams et al., *MPPI*: https://scholar.google.com/scholar?q=Information+theoretic+MPC+for+model-based+reinforcement+learning+Williams

## Pitfalls
- Speculative contacts are CPU only, so budget the RL for the trigger accordingly. The trigger policy is tiny, so this is fine.
- Measure compute honestly: wall-clock per planner call on CPU, not just call counts.
- Keep the analytic MPC strong. A weak baseline makes the trigger results meaningless.
