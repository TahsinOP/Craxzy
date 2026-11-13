# P1: Card Catcher

**Catch a fluttering playing card in mid-air with a robot arm.**

## Goal
A 7-DoF arm (Menagerie `franka_emika_panda` or `universal_robots_ur5e`) catches thin objects thrown toward it, in increasing difficulty: a ball, then a plate, then a **0.7 mm playing card** that flutters because of aerodynamics. The catcher is hybrid:

1. **A physics predictor:** ballistic motion plus a drag model, tracked with an EKF from noisy position observations.
2. **A learned residual:** a model of the aerodynamic error (flutter) that the physics doesn't capture.
3. **An interception planner:** model-based, with a QP or trajectory optimisation, that moves the gripper to the predicted catch point.
4. **A learned "when to replan" trigger:** replan only when the prediction has drifted, not at every step.

## Why it's a hybrid research question
- The ballistic model is right on average but wrong in the details (flutter). This is a textbook case for *physics prior + learned residual* (in the spirit of Neural-Fly).
- The question "how much should I trust my predictor right now?" maps directly onto the **covariance-gated authority** idea from your transport work.
- Replanning on prediction error rather than at a fixed rate is the **event-triggered compute** idea, which ties to Gehrig's group.

## MuJoCo 4.0 features used
- `<option timestep="0.008"><flag speculative="enable"/></option>`. DeepMind reports that an arm catches a 0.7 mm card at 2 m/s with a 16 ms timestep using speculative contacts. **CPU only.**
- MuJoCo's **fluid model** on the card geom (`fluidshape="ellipsoid"`, `fluidcoef`), which gives lift and drag and therefore flutter. Use `<option density="1.2" viscosity="1.8e-5"/>`.
- The `adhesion` geom attribute as an optional suction-cup end effector (a "cheat mode" for comparison).
- MuJoCo Studio with `--hot_reload=all` while tuning the throw.

## Milestones

**M0: Scene and throws (CPU)**
- Build an MJCF with the arm from `mujoco-menagerie`, a table and a throw generator: random initial position, velocity and spin for the ball, plate and card.
- Turn on speculative contacts and verify the card doesn't tunnel through the gripper at timesteps of 8 and 16 ms. Repeat with the flag off.
- *Accept when:* a video shows tunnelling with the flag off and none with it on; you have a dataset of 1,000 card trajectories with fluid forces on.

**M1: Physics predictor + EKF**
- State `[p, v]` (plus a drag coefficient); measurement = noisy position at 60–120 Hz, plus dropout.
- Predict the landing point in the gripper's workspace plane. Report the prediction error against time-to-catch for the ball, plate and card.
- *Accept when:* the ball error is under 1 cm at 200 ms before the catch. The card error is expected to be large; that is the motivation for M2.

**M2: Learned aero residual**
- A small MLP or GRU predicts the acceleration residual (true minus physics model), conditioned on the recent state history.
- Train it to output a mean and a log-variance (Gaussian NLL), so it knows when it's unsure.
- Compare: physics only; residual; a pure learned predictor (no physics).
- *Accept when:* the residual model beats both on the card, and its predicted variance is calibrated (NLL and coverage plot).

**M3: Interception controller**
- The planner solves for the gripper pose and timing at the predicted catch point. Track it with operational-space control, or with a QP (OSQP) under joint, velocity and torque limits.
- Close the gripper from predicted contact time, or from contact sensor events (the `contact` sensor).
- *Accept when:* the ball catch rate is above 90%; report the card catch rate.

**M4: Event-triggered replanning**
- Baseline: replan at a fixed 50 Hz.
- Variance-triggered baseline: replan when the predicted catch-point covariance grows by more than a threshold.
- Learned trigger: a small policy (PPO, CPU, `mujoco.batch`) chooses {keep plan, replan}, with reward = catch − λ·(number of replans).
- *Accept when:* you have a Pareto plot of catch rate against planner calls for all three triggers.

**Stretch (crazy)**
- Catch with a **suction cup** (`adhesion`) versus a parallel gripper.
- Two arms juggling a card between them.
- Swap the object for a paper plane, built from an IPC shell flex with plasticity so it keeps its creases. Note that speculative contacts don't cover flexes yet in 4.0.

## Evaluation
- Catch rate per object type. Prediction error against time-to-catch. NLL and coverage of the residual's uncertainty. Planner calls per catch. Wall-clock time per step.
- Ablations: speculative on/off × timestep; residual on/off; trigger type.

## Papers
- Johannink et al., *Residual RL for Robot Control* (ICRA 2019): https://arxiv.org/abs/1812.03201
- O'Connell et al., *Neural-Fly* (Science Robotics 2022), a learned aerodynamic residual inside a controller: https://arxiv.org/abs/2205.06908
- Kim, Shukla, Billard, *Catching Objects in Flight* (T-RO 2014): https://scholar.google.com/scholar?q=Catching+Objects+in+Flight+Kim+Shukla+Billard
- Forrai, Miki, Gehrig, Hutter, Scaramuzza, *Event-based Agile Object Catching with a Quadrupedal Robot* (ICRA 2023): https://scholar.google.com/scholar?q=Event-based+Agile+Object+Catching+with+a+Quadrupedal+Robot
- Trimpe & D'Andrea, *Event-Based State Estimation with Variance-Based Triggering* (IEEE TAC 2014): https://scholar.google.com/scholar?q=Event-Based+State+Estimation+with+Variance-Based+Triggering
- Kendall & Gal, *What Uncertainties Do We Need…* (NeurIPS 2017), for the NLL variance head: https://arxiv.org/abs/1703.04977
- MuJoCo docs: speculative contacts (`computation/speculative.html`), fluid forces (`computation/fluid.html`).

## Pitfalls
- Speculative contacts are **CPU only** in 4.0. Don't plan GPU RL in Playground for this project; use `mujoco.batch` multi-world CPU.
- The fluid model needs a sensible geom inertia and `fluidshape`. Check the card actually flutters before training anything.
- Keep the physics predictor genuinely physical: same drag form, fitted coefficient. Otherwise the "residual" comparison isn't meaningful.
