# P4: Learn to Fall

**A humanoid that knows when balance is lost, and falls safely.**

## Goal
Real humanoids fall, and falls break hardware. Build a **Unitree G1** (or Booster T1) controller that:
1. walks and balances normally (RL policy);
2. **predicts, with a model, when a fall has become inevitable** (capturability / capture point);
3. **switches** to a learned **safe-fall policy** that minimises damage: impact forces, head contact, and **permanent deformation of fragile parts**, measured with MuJoCo 4.0 plasticity;
4. optionally **gets back up**.

The research core is the **switch**: when should the system stop trusting the balance controller and commit to falling well?

## Why it's a hybrid research question
- **Model:** linear inverted pendulum (LIPM) capture point, `ξ = x + ẋ/ω` with `ω = √(g/z)`. If ξ leaves the reachable support region, no stepping or balance strategy can recover, so the fall is inevitable. It's cheap and interpretable, but approximate for a full humanoid.
- **Learned:** (a) the safe-fall policy (RL); (b) a learned fall predictor (classifier on state history) as the alternative to the model-based switch; (c) a hybrid switch that uses the capture point *plus* a learned correction.
- **The question:** model-based vs learned vs hybrid fall detection. It's the "knowing when to trust each" theme, with damage as the cost of getting it wrong.

## MuJoCo 4.0 + Playground 1.0 features used
- Playground 1.0 G1/T1 environments on **MuJoCo Warp** for balance and fall-policy RL (GPU).
- **Plasticity as a damage metric:** add fragile parts that deform permanently, e.g. a wrist-mounted camera on a hinge with **`springyield`**, or thin protective shells (`<elasticity yield>`). Read the damage from `mjData.plastic` after each episode. *(Check which plastic features run on Warp; otherwise score damage with CPU MuJoCo replays.)*
- `contact` and `tactile` sensors to measure impact locations and forces.
- Sensor delays (IMU latency affects how early a fall can be detected).

## Milestones

**M0: Push bench**
- G1 standing and walking (Playground baseline), plus a push generator: random impulse magnitude, direction and timing via `xfrc_applied`.
- *Accept when:* a curve of fall probability against push impulse; logged impact force and head-contact events.

**M1: Capture-point fall predictor**
- Compute the capture point from CoM state (the simulator's subtree CoM) and the support polygon / step-reachable region.
- Measure **lead time** (how early it predicts the fall) and false-alarm rate against the simulator's ground truth.
- *Accept when:* ROC curves and a lead-time histogram.

**M2: Learned and hybrid predictors**
- (a) A classifier on proprioceptive history (labels from M0 rollouts); (b) hybrid: capture-point features plus learned correction.
- *Accept when:* ROC and lead time for model-based, learned and hybrid predictors; the hybrid should improve lead time without more false alarms.

**M3: Safe-fall policy**
- RL policy triggered by the predictor. Reward: −peak impact force, −head contact, −plastic damage (`mjData.plastic` norm on fragile parts), −joint-limit violations.
- Baselines: freeze (go limp), a scripted "squat and roll", and the learned policy.
- *Accept when:* damage metrics for each baseline across push magnitudes.

**M4: The switch, end to end**
- Full system: balance policy → predictor → safe-fall policy.
- Compare switching rules: always balance; capture-point switch; learned switch; hybrid switch.
- *Accept when:* total expected damage and the fraction of unnecessary falls (switching when the robot could have recovered) for each rule.

**Stretch (crazy)**
- **Get-up policy** after the fall (recovery to standing).
- **Fall onto soft terrain:** IPC cloth "mats" or plastic clay floors, and learn to aim for them.
- **Carry a fragile object** (an egg-like plastic shell) and protect *it*, not the robot.

## Evaluation
- Fall-prediction ROC and lead time, false switches, peak impact forces, head contacts, plastic damage, get-up success, and comparison against the "always balance" baseline.

## Papers
- Pratt et al., *Capture Point: A Step toward Humanoid Push Recovery* (Humanoids 2006): https://scholar.google.com/scholar?q=Capture+Point+A+Step+toward+Humanoid+Push+Recovery
- Koolen et al., *Capturability-based analysis and control of legged locomotion* (IJRR 2012): https://scholar.google.com/scholar?q=Capturability-based+analysis+and+control+of+legged+locomotion
- Ha & Liu, *Multiple contact planning for minimizing damage of humanoid falls* (IROS 2015): https://scholar.google.com/scholar?q=Multiple+contact+planning+for+minimizing+damage+of+humanoid+falls
- Kumar, Ha, Liu, *Learning a Unified Control Policy for Safe Falling* (IROS 2017): https://scholar.google.com/scholar?q=Learning+a+Unified+Control+Policy+for+Safe+Falling
- Radosavovic et al., *Real-World Humanoid Locomotion with RL* (Science Robotics 2024): https://arxiv.org/abs/2303.03381
- He et al., *Learning Getting-Up Policies for Real-World Humanoid Robots* (2025): https://scholar.google.com/scholar?q=Learning+Getting-Up+Policies+for+Real-World+Humanoid+Robots
- Zakka et al., *MuJoCo Playground*: https://arxiv.org/abs/2502.08844

## Pitfalls
- Plasticity support on MJX/Warp may lag behind CPU. If so, train with impact-force rewards on GPU and score plastic damage in CPU replays.
- Ground-truth "fall inevitable" labels need a definition; e.g. no policy rollout among N tries recovers from that state.
- Keep the walking policy fixed while comparing switches, so the comparison stays clean.
