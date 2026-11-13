# P3: Chicken-Head Duck

**Gaze-stabilised walking for onboard vision on Open Duck Mini.**

## Goal
Birds keep their heads almost perfectly still in space while their bodies bob and sway, which is why chickens and ducks can see clearly while moving. Give **Open Duck Mini v2** the same ability: the legs walk (RL), and the **neck and head joints** keep a **head-mounted camera** stabilised, so that **visual odometry (VO) running on the head camera drifts less**. The application is a small walking robot that can map or track while it walks.

## Why it's a hybrid research question
- **Model-based part:** a vestibulo-ocular-style controller. The IMU on the body plus neck forward kinematics give the head pose, and inverse kinematics with feed-forward counter-rotation keeps the camera still. It's exact for rigid kinematics but lags under delays and limited bandwidth.
- **Learned part:** (a) the gait policy; (b) a learned *predictive* residual that anticipates the head disturbance from the gait phase, beating the feedback lag; (c) optionally, the gait itself learns to be "camera-friendly".
- **Payoff measured in your own field:** VO/SLAM drift and feature-tracking quality. It connects locomotion, estimation and biomimetics (your soft robotics and biomimetics interest) in one project.

## MuJoCo 4.0 + Playground 1.0 features used
- Playground 1.0 on the **MuJoCo Warp** backend for gait RL (GPU).
- The **`orientation` actuator** (SO(3) target, new in 4.0) as one option for driving the head orientation directly; compare it with joint-space PD on the real neck joints.
- **Sensor and actuator delays** (IMU and servo latency, the core difficulty of gaze stabilisation).
- `dcmotor` actuators for realistic servo bandwidth.
- **Offscreen camera rendering**: the classic renderer, Filament offscreen, or Playground's MJWarp batch renderer for vision in the loop. Produce an RGB stream from the head camera for VO.

## Milestones

**M0: Duck + head camera**
- Get the Open Duck Mini v2 MJCF (search the `apirrone/Open_Duck_Mini` repo; confirm the neck and head DoFs). Add a camera to the head and an IMU site to the body.
- A textured arena with posters and objects, so VO has features.
- *Accept when:* you can record a head-camera video while the joints are scripted.

**M1: Gait policy**
- Joystick walking in Playground, with domain randomisation, latency and `dcmotor` actuators. The head is held at a fixed neck pose.
- *Accept when:* stable walking at 2–3 speeds; you've measured head angular and linear velocity (the "bobbing" baseline).

**M2: Model-based gaze stabilisation**
- A controller that keeps the head orientation fixed in the world (and optionally the head position along the vertical axis) using body IMU + neck forward kinematics → inverse kinematics, with and without a delay compensator.
- *Accept when:* head angular-velocity RMS is reduced compared with M1 at every speed and delay.

**M3: Learned predictive residual**
- A small network uses gait phase, recent IMU and joint history to predict the head disturbance about 30–50 ms ahead. Its output is added as feed-forward on top of M2.
- Compare: M2 only; residual; and a pure learned head policy trained end-to-end with a head-stillness reward.
- *Accept when:* stabilisation error against delay is plotted for all three; the residual should win at larger delays.

**M4: Payoff, visual odometry**
- Run monocular or RGB-D VO (OpenCV ORB + PnP, or your own Open3D pipeline) on head-camera frames for M1 (no stabilisation), M2 and M3.
- *Accept when:* VO drift (ATE/RTE against simulator ground truth), feature tracks per frame and tracking-loss events are reported for each.

**Stretch (crazy)**
- **Bird-style head-bobbing:** the head *holds* still, then *jumps* forward. Let RL discover the hold-and-jump pattern by rewarding VO quality and penalising only rotation.
- **Event-camera emulator** on the head: generate log-brightness-change events from rendered frames, and show that the event count drops sharply with stabilisation.
- **"Look at" behaviour:** the duck walks and keeps looking at a moving ball (gaze tracking plus stabilisation).

## Evaluation
- Head angular velocity and image-motion RMS, VO ATE/RTE, feature-track length, tracking losses, walking performance (no regressions), sensitivity to delay and servo bandwidth.

## Papers
- Pete et al., *The role of passive avian head stabilization in flapping flight*, and related avian head-stabilisation literature: https://scholar.google.com/scholar?q=avian+head+stabilization+robot
- Necker, *Head-bobbing of walking birds* (Journal of Comparative Physiology A, 2007): https://scholar.google.com/scholar?q=Necker+head-bobbing+of+walking+birds
- Grandia et al. (Disney Research), *Design and Control of a Bipedal Robotic Character* (RSS 2024), the BDX lineage of Open Duck: https://scholar.google.com/scholar?q=Design+and+Control+of+a+Bipedal+Robotic+Character+Grandia
- Shibata & Schaal, *Biomimetic gaze stabilization based on feedback-error-learning with nonparametric regression networks* (Neural Networks 2001), learned VOR/OKR: https://scholar.google.com/scholar?q=Biomimetic+gaze+stabilization+based+on+feedback-error-learning
- Zakka et al., *MuJoCo Playground* (2025): https://arxiv.org/abs/2502.08844
- Campos et al., *ORB-SLAM3* (T-RO 2021), for the VO evaluation: https://arxiv.org/abs/2007.11898
- Open Duck Mini: https://github.com/apirrone/Open_Duck_Mini *(verify the repo and MJCF location)*

## Pitfalls
- Check the duck's neck and head DoFs and their joint limits; stabilisation range is limited by them.
- Rendering in the loop is slow. Train the gait without vision, and render only for evaluation in M4.
- Make delays realistic (10–30 ms). With zero delay, pure feedback trivially wins and the learned prediction has nothing to add.
