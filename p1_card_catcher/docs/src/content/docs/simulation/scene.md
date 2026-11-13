---
title: "The Scene"
description: "The MuJoCo model: a Panda with torque motors hanging from a gantry, a table, one thrown object, and the sensors that decide whether a catch happened."
sidebar:
  order: 1
---

:::note[Guiding question]
What is in the simulated world, and why does the robot hang upside down?
:::

## What is in the world

The model is assembled in Python with `MjSpec` from the Menagerie Franka Panda (`models/scene.py`):

| element | detail |
|---|---|
| arm | 7 joints, the Menagerie position servos replaced by torque motors limited to the datasheet torques (87 N·m on joints 1–4, 12 N·m on joints 5–7) |
| hand | the Panda parallel gripper, 8 cm opening; optionally a suction disc between the fingers ([Catching](/control/catching/)) |
| tool centre | a site 103.4 mm in front of the hand flange, at the centre of the fingertip pads |
| mount | the base is bolted, inverted, to a gantry 1.30 m above the floor |
| surroundings | a floor at $z = 0$ and a low table with its top at $z = 0.25$ m |
| object | one free body: ball, plate or card |
| sensors | object position and velocity; five `contact` sensors (left finger, right finger, hand, floor, table) |

The simulation options are the ones the whole project is built around:

```xml
<option timestep="0.008" integrator="implicitfast" density="1.2" viscosity="1.8e-5">
  <flag speculative="enable"/>
</option>
```

A step of 8 ms is coarse on purpose. It is what makes a few hundred worlds affordable on a laptop, and it is what makes speculative contacts and an exact treatment of the integrator necessary rather than decorative.

## Torque motors, and the limits the planner respects

Position servos would hide the dynamics inside the simulator. With torque motors the controller has to compute the torque itself, which is what [Computed torque](/control/computed-torque/) does, and torque saturation becomes something that can be counted. The planner works with the datasheet limits in joint space:

| joint | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| velocity limit [rad/s] | 2.175 | 2.175 | 2.175 | 2.175 | 2.61 | 2.61 | 2.61 |
| acceleration limit [rad/s²] | 15 | 7.5 | 10 | 12.5 | 15 | 20 | 20 |
| torque limit [N·m] | 87 | 87 | 87 | 87 | 12 | 12 | 12 |

The planner uses 95 % of the velocity limits and 90 % of the acceleration limits, which leaves the tracking controller some room to correct.

## Why the arm hangs from a gantry

Objects arrive from above, falling at 45° to 85° below the horizontal. To catch one, the palm must face up and forward, towards the incoming object.

An upright Panda can barely do that. The pitch of its tool axis in the base frame is

$$
\theta_{\text{tool}} = q_6 + q_4 - q_2 - \tfrac{\pi}{2},\qquad q_6 \le 3.75\ \text{rad},
$$

and with $q_4$ negative by construction of the elbow, tilting the palm upward drives joint 6 into its limit. This is a property of the kinematics, not of a particular solver. The script `scripts/m0_mount.py` quantifies it: for a grid of tool-centre positions over the catch window and three palm elevations, it runs inverse kinematics from several seeds and counts the positions with a solution that keeps at least 0.3 rad of margin to every joint limit.

<figure class="plot">
<img class="fig-light" src="/fig/mount-light.svg" alt="Share of window positions with a usable palm-up pose, upright against inverted mount">
<img class="fig-dark" src="/fig/mount-dark.svg" alt="Share of window positions with a usable palm-up pose, upright against inverted mount">
<figcaption>Share of catch-window positions for which the hand can face the incoming object with at least 0.3 rad of joint-limit margin. Upright, the steepest arrivals have no usable pose at all.</figcaption>
</figure>

Hanging the same arm upside down turns the problem around: "palm up" becomes the natural direction of the wrist. Nothing else about the robot is changed.

## The ready pose

Between throws the arm waits with its tool centre at the centre of the window, $c_0$, and its palm pitched towards the typical arrival direction of the object being thrown: 50° above the horizontal for the ball, 60° for the plate, 65° for the card. The jaw axis lies along $e_w$, so one finger is above the other and an object that arrives a little early or late still passes between them.

The pose is found by the same inverse kinematics that is used online, and the program refuses to start if it cannot be reached to 1 mm and 0.01 rad.

## The catch criterion

A catch is judged half a second after the object reached the plane, from sensors only:

- **caught**: both finger contact sensors report the object, its centre is within 8 cm of the tool centre, and it touches neither floor nor table and is above the table;
- **retained**: the same, with any contact with the hand in place of the two-finger pinch (this is the criterion for the suction disc, which has no pinch);
- **clean**: caught, and no geometry overlaps the object by more than the object's own thickness.

The third line exists because of a mistake that the contact ablation exposed; see [Speculative contacts](/simulation/speculative-contacts/).

:::tip[Where in the code]
`models/scene.py` (`build_catch_spec`, `CatchIndex`), `envs/catch.py` (`outcome`), `scripts/m0_mount.py`. `assets/catch_card.xml` is an exported copy of the scene for MuJoCo Studio.
:::
