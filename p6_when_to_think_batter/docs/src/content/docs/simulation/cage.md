---
title: "The Cage"
description: "The arm, the flat 1.2 cm bat, the frictionless impact, the delayed camera and the target zone."
sidebar:
  order: 1
---

:::note[Guiding question]
What does the robot have to work with, and what does it get to see?
:::

## The arm

A Menagerie Franka Panda (without its hand) stands on a 55 cm pedestal. Its seven position servos are replaced by **torque motors** at the datasheet limits (87 N m for the first four joints, 12 N m for the last three); the controller is model-based and commands torque. The arm's datasheet joint speed limits (2.2 to 2.6 rad/s) are what bound the whole problem: the fastest the blade can move along its face normal is about **2 m/s**, found by a random-restart search over strike postures (`scripts/design_ready.py`).

## The bat

A flat blade welded to the flange: 12 mm thick, 9 cm wide, 38 cm long, 300 g with its handle. A flat face rather than a round bat is a deliberate choice. With a flat face, the direction the ball leaves in is set by one thing, the face normal, so "choose the bat pose that sends the ball to the target" is a well-posed problem. And 12 mm in the direction of motion is exactly the thin geometry that speculative contacts exist for.

A site at the blade's centre, the *sweet spot*, is the bat's task frame: its x axis is the face normal, its z axis runs along the blade.

## The impact

The bat collides with the ball and nothing else, through one explicit contact pair. The pair is **frictionless** and **nearly elastic**. Following the MuJoCo 4.0 notes on speculative contacts and restitution, it uses a direct (negative) `solref` with zero stiffness and damping $b = (1+e)/h$:

```xml
<pair geom1="bat" geom2="ball" condim="1" solref="0 -180" solimp="0.995 0.995 0.001 0.5 2"/>
```

With zero stiffness the damper alone turns a closing normal speed $v$ into $-e\,v$ in one step, so the impact is a clean restitution law. The pair's $e$ is 0.8. The ball leaves with less, because the wrist gives under the impulse: the effective value the planner needs is 0.587, fitted in [M1](/planner/hitter-results/).

Frictionless means the impulse is along the face normal and the spin is unchanged. Real bats grip the ball; this simplification is what makes the impact invertible in closed form.

## The camera

The ball's position comes from a MuJoCo 4.0 **delayed sensor**: a `framepos` sensor with `nsample="3"` and `delay="0.03"`, so `sensordata` holds where the ball was three steps ago. Noise and dropouts are added in Python:

- noise per axis: 2 mm plus 2 mm per metre of range from the camera (behind the batter), so about 1.5 cm at the pitching machine and 3 mm at the plate;
- 5 % of frames missing;
- 100 fixes per second, one per 10 ms simulator step.

## The target

A disc of 0.75 m radius on the ground, its centre drawn per pitch between 3.5 and 6 m out and up to 20° to either side. A pitch ends a few steps after the ball leaves the bat; the landing point is then extrapolated with the exact flight model (a test checks it against continuing the simulation).

**Hit quality** is $\exp(-d^2 / 2\sigma^2)$ with $d$ the landing distance from the target centre and $\sigma$ = 1.5 m, and 0 for a miss. It is the reward the learned trigger optimises and the y axis of every trade-off plot.

<figure class="sim">
<video src="/video/worldmodel.mp4" width="960" height="540" loop muted playsinline controls preload="metadata"></video>
<figcaption>The cage from the side: pedestal, pitching machine on the right, target zone in yellow. (These swings are by the learned planner of Part V.)</figcaption>
</figure>

:::tip[Where in the code]
`models/scene.py` (the cage built with `MjSpec`), `control/arm.py` (kinematics, the strike velocity, computed torque), `envs/cage.py` (the batched environment), `assets/batting_cage.xml` (a standalone export for MuJoCo Studio).
:::
