---
title: "Speculative Contacts"
description: "Why a 0.7 mm card passes through a finger at an 8 ms timestep, what the speculative flag changes, and how the catch criterion had to be corrected."
sidebar:
  order: 4
---

:::note[Guiding question]
A card moves 2 cm per step and is 0.7 mm thick. What stops it from going through a finger?
:::

## The problem: discrete collision detection

A rigid-body simulator checks for contact at the end of each step. If two bodies do not overlap at step $k$ and do not overlap at step $k+1$, no contact is generated, whatever happened in between.

A card arrives at 2 to 5 m/s. In one 8 ms step it travels

$$
v\,h = 1.6\ \text{to}\ 4\ \text{cm},
$$

and a fingertip is about 2 cm thick including its rubber pad. The card can be on one side of the finger at step $k$ and on the other at step $k+1$. Nothing in between is ever tested. This is **tunnelling**, and thin fast objects are its worst case.

## What the flag does

With `<flag speculative="enable"/>`, MuJoCo 4.0 anticipates contacts. In outline, the solver is also given constraints for pairs of geoms that are still apart but would meet during the coming step at their current relative velocity; such a constraint does nothing while the gap stays open and acts as a contact when the gap would close. A body that would have skipped over an obstacle is stopped at its surface. The MuJoCo documentation is the authority on the details; what is established here is the measured effect below.

The feature is CPU-only in 4.0. That decides the architecture: every multi-world computation in the project uses `mujoco.batch`, never a GPU backend, because a card catcher without speculative contacts is not a card catcher.

## The experiment

The arm holds its ready pose with open jaws. A card is thrown face-first along the jaw axis at the outside of one fingertip, at 2 to 5 m/s, so that the only solid in its path is that finger.

**Criterion.** A throw counts as tunnelling when the card's centre of mass crosses the mid-plane of the fingertip pad, at a point at least 2 mm inside the pad laterally. The pad lies behind the whole thickness of the finger, so reaching that point from outside means the centre has passed through at least 13 mm of solid. Soft-contact penetration, which is 1 to 3 mm, cannot be mistaken for it.

<figure class="plot">
<img class="fig-light" src="/fig/tunnel-light.svg" alt="Tunnelling rate with speculative contacts off and on at 8 and 16 ms">
<img class="fig-dark" src="/fig/tunnel-dark.svg" alt="Tunnelling rate with speculative contacts off and on at 8 and 16 ms">
<figcaption>600 throws per condition (3 seeds of 200). Without the flag every card passes through the finger; with it none does, at either timestep and at every launch speed.</figcaption>
</figure>

<figure class="sim">
<video src="/video/tunnel.mp4" loop muted playsinline controls preload="metadata"></video>
<figcaption>The same throw four times: flag off and on, at 8 ms and at 16 ms. Without speculative contacts the card reappears behind the finger.</figcaption>
</figure>

:::caution[A test error that looked like a physics error]
The first version of this experiment reported 6 % tunnelling with the flag **on**. Tracing those throws showed two faults in the test: the card's lower edge was clipping the palm before it reached the finger, and the criterion had only 2 to 3 mm of margin on a tapered finger. Aiming higher on the fingertip and moving the criterion to the pad's mid-plane gave the clean 100 % against 0 %. The lesson generalises: a surprising number is first a claim about the measurement.
:::

## The second trap: catches that are not catches

The flag was also ablated in closed loop, with a perfect predictor so that only the contact physics differs. The sensor-based catch criterion gave an absurd result for the card: **more** catches with the flag off.

<figure class="plot">
<img class="fig-light" src="/fig/m3-contacts-light.svg" alt="Catch rate by timestep and speculative flag, for the sensor criterion and the clean criterion">
<img class="fig-dark" src="/fig/m3-contacts-dark.svg" alt="Catch rate by timestep and speculative flag, for the sensor criterion and the clean criterion">
<figcaption>Closed-loop catches with a perfect predictor, 3 seeds of 128 throws. For the card with the flag off, the yellow bars are cards lodged inside a finger.</figcaption>
</figure>

The explanation is in the penetration depth. Among the cards counted as caught, the deepest overlap with any other geom is

| timestep | flag | median overlap of "caught" cards |
|---|---|---|
| 8 ms | on | 0.22 mm |
| 8 ms | off | 2.56 mm |
| 16 ms | on | 0.41 mm |
| 16 ms | off | 2.92 mm |

A card is 0.7 mm thick. An overlap of 2.6 mm means a finger has passed most of the way through it: the card tunnelled part-way into a finger and stuck there, touching both fingers, near the tool centre, off the ground. Every clause of the sensor criterion is satisfied by an event that is not a catch.

The **clean** criterion adds one condition: no geometry overlaps the object by more than the object's own thickness. With it the ordering is the physical one:

| | 8 ms, on | 8 ms, off | 16 ms, on | 16 ms, off |
|---|---|---|---|---|
| card, clean [%] | 43.8 ± 0.6 | 26.8 ± 3.9 | 40.6 ± 1.1 | 12.2 ± 1.6 |
| ball, clean [%] | 99.2 ± 0.0 | 99.2 ± 0.0 | 99.2 ± 0.6 | 1.8 ± 0.4 |

For the ball the flag is irrelevant at 8 ms, where a 5 cm ball cannot skip a finger, and decisive at 16 ms, where it travels 8 cm per step and passes through the hand.

:::tip[Where in the code]
`scripts/m0_tunnel.py` (the fingertip experiment and its video), `envs/catch.py` (`final_penetration`, `outcome`), `scripts/m3_catch.py` (the closed-loop ablation).
:::
