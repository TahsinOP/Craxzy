---
title: "Speculative Contacts"
description: "Why a baseball passes through a 1.2 cm bat at a 10 ms timestep, and what MuJoCo 4.0's speculative flag changes."
sidebar:
  order: 3
---

:::note[Guiding question]
The ball and the bat close by 10 to 20 cm per step. The bat is 1.2 cm thick. What stops the ball going through?
:::

## The problem: discrete collision detection

A rigid-body simulator checks for contact at the end of each step. A pitch arrives at 8 to 12 m/s and the bat swings into it at up to 2 m/s, so in one 10 ms step they close by

$$
(v_\text{ball} + v_\text{bat})\,h \approx 10\ \text{to}\ 14\ \text{cm},
$$

against a combined thickness of 1.2 cm of bat and 7.3 cm of ball. Whether the ball is sampled overlapping the blade, and so gets a contact, depends on where in its stride it happens to be. Often it is not, and it reappears behind the bat. This is **tunnelling**.

## What the flag does

With `<flag speculative="enable"/>`, MuJoCo 4.0 sweeps fast geoms over the coming step, finds the time of impact $t \in (0,1)$ by conservative advancement, and creates a contact *before* the geoms touch, with a synthetic distance chosen so that the solver removes exactly $(1-t)$ of the closing speed and lands the ball on the surface at the end of the step. The feature is CPU-only in 4.0, which decides the architecture: every multi-world run here uses `mujoco.batch`, never a GPU backend.

The restitution law of the bat-ball pair was chosen for this: with zero stiffness in `solref`, as the 4.0 modelling notes ask, the spring term does not cancel the elastic damper during the speculative approach step.

## The experiment

**Static bat.** The arm holds the bat at the nominal strike posture and pitches are aimed at the blade (±3 cm across it, ±12 cm along it), 200 per seed. "Bounced" means the ball comes back.

**Swinging bat.** The M1 batter replans at every step; "returned" means the ball leaves the bat forwards.

<figure class="plot">
<img class="fig-light" src="/fig/tunnel-light.svg" alt="Share of pitches that bounce off the bat, by timestep, with speculative contacts off and on">
<img class="fig-dark" src="/fig/tunnel-dark.svg" alt="Share of pitches that bounce off the bat, by timestep, with speculative contacts off and on">
<figcaption>3 seeds of 200 pitches per bar. With the flag off, the share of pitches that pass through grows with the timestep; with it on, every pitch is a clean impact at all three timesteps.</figcaption>
</figure>

| timestep | static, flag off | static, flag on | swinging, flag off | swinging, flag on |
|---|---|---|---|---|
| 8 ms | 50.3 ± 5.4 % | 100.0 % | 53.2 ± 1.8 % | 99.8 ± 0.2 % |
| 10 ms | 44.7 ± 1.0 % | 100.0 % | 43.8 ± 3.5 % | 99.8 ± 0.2 % |
| 16 ms | 25.0 ± 1.8 % | 100.0 % | 26.2 ± 1.9 % | 99.8 ± 0.2 % |

The controller, the filter and the flight model take their timestep from the model, so the same code runs unchanged at 8, 10 and 16 ms. With speculative contacts on, even 16 ms steps give 95 % of pitches in the target zone.

<figure class="sim">
<video src="/video/tunnel.mp4" width="1280" height="480" loop muted playsinline controls preload="metadata"></video>
<figcaption>The same four pitches with the flag off (left) and on (right), 10 ms steps, slowed down. On the left the bat swings through balls it was placed to hit.</figcaption>
</figure>

:::caution[A planner that was right and a ball that went through anyway]
Without the flag the batter's planning was identical: same forecast, same swing, same bat position at the impact step. The difference is entirely in the contact physics. A tunnelling simulator would have made the planner look broken.
:::

:::tip[Where in the code]
`scripts/m0_cage.py` (`static_bat`, `swinging_bat`, `video`), `models/scene.py` (the pair), `tests/test_env.py::test_speculative_contacts_stop_the_ball_tunnelling`.
:::
