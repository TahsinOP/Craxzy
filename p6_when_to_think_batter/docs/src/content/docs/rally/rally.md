---
title: "Two Arms, One Ball"
description: "The rally's setup: a second Panda that is the first one rotated, thin paddles, a midi table, and a table bounce that had to be made exact."
sidebar:
  order: 1
---

:::note[Guiding question]
How much of the batter carries over to a two-arm ping-pong rally, and what has to be new?
:::

<figure class="sim">
<video src="/video/rally.mp4" width="1280" height="720" loop muted playsinline controls preload="metadata"></video>
<figcaption>Two rallies, rendered with Filament. Green: the filter's forecast of the ball, through the bounce. Yellow markers: where the ball was when a planner ran.</figcaption>
</figure>

## Two arms, one planner

Arm B is a second copy of the Panda, attached with the name prefix `b_` at the far end of the table and rotated half a turn about the vertical. Seen from its own frame, $(x, y, z) \mapsto (L - x, -y, z)$, arm B *is* arm A. So everything that plans a stroke works in the hitting arm's own frame, and the batter's flight model, EKF, aiming, impact inversion, inverse kinematics and swing QP are reused unchanged. A test checks that B's paddle, in B's frame, is where A's kinematics put it, to $10^{-9}$ m.

The rally planner adds what ping-pong needs on top of the batter's planner:

- **Candidates:** strike plane (2) × time to the bounce on the far half (3) × landing point (2 depths × 3 across): 36.
- **Checks:** every candidate's return is rolled out through its bounce, all at once. It must clear the net and cross the opponent's strike plane within reach. The score prefers a comfortable height for the opponent.
- **Strike:** the four best go through impact inversion, IK and the swing QP, and the cheapest feasible swing wins.

## The paddle and the ball

A round blade 75 mm in radius and **6 mm thick** on a 0.30 m handle, and a 40 mm, 2.7 g ball. At 2 to 5 m/s the ball moves 2 to 5 cm per 10 ms step against 6 mm of paddle, so speculative contacts are on. The paddle contact is frictionless with restitution 0.85, as the bat's was, so its impact inverts in closed form. Its effective restitution, fitted on true-state hits, is 0.83.

## Why a midi table

<figure class="plot">
<img class="fig-light" src="/fig/rally-feasibility-light.svg" alt="Paddle speed a return needs on an ITTF table and on a midi table, against the arm's limit">
<img class="fig-dark" src="/fig/rally-feasibility-dark.svg" alt="Paddle speed a return needs on an ITTF table and on a midi table, against the arm's limit">
<figcaption>Paddle speed needed for the first return of 40 serves, planned with the true ball state.</figcaption>
</figure>

A ping-pong ball loses speed fast: about 18 m/s² of drag at 8 m/s on 2.7 g. It reaches the paddle at about 2 m/s and has to be driven back. On an ITTF table (2.74 m) a return must leave at about 7 m/s, which needs **3.0 m/s** of paddle speed. The Panda's joint speed limits give **1.6 m/s** at the best posture. On a midi table (1.80 × 1.00 m, a size sold for home play) a return needs 4.2 m/s and 1.5 m/s of paddle: just inside, with a long handle for leverage.

## Making the bounce exact

The batter's flight model was exact because the simulator's step had a closed form. A table bounce needed the same, so 2,000 bounces were simulated and measured.

| table-ball contact | $v_z'/|v_z|$, median | std | max | leaves faster than it arrived |
|---|---|---|---|---|
| frictionless | 0.850 | 0.000 | 0.85 | 0 % |
| friction 0.25 | 1.108 | 0.380 | 3.28 | 67 % |

**Frictionless,** a bounce takes exactly one step: the step whose ballistic prediction would end below the surface. In that step the vertical velocity is reversed and scaled by $e = 0.850$, and everything else follows the ordinary flight step. The table-aware flight model is therefore exact through the bounce: 0.05 mm over a flight, in `tests/test_rally.py`.

**With friction,** Coulomb's law held to the digit, but the ball left the table faster than it arrived in two bounces out of three, up to 3.3 times. With this one-step, zero-stiffness restitution law, the frictional contact adds energy. A table that adds energy is unusable, so the table is frictionless, like the paddle.

:::caution[The first bounce study was aimed at the net]
The first batch of simulated bounces gave a scattered, nonsensical restitution. Every test ball had been aimed at x = 2.12 m, which is where the net was. A surprising number is first a claim about the measurement.
:::

:::tip[Where in the code]
`rally/scene.py`, `rally/flight.py` (`BounceModel.identify`, `TableFlight`), `rally/planner.py`, `scripts/rally.py` (`feasibility`, `bounce_study`).
:::
