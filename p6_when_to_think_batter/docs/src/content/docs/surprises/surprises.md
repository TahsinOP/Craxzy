---
title: "Mid-Flight Surprises"
description: "A spin change or a gust in mid-flight: which triggers notice, how fast, and what it buys."
sidebar:
  order: 1
---

:::note[Guiding question]
When the ball stops doing what the model says, which triggers notice?
:::

## The surprises

Between 25 % and 60 % of the way to the plate, either

- **the spin changes:** it jumps to 100 to 220 rad/s in the opposite sense, so a floating ball starts to dive or the reverse (written into the ball's angular velocity); or
- **a gust blows:** 0.5 to 1.2 N across the flight path for 100 ms through `xfrc_applied`, 3 to 8 m/s² on a 145 g ball.

Every controller sees the same surprised pitches. Each pitch is also run **without** its surprise (same release, same camera-noise seed), which gives a placebo: how often a trigger would have replanned in that window anyway. Without it, a trigger that simply calls the planner often would look as if it "reacted".

## Reaction at a matched budget

Each trigger at the setting that costs about 4 planner calls on ordinary pitches (from Part VI). Gusts:

| trigger | calls: plain → gust | replanned within 150 ms (placebo) | hit quality: plain → gust | hit rate: plain → gust |
|---|---|---|---|---|
| fixed rate | 4.4 → 4.4 | 100 % (100 %) | 0.91 → 0.47 | 98 % → 79 % |
| variance | 3.0 → 3.0 | 11 % (11 %) | 0.28 → 0.16 | 49 % → 29 % |
| drift | 3.4 → 3.9 | 24 % (12 %) | 0.46 → 0.26 | 68 % → 53 % |
| innovation | 4.1 → 5.9 | 35 % (29 %) | 0.46 → 0.31 | 58 % → 49 % |
| learned, never saw a surprise | 4.1 → 5.8 | 93 % (78 %) | 0.95 → 0.65 | 99 % → 93 % |
| learned, trained with surprises | 4.2 → 5.9 | 92 % (77 %) | 0.94 → 0.64 | 100 % → 92 % |

<figure class="plot">
<img class="fig-light" src="/fig/reaction-light.svg" alt="Latency from the gust to the next planner call, and hit quality with and without the gust, per trigger">
<img class="fig-dark" src="/fig/reaction-dark.svg" alt="Latency from the gust to the next planner call, and hit quality with and without the gust, per trigger">
<figcaption>Left: share of pitches with a planner call by a given time after the gust began (solid) and on the same pitches without the gust (dotted). Right: hit quality without and with the gust. 3 seeds of 200 pitches.</figcaption>
</figure>

## Reading it

- **Fixed rate and variance are blind by construction.** One follows the clock; the other follows the filter's covariance, which does not depend on what the camera saw. They spend exactly the same calls with and without the gust.
- **Innovation, drift and the learned triggers notice.** They spend 0.5 to 1.8 extra calls; the learned triggers' share of pitches with a replan in the 150 ms after the onset rises from 77 % to 92 %.
- **The policy that never saw a surprise reacts as much as the one trained with them.** It reads the innovation and the drift, and a gust moves both.
- **A spin change is too gradual to see in time.** No trigger replans sooner because of it; the innovation trigger replans within 150 ms in 28 % of pitches against 29 % without the change.
- **Reacting does not buy the quality back.** Even replanning at every step reaches only 0.78 on gust pitches. With the 5.9 calls it uses on gust pitches, the learned trigger reaches 0.64, against 0.58 for a fixed rate at 6.5 calls: a modest gain, and the hit rate is where it shows (92 % against 79 % at the matched setting).

<figure class="plot">
<img class="fig-light" src="/fig/surprise-pareto-light.svg" alt="Hit quality against planner calls on pitches with a spin change and with a gust">
<img class="fig-dark" src="/fig/surprise-pareto-dark.svg" alt="Hit quality against planner calls on pitches with a spin change and with a gust">
<figcaption>The Pareto curves of Part VI, on surprised pitches. The ordering is unchanged; every curve is lower.</figcaption>
</figure>

<figure class="sim">
<video src="/video/surprise.mp4" width="1920" height="420" loop muted playsinline controls preload="metadata"></video>
<figcaption>Four pitches with a gust: a fixed low rate (left), the innovation trigger (middle) and the learned trigger (right). Yellow markers: planner calls. Orange: the forecast the cached swing was built on; green: the current forecast. Where they part, the cached swing is aiming at a ball that is no longer coming.</figcaption>
</figure>

:::tip[Where in the code]
`models/pitch.py` (surprises are stored with the pitch), `envs/cage.py` (`step` applies them), `scripts/m4_surprises.py`; every table in `results/M4.md`.
:::
