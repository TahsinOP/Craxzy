---
title: "The Pitching Machine"
description: "Random speed, height, aim and spin, with the machine zeroed in so that spin changes the path but not the arrival point."
sidebar:
  order: 2
---

:::note[Guiding question]
How can two pitches that arrive at the same point be different problems?
:::

## A pitch

The machine stands 6.5 m from the robot. A pitch is drawn as

- a release point: 1.00 to 1.45 m high, ±10 cm to the side;
- a release speed: 9 to 13 m/s;
- a spin: 0 to 150 rad/s, backspin or topspin, with the axis tilted up to 30° from horizontal (so some sidespin too);
- an aim point on the hitting plane, 0.62 m in front of the robot: ±15 cm sideways, 0.78 to 1.10 m high.

The machine is **zeroed in**: the release direction is solved, by a fixed-point iteration on the exact flight model, so that the ball *with its drag and its spin* crosses the hitting plane at the drawn aim point (to 1.5 mm). The spin therefore does not move where the pitch arrives. It changes how it gets there.

<figure class="plot">
<img class="fig-light" src="/fig/pitches-light.svg" alt="Side view of pitches coloured by spin, and the height change at the plate caused by spin">
<img class="fig-dark" src="/fig/pitches-dark.svg" alt="Side view of pitches coloured by spin, and the height change at the plate caused by spin">
<figcaption>Left: pitches with backspin are released lower and float; pitches with topspin are released higher and dive; all reach the same window. Right: the same release without its spin would cross the plate up to 23 cm higher or lower, against a 1.2 cm bat.</figcaption>
</figure>

## Why this makes the problem hard

Spin acts through lift, $c\,\omega \times v$, a force that bends the path gradually. In the first 100 ms of flight a floating and a diving pitch differ by a few millimetres, below the camera noise at that range. The filter starts with a prior over spin and narrows it as the curvature shows; until then, the forecast of the crossing point is uncertain by tens of centimetres.

| | range over 1,000 pitches |
|---|---|
| time of flight to the plate | 480 to 761 ms |
| speed at the plate | 8.2 to 12.1 m/s |
| descent angle at the plate | 11° to 31° |
| what the spin does at the plate | −23 to +22 cm in height (std 11 cm), −11 to +9 cm sideways |

## Sizing to the arm

The pitches are slow by baseball standards, and the machine is close. That is not an accident: the Panda can move the bat at about 2 m/s, and the ball has to carry several metres to the target off a bat that slow. Below about 8.5 m/s a pitch cannot reach the plate from 6.5 m at all with this drag. The scenario was sized to what the arm can do, not the other way round.

:::tip[Where in the code]
`models/pitch.py` (`generate`, `solve_release`), `configs/default.yaml` (`pitch`, `target`), `scripts/m0_cage.py` (the distribution plots).
:::
