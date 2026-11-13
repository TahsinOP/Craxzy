---
title: "Keeping a Rally Going"
description: "How long the rallies last, what ends them, and the when-to-think rule that doubled their length."
sidebar:
  order: 2
---

:::note[Guiding question]
With the batter's stack in both arms, how long does a rally last, and what does it take to make it last?
:::

## The rules

The feeder serves to arm A. A **return** counts when the receiver hits a ball that bounced exactly once on its half. A rally ends in the net, on the hitter's own half, after a second bounce (a miss), off the table, or at 12 returns. Each condition: 3 seeds of 48 rallies.

<figure class="plot">
<img class="fig-light" src="/fig/rally-light.svg" alt="Mean returns per rally for six conditions, and the share of rallies still going after each return">
<img class="fig-dark" src="/fig/rally-dark.svg" alt="Mean returns per rally for six conditions, and the share of rallies still going after each return">
<figcaption>Left: returns per rally, mean ± std over 3 seeds. Right: how many rallies are still going after n returns.</figcaption>
</figure>

| condition | returns per rally | reach 12 | planner calls per return | how rallies end (main) |
|---|---|---|---|---|
| true ball state, replanning every step | 11.53 ± 0.09 | 88 % | 78 | cap 88 %, missed 5 % |
| EKF, replanning every step (always think) | 4.53 ± 0.30 | 4 % | 86 | net 52 %, own half 38 % |
| **EKF, confidence-gated, every step** | **9.97 ± 0.15** | **63 %** | 60 | cap 63 %, net 12 %, own half 10 % |
| EKF, gated, 25 Hz | 9.24 ± 0.25 | 49 % | 18 | cap 49 %, net 22 % |
| EKF, gated, 12.5 Hz | 9.22 ± 0.56 | 51 % | 10 | cap 51 %, net 20 % |
| EKF, gated, speculative contacts off | 0.56 ± 0.09 | 0 % | 145 | missed 44 %, own half 27 %, off the table 26 % |

## Reading it

- **Speculative contacts are not optional here.** Without them the 6 mm paddle swings through the ball, and the rally rarely gets past the serve.
- **Thinking all the time halves the rally.** With the camera and the filter, replanning at every step gives 4.5 returns. The same stack that declines to replan while the filter is unsure gives 10.0, two thirds of the way from 4.5 to the true-state ceiling of 11.5.
- **Thinking less often costs little.** At 12.5 Hz the gated batter makes 10 planner calls per return instead of 60, for 9.2 returns instead of 10.0.

## What it took

Each of these came out of measuring why rallies ended, not out of the design:

1. **Commit before the impact.** The two strike planes are 12 cm apart, which at ping-pong speeds is 60 ms. A replan in the last 60 ms that switched planes pulled the paddle off a stroke that was about to land. With the true ball state, the first rallies ended after 1.3 returns; with the commit rule, at the cap. Within 60 ms of the impact only the same stroke may now be corrected.
2. **Tell the filter that the bounce's timing is uncertain.** A few millimetres of height error decide which 10 ms step the bounce falls in, and in the wrong step the vertical speed is off by $(1+e)|v_z|$. Before the fix the filter's vertical-speed error near the bounce was 0.4 m/s while it believed 0.03. Process noise around the bounce lets the next frames correct it.
3. **Do not think while the filter does not know.** Even so, for a few steps after every bounce the forecast swings by centimetres. Replanning on it made the arm drop a reachable stroke for one it could no longer reach: in the failing returns, the final swing was 0.8 m/s short of its target speed. Gating replanning on the filter's own vertical-speed uncertainty (0.08 m/s) fixed most of it.
4. **Calibrate the impact.** The paddle's effective restitution, fitted on 136 true-state hits, is 0.83 against the pair's 0.85.

This is the project's question in a new setting. In the batter, the learned trigger found that thinking is worth most late in the flight. In the rally, the most useful rule is *not* to think while the estimate is in transition. Both are about the value of a planner call, not its cost.

:::caution[Tried, and not kept]
Penalising the face's rotation through the stroke made true-state rallies shorter (8.9 against 12.0 returns on 8 test rallies). Weighting the swing QP against early accelerations, so the arm commits later, and making the planner stick harder to its first candidate, did not help either. All three are still in the code, off.
:::

## Limits

- A midi table, a long paddle handle and a frictionless table (see [Two arms, one ball](/rally/rally/)).
- Both arms run the same cooperative controller, aiming at the other's comfortable height; nothing is adversarial.
- The filter still costs about 1.6 returns against the true ball state. The rallies that end early mostly end with a return that is slightly slow or low.
- There is no learned trigger for the rally. The confidence gate is a hand-made rule, tuned on 48 rallies per setting.

:::tip[Where in the code]
`rally/env.py` (the referee, the commit rule, the confidence gate), `rally/flight.py` (`extra_process_noise`), `scripts/rally.py`; every number in `results/RALLY.md`.
:::
