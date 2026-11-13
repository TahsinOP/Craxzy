---
title: "Results at a Glance"
description: "Every milestone, its acceptance criterion, and the measured result."
sidebar:
  order: 2
---

| milestone | acceptance | result |
|---|---|---|
| M0 batting cage | video of the bat passing through the ball with speculative contacts off and hitting it with them on; pitch distribution | met: with the flag off the ball passes through a bat held still in 50 to 75 % of pitches (8, 10, 16 ms timesteps); with it on, it bounces off in 100 % |
| M1 model-based hitter | hit rate and target accuracy against pitch speed | met: 100.0 ± 0.0 % hit, 97.8 ± 0.2 % in the 0.75 m target zone with replanning every step; planned once, 11.8 ± 1.8 % hit |
| M2 learned world-model planner | compared with the analytic planner on hit rate and planning time | done: 96.9 ± 4.4 % hit but 54.9 ± 9.1 % on target, against 96.9 % for the analytic planner in the same run; 8.7 ms per call against 9.3 ms |
| M3 compute triggers | Pareto curves of hit quality against planner calls | met: every step costs 48 calls for quality 0.98; the learned trigger holds 0.94 with 4.0 calls (fixed rate: 0.91 with 4.4) and 0.81 with 2.9 (fixed rate: 0.58 with 2.9) |
| M4 surprises | response latency to surprises for each trigger type | met: a gust adds 1.8 planner calls for the learned and innovation triggers and none for fixed-rate and variance triggers; hit rate under gusts at ~4 calls: learned 92 %, fixed rate 79 % |
| Stretch: ping-pong rally | two-arm rally, thin paddles, speculative contacts | done: 10.0 ± 0.2 returns per rally with the filter (63 % reach the cap of 12), 11.5 with the true ball state; replanning at every step without a confidence gate gives 4.5; speculative contacts off, 0.6 |
| Stretch: Filament videos | nice-looking videos | done: every video on this site is rendered offscreen with MuJoCo 4.0's Filament renderer |
| Stretch: event camera | trigger on events instead of state innovation | done: within 150 ms of a gust the event trigger replans in 48 % of pitches (30 % without the gust), the frame-based innovation trigger in 31 % (21 %) |

All rates are means ± standard deviations over 3 seeds. Hit quality is $\exp(-d^2/2\sigma^2)$ for a hit that lands $d$ from the target centre ($\sigma$ = 1.5 m) and 0 for a miss; it is 1 for a perfect hit and about 0.88 for a hit landing at the edge of the 0.75 m target zone.

:::tip[The per-milestone reports]
Each milestone also has a report generated from its result files, with every table: `results/M0.md` to `results/M5.md` in the repository.
:::
