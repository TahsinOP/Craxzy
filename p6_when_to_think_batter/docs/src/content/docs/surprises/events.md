---
title: "An Event Camera"
description: "An emulated event camera from offscreen renders, and a trigger that fires on events instead of on state innovation."
sidebar:
  order: 2
---

:::note[Guiding question]
The frame camera reports where the ball was 30 ms ago. An event camera reports change as it happens. Does triggering on events help?
:::

## The emulator

An event camera reports, per pixel and asynchronously, when the log-brightness has changed by more than a contrast threshold since that pixel's last event. The emulator renders the scene offscreen at every simulator step (480 × 240 pixels, no shadows), keeps a reference log-intensity per pixel, and emits an ON or OFF event wherever the current log-intensity differs from it by more than 0.25, resetting the reference there. It is the usual frame-based emulation, so events are timed to the 10 ms step, not to microseconds. The camera looks down on the flight corridor from the side; the columns that show the robot are masked; the ball is about 6 pixels across.

<figure class="plot">
<img src="/fig/event-frame.png" alt="Events of one pitch accumulated over the camera's first frame" width="480" height="240">
<figcaption>The ON events of one pitch (red) over the camera's first frame. Every pixel the ball enters fires an ON event, and fires an OFF event one step later when the ball leaves, so the OFF events fall on the previous step's ON events and are not drawn. Left of the white line is masked.</figcaption>
</figure>

## The event trigger

The ON events of a step mark where the ball has just arrived. Their centroid is an image measurement of the ball **now**, accurate to about 0.3 pixel. It is compared with the image position the cached plan's forecast predicts for this instant:

$$
\chi^2 = \nu^\top \big(J P J^\top + \sigma_\text{px}^2 I\big)^{-1} \nu, \qquad \nu = u_\text{events} - \pi\big(\hat p(t \mid t_p)\big),
$$

with $\pi$ the camera projection and $J$ its Jacobian. The trigger fires when the mean over four steps passes a gate: the innovation trigger of Part VI, in pixels, without the 30 ms of latency.

Only the decision *when* to plan comes from events. The planner still runs on the EKF, which is fed by the delayed frame camera.

## Results

<figure class="plot">
<img class="fig-light" src="/fig/events-light.svg" alt="Quality against planner calls for the event trigger, the innovation trigger and a fixed rate, and latency after a gust">
<img class="fig-dark" src="/fig/events-dark.svg" alt="Quality against planner calls for the event trigger, the innovation trigger and a fixed rate, and latency after a gust">
<figcaption>Hit quality against planner calls. 3 seeds of 64 pitches per point.</figcaption>
</figure>

<figure class="plot">
<img class="fig-light" src="/fig/events-latency-light.svg" alt="Share of pitches with a planner call by a given time after a gust, per trigger">
<img class="fig-dark" src="/fig/events-latency-dark.svg" alt="Share of pitches with a planner call by a given time after a gust, per trigger">
<figcaption>Solid: after a gust; dotted: the same pitches without it. Each trigger at the setting that costs about 4 calls on ordinary pitches.</figcaption>
</figure>

| trigger, at ~4 calls on ordinary pitches | replanned within 150 ms of a gust (no gust) | hit quality: plain → gust |
|---|---|---|
| fixed rate | 100 % (100 %) | 0.89 → 0.44 |
| innovation trigger, frame camera | 31 % (21 %) | 0.47 → 0.30 |
| event trigger, event camera | 48 % (30 %) | 0.47 → 0.37 |

Events react sooner and more often after a gust, as a latency-free sensor should. As a way of spending planner calls, the event trigger is no better than the frame-based innovation trigger, and both lose to a fixed rate, for the reason of Part VI: on an ordinary pitch nothing contradicts the forecast, so innovation-type triggers fire on noise. And the planner they wake up is looking at frames that are 30 ms old. For the earlier trigger to pay, the events would have to feed the estimator, not only the trigger.

:::caution[Rendering from 24 processes]
Every worker of the evaluation pool renders. Two dozen EGL contexts on the GPU made the desktop driver reject commands. The study renders in software instead (Mesa llvmpipe, one thread per process), selected by probing the EGL devices (`eventcam.use_software_rendering`).
:::

:::tip[Where in the code]
`eval/eventcam.py` (`EventCamera`, `EventRig`, `run_event_episode`), `scripts/m5_events.py`; tables in `results/M5.md`.
:::
