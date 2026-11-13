---
title: "Lessons"
description: "What this project taught about when to think, about baselines and measurement, and about doing it on a CPU."
sidebar:
  order: 1
---

:::note[Guiding question]
What would be done differently, or the same, next time?
:::

## About when to think

**1. The value of a fresh plan peaks late; the change in the estimate peaks early.** The crossing-point forecast improves by centimetres per frame at the start of a pitch and by millimetres per frame at the end. But a 9 cm blade meeting a 7.3 cm ball is decided by those last millimetres, and the arm needs time to act on them. Any trigger that thresholds *how much the estimate changed* (drift, and variance, which is its expected value) spends its budget early and goes quiet exactly when a call is worth most. What should trigger a replan is the expected change in the *outcome*, not in the estimate. The learned trigger, which only ever saw outcomes, found that schedule by itself.

**2. An innovation trigger needs something to contradict.** On an ordinary pitch the camera never contradicts the forecast; it only sharpens it. The innovation χ² then measures noise, which is largest early, when the filter is least certain. Innovation-type triggers, frame-based or event-based, only earn their keep when the world departs from the model, which is what they were designed for, and are a poor way to schedule routine refinement.

**3. A strong baseline changes the conclusion.** Against a 50 Hz baseline (24 calls per pitch, quality 0.977) the drift and variance triggers look like savings: 19 and 23 calls for about the same quality. Against a fixed rate swept down to the same budgets, all three classical triggers lose at every budget. The brief's warning, *keep the analytic MPC strong*, applies to the baseline schedule just as much as to the planner.

**4. A learned policy can generalise to what it never saw, if it reads the right signals.** The trigger trained without surprises reacted to gusts as much as the one trained with them, because a gust moves the innovation and the drift, which it was already reading.

**5. Sometimes the best decision is not to think.** In the rally the filter is unsure of the ball's vertical speed for a few steps after every bounce. Replanning on those transient forecasts made the arms drop good strokes for ones they could no longer reach. Declining to replan until the filter's own uncertainty fell below a threshold took the rally from 4.5 to 10.0 returns, with fewer planner calls. The value of a planner call depends on the quality of what it is fed, and the filter reports that quality.

**6. Commit before you act.** A replan in the last 60 ms that switched between two strike planes, 60 ms apart at ping-pong speeds, ruined strokes that were about to land. A committed stroke may still be corrected; it may not be replaced.

## About measurement

**7. Reaction needs a placebo.** A trigger that calls the planner often will have a call soon after any event. Running every surprised pitch also without its surprise, with the same noise, separated reacting from calling anyway. Without that control the learned triggers would have looked as if they reacted in 92 % of gust pitches; against the placebo (77 %) the reaction is 15 points. And the median latency turned out to be the wrong statistic altogether for a trigger that already calls often: it was the same with and without the gust.

**8. Say which statistic a timing claim rests on.** A planner call takes 9.1 ms at the median and 21.5 ms at the 90th percentile, against a 10 ms control period. "Real-time" is true of the first number and false of the second. A first version of the M1 report said real-time from the median alone; the report now checks the tail. And timings taken inside a loaded process pool were about twice the single-core figures, so planning times are quoted from a separate, uncontended run.

**9. A surprising number is first a claim about the measurement.** Three examples here:
- Without speculative contacts the planner's swings were identical and correct, and the ball passed through the bat anyway. A tunnelling simulator would have made the planner look broken.
- A test comparing the extrapolated landing point with the continued simulation disagreed by up to 15 cm. The flight model was exact; the floor's speculative contact brakes the ball a step before it touches, so the "true" landing point read off the simulation was the wrong one.
- The first event-camera centroid was off by 20 to 100 pixels: other things in view were moving too. Masking the robot brought it to 0.3 pixel.

**10. Calibrate a parameter on what you are judged by.** The contact pair's restitution is 0.8; the impact behaves like 0.587 because the wrist gives. Fitted on velocity changes it comes out at 0.573; fitted on landing error, the thing hit quality measures, at 0.587. The difference is small, and it is the difference between landing short on average and not.

## About modelling

**11. Exactness is worth more than sophistication.** Choosing the `Euler` integrator made the ball's step explicit, so the flight model could be the simulator's step to $10^{-9}$ m. Everything downstream rests on that: the pitching machine hits its aim points, the filter needs almost no process noise, a landing point can be extrapolated instead of simulated, and in M4 the innovation measures the surprise and nothing else.

**12. Learning can replace the models at run time but still leaned on them to learn.** The world-model planner sees only raw fixes and hits 97 % of pitches. It learned from data made by the analytic batter, and it cannot aim: landing a ball on a 0.75 m disc five metres out needs a precision its outcome model does not reach. Hitting is a coarse target, aiming a fine one, and the fine one is where the model was worth having.

**13. Size the task to the actuator, and spend the freedom you have.** The Panda's joint speed limits cap the bat at about 2 m/s, and the pitches were sized to that. Getting even 2 m/s took constraining only the bat's speed along its face normal; asking for pure translation of the blade capped it at 0.9 m/s. The rally met the same wall: a ping-pong ball loses speed so fast that returning it across an ITTF table needs about twice the paddle speed a Panda has, so the rally is on a midi table with a long handle.

**14. A simulator's contact model is a modelling decision, and it has to be measured.** The rally's table bounce, with friction, returned the ball faster than it arrived in two bounces out of three: Coulomb's law held, energy did not. Frictionless, the bounce was an exact restitution law. Neither result was in the documentation; both took a thousand simulated bounces to see, and the first batch of them was aimed at the net.

## About doing it on a CPU

**15. `mujoco.batch` is not enough when the controller is Python.** The batch steps worlds in parallel, but the planner is single-threaded and dominates. Splitting pitches over 24 processes, each with a small batch and one simulation thread, is what made 3 seeds × every setting affordable.

**16. Rendering does not scale like simulation.** Two dozen processes rendering at once on the GPU made the desktop driver reject commands. Rendering in software (Mesa llvmpipe) through MuJoCo's EGL device selection fixed it; forcing software rendering with the usual environment variable segfaulted.
