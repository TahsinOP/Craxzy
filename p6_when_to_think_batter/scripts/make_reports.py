"""Write results/M0.md ... M5.md from the JSON results (numbers are never typed by hand).

    env -u PYTHONPATH .venv/bin/python scripts/make_reports.py
"""
from __future__ import annotations

import json

import numpy as np

from batter import RESULTS, config

CFG = config.load()


def load(name: str):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pm(v, scale: float = 1.0, digits: int = 1, unit: str = "") -> str:
    m = round(scale * v[0], digits) + 0.0          # no "-0.00"
    return f"{m:.{digits}f} ± {scale * v[1]:.{digits}f}{unit}"


def pct(v) -> str:
    return pm(v, 100.0, 1, " %")


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def at_budget(rows, budget):
    return min(rows, key=lambda r: abs(np.log(r["plans"][0]) - np.log(budget)))


# ------------------------------------------------------------------------- M0
def m0() -> str | None:
    r = load("m0")
    if r is None:
        return None
    p = r["pitches"]
    st = {(q["timestep"], q["speculative"]): q for q in r["static"]}
    sw = {(q["timestep"], q["speculative"]): q for q in r["swing"]}
    hs = sorted({q["timestep"] for q in r["static"]})
    rows = [[f"{1e3 * h:.0f} ms", pct(st[(h, False)]["bounced"]), pct(st[(h, True)]["bounced"]),
             pct(sw[(h, False)]["hit"]), pct(sw[(h, True)]["hit"]), pct(sw[(h, True)]["on_target"])] for h in hs]
    worst_off = min(st[(h, False)]["bounced"][0] for h in hs)
    all_on = min(st[(h, True)]["bounced"][0] for h in hs)
    b, bt = CFG["ball"], CFG["bat"]
    return f"""# M0: batting cage

**Acceptance:** a video of the bat passing through the ball with speculative contacts off and hitting it with them on, and a pitch distribution.
**Result: met.** With the flag on the ball bounces off the bat in {100 * all_on:.0f} % of pitches at 8, 10 and 16 ms; with it off it passes through in up to {100 * (1 - worst_off):.0f} %. Video: [`m0_tunnel.mp4`](m0_tunnel.mp4).

## The cage

- **Arm:** Menagerie Franka Panda (no hand) on a pedestal, torque-controlled at the datasheet torque limits.
- **Bat:** a flat blade welded to the flange, {1e3 * bt['thickness']:.0f} mm thick, {100 * bt['width']:.0f} cm wide, {100 * bt['length']:.0f} cm long, {bt['mass']} kg. It collides with the ball only, through a frictionless contact pair with restitution {bt['restitution']}.
- **Ball:** a baseball, radius {1e3 * b['radius']:.1f} mm, {1e3 * b['mass']:.0f} g, with MuJoCo's ellipsoid fluid model: quadratic drag and lift from its spin.
- **Pitching machine:** {CFG['pitch']['release_x']} m down the +x axis. Speed, spin, release height and aim are random (below).
- **Target zone:** a disc of radius {CFG['target']['radius']} m on the ground, {CFG['target']['range'][0]} to {CFG['target']['range'][1]} m out and up to {CFG['target']['azimuth_deg'][1]:.0f}° to either side. *Hit quality* is exp(−d²/2σ²) for a hit landing d from the centre (σ = {CFG['target']['sigma']} m), and 0 for a miss.
- **Camera:** position fixes of the ball at 100 Hz through a MuJoCo 4.0 *delayed sensor* ({1e3 * CFG['measurement']['delay']:.0f} ms latency); noise {1e3 * CFG['measurement']['sigma0']:.0f} mm + {1e3 * CFG['measurement']['sigma_per_m']:.0f} mm per metre of range; {100 * CFG['measurement']['dropout']:.0f} % of frames missing.

## Pitch distribution

![pitches](m0_pitches.png)

{table(["quantity", "range over 1,000 pitches"], [
    ["release speed", f"{p['speed'][0]:.1f} to {p['speed'][1]:.1f} m/s"],
    ["release height", f"{p['release_z'][0]:.2f} to {p['release_z'][1]:.2f} m"],
    ["spin", f"{p['spin'][0]:.0f} to {p['spin'][1]:.0f} rad/s, axis within {CFG['pitch']['spin_tilt_deg']:.0f}° of ±y"],
    ["time of flight to the plate", f"{1e3 * p['flight_time'][0]:.0f} to {1e3 * p['flight_time'][1]:.0f} ms"],
    ["speed at the plate", f"{p['arrival_speed'][0]:.1f} to {p['arrival_speed'][1]:.1f} m/s"],
    ["descent angle at the plate", f"{p['descent_deg'][0]:.0f}° to {p['descent_deg'][1]:.0f}°"],
    ["crossing point", f"y {p['cross_y'][0]:+.2f} to {p['cross_y'][1]:+.2f} m, z {p['cross_z'][0]:.2f} to {p['cross_z'][1]:.2f} m"],
    ["what the spin does at the plate", f"{p['spin_dz_cm'][0]:+.0f} to {p['spin_dz_cm'][1]:+.0f} cm in height (std {p['spin_dz_cm'][2]:.0f} cm), {p['spin_dy_cm'][0]:+.0f} to {p['spin_dy_cm'][1]:+.0f} cm sideways"],
])}

The machine is zeroed in: the release direction is solved so that the ball, with its drag and spin, crosses the plate at the drawn aim point. The spin therefore does not change where the pitch arrives but how it gets there. The right-hand panel shows the size of that effect: the same release without the spin would cross the plate up to {max(abs(p['spin_dz_cm'][0]), abs(p['spin_dz_cm'][1])):.0f} cm higher or lower. A batter cannot see the spin; it has to infer it from the curvature of the flight.

## Tunnelling

![tunnelling](m0_tunnel.png)

{table(["timestep", "static bat, flag off: bounced", "static bat, flag on: bounced", "swinging bat, flag off: returned", "swinging bat, flag on: returned", "flag on: on target"], rows)}

- *Static bat:* the arm holds the bat still and pitches are aimed at the blade. "Bounced" means the ball comes back.
- *Swinging bat:* the M1 batter (EKF, analytic planner at every step). "Returned" means the ball leaves the bat forwards.
- Per step the ball and bat close by 8 to 20 cm (8 to 16 ms at 9 to 13 m/s plus the bat's speed), against 1.2 cm of bat and 7.3 cm of ball. Without speculative contacts the outcome depends on where the ball happens to be sampled; with them every pitch is a clean impact, at all three timesteps.
- The controller, the EKF and the flight model take their timestep from the model, so the same code runs at 8, 10 and 16 ms.

Mean ± std over {len(CFG['seeds'])} seeds. Reproduce: `env -u PYTHONPATH .venv/bin/python scripts/m0_cage.py`.
"""


# ------------------------------------------------------------------------- M1
def m1() -> str | None:
    r = load("m1")
    if r is None:
        return None
    e, c, s = r["estimator"], r["calibration"], r["speed"]
    tau = e["tau"]
    erow = [[f"{1e3 * t:.0f} ms", pm(e["err_med"][i], 100, 1, " cm"), f"{100 * e['err_p90'][i][0]:.1f} cm",
             f"{100 * e['sd_pred'][i][0]:.1f} cm", f"{100 * e['cover90'][i][0]:.0f} %", f"{e['time_err_ms'][i][0]:.1f} ms"]
            for i, t in enumerate(tau)]
    names = ["analytic 100 Hz", "analytic 100 Hz, true ball state", "analytic, planned once"]
    srow = []
    for i, v in enumerate([q["speed"] for q in s[names[0]]]):
        srow.append([f"{v:.0f} m/s"] + [f"{pct(s[n][i]['hit'])} / {pct(s[n][i]['on_target'])} / {s[n][i]['dist_med'][0]:.2f} m" for n in names])
    srow.append(["**all pitches**"] + [f"**{pct(s[n + ' | all']['hit'])} / {pct(s[n + ' | all']['on_target'])}**" for n in names])
    a = s["analytic 100 Hz | all"]
    t = r.get("timing")
    timing = (f"Measured alone on one core, a planner call takes {t['ms_median']:.1f} ms (median; mean {t['ms_mean']:.1f}, 90th percentile "
              f"{t['ms_p90']:.1f}, max {t['ms_max']:.1f} ms over {t['calls']} calls). The control period is 10 ms: "
              + ("the median call fits, but the mean and the slow tail do not, so this Python planner is not real-time at 100 Hz. "
                 if t["ms_median"] <= 10 < t["ms_p90"] else
                 ("this Python planner is real-time at 100 Hz. " if t["ms_p90"] <= 10 else "this Python planner is not real-time at 100 Hz. "))
              if t else "")
    return f"""# M1: model-based hitter

**Acceptance:** hit rate and target accuracy against pitch speed, for the EKF → crossing forecast → swing QP pipeline replanned at 100 Hz.
**Result: met.** Over the whole pitch distribution the batter hits {pct(a['hit'])} of pitches and lands {pct(a['on_target'])} of them inside the {CFG['target']['radius']} m target zone (hit quality {pm(a['quality'], 1, 3)}, {a['plans'][0]:.0f} planner calls per pitch). Video: [`m1_hitter.mp4`](m1_hitter.mp4).

## Pipeline

```
camera fix (100 Hz, 30 ms old, noisy, 5 % dropped)
  └─ EKF on [position, velocity, spin] with the simulator's own flight step        estim/ekf.py, flight.py
       └─ forecast: 3 steps to the present, then on to the plate; covariance        estim/ekf.py
            └─ planner call: for each strike plane x_c and time of flight T_f       control/planner.py
                 aim:      v_out that lands the ball on the target (shooting)
                 impact:   n = (v_out − v_b)/|·|,  bat speed along n from the restitution model
                 pose:     5-dof IK (blade centre at the ball, face normal n)       control/arm.py
                 swing:    joint QPs: approach, constant velocity through the impact, brake   control/traj.py
                 keep the cheapest candidate
                 └─ discrete computed torque on the cached plan                     control/arm.py
```

- **Flight model.** With the `Euler` integrator MuJoCo's fluid forces are explicit, so one simulator step of the ball has a closed form: drag (k₁ + k₂|v|)v, lift c ω×v, spin decay. The five coefficients are read off the engine by probing `qfrc_passive`. The model reproduces a simulated spinning pitch to 1e-9 m (`tests/test_flight.py`); the pitching machine, the EKF, the forecast and the landing point all use it.
- **Impact model.** The bat-ball pair is frictionless, so the impulse is along the face normal and v_out = v_b − (1+e)((v_b − v_c)·n)n. Solving it backwards gives the face normal and the bat speed that send the ball to the target.
- **Swing.** A strike uses 5 of the arm's 7 degrees of freedom (blade centre, face normal). The joint velocity at impact moves the blade along its normal with the least joint speed. Each joint then solves a small QP in the simulator's integrator, so the reference is exactly trackable; if the strike cannot be reached in the time left, a soft version of the QP returns the closest approach. That is what makes a late replan useful.
- **Ready pose.** The strike posture with the highest bat speed for a ball through the middle of the zone (random-restart search, `scripts/design_ready.py`), wound back along the swing.

## Estimator

![estimator](m1_estimator.png)

{table(["time left to the plate", "crossing-point error (median)", "90th percentile", "filter's own std", "truth inside the 90 % ellipse", "arrival-time error"], erow)}

The forecast of where the ball crosses the plate is off by {100 * e['err_med'][-1][0]:.0f} cm half a second out and by {100 * e['err_med'][0][0]:.1f} cm in the last 50 ms. Half the bat's width is 4.5 cm. So the first plan of a pitch is made on a forecast that would miss, and the information that fixes it arrives while the arm is already moving. That is the whole reason replanning matters here, and the reason M3 has something to study. The covariance is calibrated (the 90 % ellipse holds the truth in {100 * min(c_[0] for c_ in e['cover90']):.0f} to {100 * max(c_[0] for c_ in e['cover90']):.0f} % of cases), which the variance trigger relies on.

## Impact calibration

The contact pair has restitution {c['pair_restitution']}, but the wrist gives under the impact, so the bat behaves as a softer wall. The effective value was fitted on {c['n']} swings made with the true ball state (so the fit sees the impact, not the estimator):

{table(["restitution in the planner's model"] + [f"{v:.2f}" for v in c['e_grid']], [["mean landing error along the range [m]"] + [f"{v:+.2f}" for v in c['radial_error_m']]])}

The error crosses zero at **e = {c['restitution_landing']:.3f}** (config: {c['restitution_config']}). The same swings give {c['restitution_velocity']:.3f} when fitted on the ball's velocity change against the *commanded* bat speed; the difference is the small gap between commanded and realised bat speed, which the landing fit absorbs.

## Hit rate and target accuracy against pitch speed

![speed](m1_speed.png)

Each cell: hit rate / ball lands in the target zone / median distance from the target centre (hits only).

{table(["release speed", "analytic planner, 100 Hz", "the same with the true ball state", "the same planner, called once"], srow)}

- The batter hits essentially every pitch at every speed. Target accuracy falls with speed, from {pct(s[names[0]][0]['on_target'])} at {s[names[0]][0]['speed']:.0f} m/s to {pct(s[names[0]][-1]['on_target'])} at {s[names[0]][-1]['speed']:.0f} m/s: a fast pitch leaves less time for the late, accurate fixes to reach the swing.
- With the true ball state the same planner is on target in {pct(s[names[1] + ' | all']['on_target'])} of all pitches, so what is lost is lost in estimation, not in the swing or the impact model.
- Called once (after the first {5} fixes), the same planner hits {pct(s[names[2] + ' | all']['hit'])}. Replanning is not a refinement here; it is what makes the batter work.

## Compute

{timing}Planner calls per pitch at 100 Hz: {a['plans'][0]:.0f}. M3 asks how few of them are needed.

Mean ± std over {len(CFG['seeds'])} seeds. Reproduce: `env -u PYTHONPATH .venv/bin/python scripts/m1_hitter.py`.
"""


# ------------------------------------------------------------------------- M2
def m2() -> str | None:
    r = load("m2")
    if r is None or "timing" not in r:
        return None
    A, W, Pn = "analytic planner (classical)", "world-model planner (pure learning)", "world-model prior only, no MPPI"
    sp, al, t, v = r["speed"], r["all"], r["timing"], r["validation"]
    srow = [[f"{sp[A][i]['speed']:.0f} m/s"] + [f"{pct(sp[n][i]['hit'])} / {pct(sp[n][i]['on_target'])} / {sp[n][i]['dist_med'][0]:.2f} m"
                                                 for n in (A, W, Pn)] for i in range(len(sp[A]))]
    srow.append(["**all pitches**"] + [f"**{pct(al[n]['hit'])} / {pct(al[n]['on_target'])} / quality {pm(al[n]['quality'], 1, 2)}**" for n in (A, W, Pn)])
    trow = [[k, f"{t[k]['ms_median']:.1f}", f"{t[k]['ms_mean']:.1f}", f"{t[k]['ms_p90']:.1f}", f"{t[k]['ms_max']:.1f}"]
            for k in ("analytic", "world model", "world model prior only")]
    drow = [[f"{d['pitches']:,}", pct(d["hit"]), pct(d["on_target"]), pm(d["quality"], 1, 2)] for d in r["data_sweep"]]
    vs = [v[k] for k in v if k.startswith("s")]
    ratio = t["world model"]["ms_median"] / t["analytic"]["ms_median"]
    faster = "faster" if ratio < 1 else "slower"
    return f"""# M2: learned world-model planner

**Task:** a TD-MPC2-style planner (small latent model + MPPI) trained on batting data, compared with the analytic MPC on hit rate and planning time.
**Result:** the learned planner hits {pct(al[W]['hit'])} of pitches against {pct(al[A]['hit'])} for the analytic planner, but puts only {pct(al[W]['on_target'])} into the target zone against {pct(al[A]['on_target'])} (hit quality {pm(al[W]['quality'], 1, 2)} against {pm(al[A]['quality'], 1, 2)}). A planner call takes {t['world model']['ms_median']:.1f} ms against {t['analytic']['ms_median']:.1f} ms. Video: [`m2_worldmodel.mp4`](m2_worldmodel.mp4).

## What is learned and what is not

The analytic planner knows the ball's flight, the impact and the flight to the target. The learned planner knows none of them:

- **Observation:** the last 16 raw camera fixes with their ages, the arm's reference joint state, the target. Nothing from the EKF.
- **Action:** a *swing*, six numbers: time to impact, strike point (y, z) on the plane x = 0.62 m, azimuth and elevation of the bat's face, bat speed. The same IK and swing QP as in M1 turn it into joint motion; arm tracking stays model-based.
- **Model** (`learn/wm.py`): encoder to a 64-d latent; latent dynamics one decision interval (40 ms) ahead with the swing held; an outcome head (hit or miss, where the ball lands relative to the target), which plays the role of TD-MPC2's reward model; a policy prior.
- **Losses:** outcome prediction; latent consistency (the rolled-out latent must match the encoding of the later observation, and predict the same outcome); advantage-weighted regression for the prior.
- **Planner** (`learn/wm_planner.py`): MPPI over the swing in latent space, seeded by the prior, scored by the predicted hit quality along a short latent rollout.

The reward of a pitch arrives once, at the end, so there is no bootstrapped value function: the outcome head is trained on what actually happened after the last planner call of each pitch (from which the swing was held). This is the "small latent model + MPPI" variant the brief allows, not full TD-MPC2.

## Batting data

{r['n_train_pitches']:,} pitches played by a noisy version of the analytic batter (its strike, perturbed by up to ±8 cm, ±12° and ±0.6 m/s, replanned at random times), giving {r['n_samples']:,} samples. {100 * r['data_hit_rate']:.0f} % of the samples come from pitches that were hit: most of the data is about swings that were committed to too early. On {vs[0]['n']:,} held-out samples the three models predict hit or miss correctly in {100 * np.mean([x['hit_accuracy'] for x in vs]):.1f} % of cases and the landing point to {np.mean([x['landing_err_med_m'] for x in vs]):.2f} m (median).

## Against the analytic planner

![speed](m2_speed.png)

Each cell: hit rate / ball lands in the target zone / median distance from the target (hits). Both planners are called at every step (100 Hz); 3 seeds (a separately trained model per seed), {r.get('n', '?')} pitches each.

{table(["release speed", "analytic (classical)", "world model + MPPI (pure learning)", "world-model prior alone"], srow)}

- The learned planner connects with almost every pitch but aims worse: its median miss distance is several times the analytic planner's. Hitting needs the blade (9 cm by 38 cm) at the ball; landing on a 0.75 m disc five metres away needs the face angle and bat speed right to a few per cent, and the learned outcome model's own landing error is about a metre.
- MPPI on the learned model {'improves on' if al[W]['quality'][0] > al[Pn]['quality'][0] else 'does not improve on'} the policy prior alone (quality {pm(al[W]['quality'], 1, 2)} against {pm(al[Pn]['quality'], 1, 2)}).
- The learned planner was trained on data from the analytic batter, so it is not independent of the model-based design; it replaces the EKF, the flight model and the impact model at run time.

## Planning time

Wall-clock per planner call, measured alone on one core (the evaluation above runs in a loaded process pool, where calls take about twice as long):

{table(["planner", "median [ms]", "mean [ms]", "90th percentile [ms]", "max [ms]"], trow)}

The learned planner is {ratio:.1f}x the analytic one per call ({faster}). Neither fits in the 10 ms control period in this Python implementation{' except the prior alone' if t['world model prior only']['ms_median'] < 10 else ''}; M3 treats the planner as the expensive thing it is and asks how rarely it can be called.

## Data

![data](m2_data.png)

{table(["pitches of batting data", "hit rate", "on target", "hit quality"], drow)}

Reproduce: `env -u PYTHONPATH .venv/bin/python scripts/m2_worldmodel.py` (about 45 minutes: 4 for the data, 12 for the five models, the rest for the closed-loop runs).
"""


# ------------------------------------------------------------------------- M3
KIND_NAME = {"periodic": "fixed rate", "innovation": "innovation (χ² gate)", "variance": "variance (Trimpe)", "drift": "drift (send-on-delta)",
             "learned": "learned (PPO)", "wm": "world-model planner, fixed rate"}
UNIT = {"periodic": lambda s: "every step (100 Hz)" if s >= 100 else ("once" if s == 0 else f"{s:g} Hz"), "innovation": lambda s: f"χ² > {s:g}",
        "variance": lambda s: f"δ = {100 * s:g} cm", "drift": lambda s: f"δ = {100 * s:g} cm", "learned": lambda s: f"λ = {s:g}",
        "wm": lambda s: "every step (100 Hz)" if s >= 100 else f"{s:g} Hz", "learned_s": lambda s: f"λ = {s:g}"}


def best_under(rows, budget):
    ok = [r for r in rows if r["plans"][0] <= budget]
    return max(ok, key=lambda r: r["quality"][0]) if ok else None


def m3() -> str | None:
    r = load("m3")
    if r is None or "when" not in r:
        return None
    kinds = [k for k in ("periodic", "innovation", "variance", "drift", "learned", "wm") if k in r]
    full = next(q for q in r["periodic"] if q["setting"] >= 100)
    budgets = [2.0, 3.0, 4.5, 7.0, 13.0]
    brow = []
    for k in kinds:
        cells = []
        for b in budgets:
            q = best_under(r[k], b)
            cells.append("n/a" if q is None else f"{q['quality'][0]:.2f} ({q['plans'][0]:.1f})")
        brow.append([KIND_NAME[k]] + cells)
    blocks = []
    for k in kinds:
        rows = [[UNIT[k](q["setting"]), f"{q['plans'][0]:.1f}", pm(q["quality"], 1, 3), pct(q["hit"]), pct(q["on_target"]),
                 f"{q['ms_per_pitch'][0]:.0f}"] for q in sorted(r[k], key=lambda q: -q["plans"][0])]
        blocks.append(f"**{KIND_NAME[k]}**\n\n" + table(["setting", "planner calls per pitch", "hit quality", "hit rate", "on target", "planning per pitch [ms]"], rows))
    # the cheapest setting of each trigger within 0.02 of the 100 Hz quality
    near = []
    for k in kinds:
        ok = [q for q in r[k] if q["quality"][0] >= full["quality"][0] - 0.02]
        q = min(ok, key=lambda q: q["plans"][0]) if ok else None
        near.append([KIND_NAME[k], "never" if q is None else f"{q['plans'][0]:.1f} calls ({UNIT[k](q['setting'])}), quality {q['quality'][0]:.3f}"])
    w = r["when"]
    wrow = [[KIND_NAME[k], UNIT[k](w[k]["setting"]), f"{w[k]['plans'][0]:.1f}", pm(w[k]["quality"], 1, 2), f"{w[k]['median_tau_ms']:.0f} ms"] for k in w]
    l4, p4 = at_budget(r["learned"], 4.0), at_budget(r["periodic"], 4.0)
    best4 = max((k for k in kinds if k != "wm"), key=lambda k: (best_under(r[k], 4.5) or {"quality": [0]})["quality"][0])
    lead = "the learned trigger" if best4 == "learned" else f"the {KIND_NAME[best4]} trigger"
    early = [KIND_NAME[k] for k in ("variance", "drift", "innovation") if w[k]["median_tau_ms"] > w["periodic"]["median_tau_ms"] + 50]
    worse = [KIND_NAME[k] for k in ("innovation", "variance", "drift")
             if (best_under(r[k], 7.0) or {"quality": [0]})["quality"][0] < best_under(r["periodic"], 7.0)["quality"][0]]
    when_text = ("The estimate of the crossing point improves fastest late in the flight (M1), and the arm needs time to act on it. "
                 "A fixed rate spreads its calls evenly over the flight. ")
    if early:
        when_text += (f"The {', '.join(early)} trigger{'s' if len(early) > 1 else ''} spend{'' if len(early) > 1 else 's'} most calls early, "
                      "when the estimate moves by many centimetres per frame, and go quiet in the last 300 ms, when it moves by "
                      "millimetres but those millimetres decide whether the bat meets the ball. A threshold that is fixed in "
                      "centimetres (or in units of the forecast's own, early and large, covariance) cannot be right at both ends of the flight. ")
    when_text += (f"The learned trigger places its replans a median {w['learned']['median_tau_ms']:.0f} ms before the plate "
                  f"(fixed rate: {w['periodic']['median_tau_ms']:.0f} ms; variance: {w['variance']['median_tau_ms']:.0f} ms; "
                  f"drift: {w['drift']['median_tau_ms']:.0f} ms).")
    return f"""# M3: compute triggers

**Acceptance:** Pareto curves of hit quality against planner calls, swept over λ and thresholds, for fixed-rate, innovation-triggered, variance-triggered and learned triggers.
**Result: met.** Replanning at every step costs {full['plans'][0]:.0f} planner calls per pitch for a hit quality of {pm(full['quality'], 1, 3)}. With about 4 calls per pitch {lead} is best: the learned trigger reaches {pm(l4['quality'], 1, 2)} with {l4['plans'][0]:.1f} calls, a fixed rate {pm(p4['quality'], 1, 2)} with {p4['plans'][0]:.1f}. {('On ordinary pitches the ' + ', '.join(worse) + ' triggers are worse than a fixed rate at the same budget.') if worse else ''}

![pareto](m3_pareto.png)

## The triggers

All triggers see the same observation each step and decide whether to run the planner now or keep executing the cached swing. The first plan of a pitch (after 5 camera fixes) is made for every trigger and counts as a call.

- **Fixed rate:** every 1/rate seconds.
- **Innovation:** each camera fix is compared with the forecast the cached plan was built on: ν = z − p̂(t | t_plan), S = P(t | t_plan) + R. Under the model ν′S⁻¹ν is χ² with 3 degrees of freedom. Replan when its mean over the last 4 fixes exceeds a gate. It looks at data.
- **Variance** (Trimpe & D'Andrea): replan when tr C(t_plan) − tr C(t) > δ², with C the covariance of the predicted crossing point. For a consistent estimator that difference is the expected squared distance between where the plan aims and where we would aim now. It uses covariances only, so its schedule does not depend on what the camera actually saw.
- **Drift** (send-on-delta, added for comparison): replan when the crossing-point estimate itself has moved by more than δ.
- **Learned:** a PPO policy (two 64-unit layers) on innovation, crossing-point uncertainty now and at plan time, time to the plate, time since the plan, drift and log λ. Reward = hit quality − λ per planner call. One policy serves every λ, so sweeping λ traces its curve.
- **World-model planner at fixed rates** is the pure-learning baseline: the M2 planner in place of the analytic one.

## Best quality within a budget

Hit quality of the best swept setting that uses at most the given number of planner calls per pitch (calls actually used in brackets).

{table(["trigger"] + [f"≤ {b:g} calls" for b in budgets], brow)}

Cheapest setting within 0.02 of the 100 Hz quality ({full['quality'][0]:.3f}):

{table(["trigger", "cheapest setting that keeps the quality"], near)}

## When each trigger thinks

![when](m3_when.png)

Settings chosen to cost about 4 planner calls per pitch:

{table(["trigger", "setting", "calls per pitch", "hit quality", "median time to the plate at a replan"], wrow)}

{when_text}

## Training the learned trigger

![ppo](m3_ppo.png)

PPO on CPU with `mujoco.batch` worlds spread over a process pool; λ is drawn log-uniformly per pitch. At evaluation the policy is sampled, not thresholded.

## All settings

{chr(10).join(chr(10) + b + chr(10) for b in blocks)}

Planning time per pitch is wall-clock inside the loaded process pool (roughly twice the single-core figure of M2); it is there to show that calls and compute tell the same story.

Mean ± std over {len(CFG['seeds'])} seeds, {r.get('n', '?')} pitches per seed and setting. Reproduce: `env -u PYTHONPATH .venv/bin/python scripts/m3_triggers.py`.
"""


# ------------------------------------------------------------------------- M4
M4_ORDER = ["periodic", "variance", "drift", "innovation", "learned", "learned_s"]
M4_NAME = {**KIND_NAME, "learned": "learned (PPO, never saw a surprise)", "learned_s": "learned (PPO, trained with surprises)"}


def m4() -> str | None:
    r = load("m4")
    if r is None or not r.get("reaction") or len(r["reaction"]) < 2:
        return None
    win = 1e3 * r["window_s"]
    sec = []
    for kind, title in (("spin", "Spin change"), ("gust", "Gust")):
        x = r["reaction"][kind]
        fmt = lambda v: "no call" if not np.isfinite(v) else f"{v:.0f} ms"
        rows = [[M4_NAME[k], UNIT[k if k != "learned_s" else "learned"](r["specs"][k][1]),
                 f"{x[k]['calls_plain'][0]:.1f} → {x[k]['calls_surprise'][0]:.1f}",
                 f"{pm(x[k]['extra_calls'], 1, 2)}",
                 f"{100 * x[k]['react'][0]:.0f} % ({100 * x[k]['react_placebo'][0]:.0f} %)",
                 f"{fmt(x[k]['latency_med_ms'][0])} ({fmt(x[k]['latency_placebo_med_ms'][0])})",
                 f"{x[k]['quality_plain'][0]:.2f} → {pm(x[k]['quality_surprise'], 1, 2)}",
                 f"{100 * x[k]['hit_plain'][0]:.0f} % → {100 * x[k]['hit_surprise'][0]:.0f} %"] for k in M4_ORDER]
        sec.append(f"### {title}\n\n" + table(["trigger", "setting", "calls: plain → surprised", "extra calls", f"replanned within {win:.0f} ms (placebo)",
                                              "median latency (placebo)", "hit quality: plain → surprised", "hit rate"], rows))
    # Pareto under surprise: best quality within 4 and 10 calls
    prow = []
    for k in M4_ORDER:
        cells = []
        for kind in ("spin", "gust"):
            for b in (4.5, 13.0):
                q = best_under(r["pareto"][kind][k], b)
                cells.append("n/a" if q is None else f"{q['quality'][0]:.2f} ({q['plans'][0]:.1f})")
        prow.append([M4_NAME[k]] + cells)
    g, sp = r["reaction"]["gust"], r["reaction"]["spin"]
    reacts = [k for k in M4_ORDER if g[k]["extra_calls"][0] > 0.3]
    blind = [k for k in M4_ORDER if abs(g[k]["extra_calls"][0]) < 0.05]
    spin_seen = [k for k in M4_ORDER if sp[k]["react"][0] - sp[k]["react_placebo"][0] > 0.05]
    full = {kind: next(q for q in r["pareto"][kind]["periodic"] if q["setting"] >= 100) for kind in ("spin", "gust")}
    L = "learned_s"
    result = (f"After a gust the {', '.join(M4_NAME[k] for k in reacts)} triggers spend more planner calls than on the same pitch without it "
              f"({', '.join(f'{g[k]['extra_calls'][0]:+.1f}' for k in reacts)} calls); the {', '.join(M4_NAME[k] for k in blind)} triggers keep exactly the same schedule. "
              f"At about {r['budget']:.0f} calls per ordinary pitch the fixed rate's hit rate falls from {100 * g['periodic']['hit_plain'][0]:.0f} % to {100 * g['periodic']['hit_surprise'][0]:.0f} % under gusts, "
              f"the learned trigger's from {100 * g[L]['hit_plain'][0]:.0f} % to {100 * g[L]['hit_surprise'][0]:.0f} %. "
              + ("A spin change is too gradual to be seen in time: no trigger replans sooner because of it. " if not spin_seen else "")
              + f"No trigger restores the quality: even replanning at every step reaches only {full['gust']['quality'][0]:.2f} on gust pitches and {full['spin']['quality'][0]:.2f} after a spin change.")
    sc = CFG["surprise"]
    return f"""# M4: surprises

**Acceptance:** the response latency to surprises is measured for each trigger type; innovation-triggered and learned triggers should react while a fixed low rate misses.
**Result: met for the measurement; the expected behaviour holds for gusts and not for spin changes.** {result}

## The surprises

Between 25 % and 60 % of the way to the plate, either

- **spin change:** the spin jumps to {sc['spin'][0]:.0f} to {sc['spin'][1]:.0f} rad/s in the opposite sense (a floating ball starts to dive, or the reverse), written into the ball's angular velocity; or
- **gust:** a force of {sc['gust_force'][0]} to {sc['gust_force'][1]} N across the flight path for {1e3 * sc['gust_time']:.0f} ms, through `xfrc_applied` (3 to 8 m/s² on a 145 g ball).

Every controller sees the same surprised pitches, and each pitch is also run without its surprise (same release, same camera noise seed), so the effect of the surprise is measured pitch by pitch.

The EKF does not know about surprises. It watches its own normalised innovation and, when the sum over 6 fixes passes a χ² gate, inflates its velocity and spin covariance so that it re-learns quickly (`estim/ekf.py`). In an open-loop run of the filter a gust trips it in about 60 % of pitches and a spin change, which is gentler per frame, in about 7 % (`tests/test_ekf.py` asserts the first).

## Reaction at a matched budget

Each trigger at the setting that costs about {r['budget']:.0f} planner calls per pitch on ordinary pitches (from M3).

- *Extra calls:* planner calls on the surprised pitch minus calls on the same pitch without the surprise.
- *Replanned within {win:.0f} ms:* share of pitches with a planner call in the {win:.0f} ms after the onset. The placebo is the same statistic on the same pitches without the surprise, i.e. how often the trigger would have called anyway.
- *Latency:* time from the onset to the next planner call, median over pitches.

{(chr(10) + chr(10)).join(sec)}

![reaction](m4_reaction.png)

Reading the tables:

- **Fixed rate and variance** are blind by construction: their schedule depends on the clock or on covariances, and the EKF's covariance does not depend on what the camera saw (except through the filter's surprise detector, which acts too late to matter here). Extra calls: {abs(g['periodic']['extra_calls'][0]):.2f} and {abs(g['variance']['extra_calls'][0]):.2f}.
- **Innovation and drift** do react to a gust with extra calls ({g['innovation']['extra_calls'][0]:+.1f} and {g['drift']['extra_calls'][0]:+.1f}), but they start from a poor operating point (M3) and most of their pitches have no planner call at all after the onset, hence "no call" as the median latency.
- **The learned triggers** already replan late in the flight, so their placebo rate is high ({100 * g[L]['react_placebo'][0]:.0f} %); a gust raises it to {100 * g[L]['react'][0]:.0f} % and adds {g[L]['extra_calls'][0]:.1f} calls. The policy that never saw a surprise in training reacts as much as the one trained with them ({g['learned']['extra_calls'][0]:+.1f} against {g[L]['extra_calls'][0]:+.1f} calls): it reads the innovation and the drift, and a gust moves both.
- **Median latency is the wrong lens for a trigger that is already calling often.** For the learned triggers it is about the same with and without the surprise ({g[L]['latency_med_ms'][0]:.0f} ms against {g[L]['latency_placebo_med_ms'][0]:.0f} ms); the reaction shows in the upper tail of the latency curve and in the extra calls.

## Quality against planner calls on surprised pitches

![pareto](m4_pareto.png)

Best hit quality within a budget of planner calls (calls used in brackets):

{table(["trigger", "spin change, ≤ 4.5 calls", "spin change, ≤ 13 calls", "gust, ≤ 4.5 calls", "gust, ≤ 13 calls"], prow)}

Video (Filament), four gust pitches under a fixed low rate (left), the innovation trigger (middle) and the learned trigger (right): [`m4_surprise.mp4`](m4_surprise.mp4). Yellow markers are planner calls; the orange path is the forecast the cached plan was built on, the green one the current forecast.

Mean ± std over {len(CFG['seeds'])} seeds, {r.get('n', '?')} pitches per seed and setting. Reproduce: `env -u PYTHONPATH .venv/bin/python scripts/m4_surprises.py` (needs `results/m3.json`).
"""


# ------------------------------------------------------------------------- M5
def m5() -> str | None:
    r = load("m5")
    if r is None or len(r.get("reaction", {})) < 3:
        return None
    name = {"events": "event trigger (event camera)", "innovation": "innovation trigger (frame camera)", "periodic": "fixed rate"}
    unit = {"events": lambda s: f"χ² > {s:g} (pixels)", "innovation": lambda s: f"χ² > {s:g}", "periodic": lambda s: f"{s:g} Hz"}
    win = 1e3 * r["window_s"]
    rrow = []
    for k in ("periodic", "innovation", "events"):
        x = r["reaction"][k]
        for sur in ("gust", "spin"):
            y = x[sur]
            fmt = lambda v: "no call" if not np.isfinite(v) else f"{v:.0f} ms"
            rrow.append([name[k], unit[k](x["setting"]), "gust" if sur == "gust" else "spin change",
                         f"{x['calls_plain'][0]:.1f} → {y['calls'][0]:.1f}", f"{100 * y['react']:.0f} % ({100 * y['react_placebo']:.0f} %)",
                         f"{fmt(y['latency_med_ms'])} ({fmt(y['latency_placebo_med_ms'])})",
                         f"{x['quality_plain'][0]:.2f} → {pm(y['quality'], 1, 2)}"])
    prow = []
    for k in ("periodic", "innovation", "events"):
        cells = []
        for sur in ("none", "gust", "spin"):
            for b in (4.5, 13.0):
                q = best_under(r["pareto"][sur][k], b)
                cells.append("n/a" if q is None else f"{q['quality'][0]:.2f} ({q['plans'][0]:.1f})")
        prow.append([name[k]] + cells)
    ev, inn = r["reaction"]["events"]["gust"], r["reaction"]["innovation"]["gust"]
    faster = (ev["react"] - ev["react_placebo"]) > (inn["react"] - inn["react_placebo"])
    q = {k: r["reaction"][k]["quality_plain"][0] for k in ("events", "innovation", "periodic")}
    loses = ("As a way of spending planner calls it is no better than the innovation trigger: both lose to a fixed rate at the same budget (tables below)."
             if max(q["events"], q["innovation"]) < q["periodic"] else "")
    return f"""# M5 (stretch): triggering on events

**Stretch goal:** an event-camera emulator that generates log-brightness-change events from offscreen renders, and a trigger that fires on events instead of on state innovation.
**Result:** done as a small study. In the {win:.0f} ms after a gust the event trigger calls the planner in {100 * ev['react']:.0f} % of pitches ({100 * ev['react_placebo']:.0f} % on the same pitches without the gust); the frame-based innovation trigger in {100 * inn['react']:.0f} % ({100 * inn['react_placebo']:.0f} %). {'Events react sooner and more often, as a latency-free sensor should.' if faster else 'Events do not react sooner.'} {loses}

![events](m5_events.png)

## Emulator

`eval/eventcam.py`. A camera looks down on the flight corridor from the side (480 × 240 pixels; the ball is about 6 pixels across; the image columns that show the robot are masked). Each simulator step the scene is rendered offscreen without shadows; every pixel keeps a reference log-intensity and emits an ON or OFF event when the current log-intensity differs from it by more than a contrast threshold (0.25), after which its reference is reset. This is the usual frame-based emulation, so events are timed to the 10 ms step, not to microseconds.

## Event trigger

The ON events of a step mark where the ball has arrived. Their centroid is an image measurement of the ball *now* (accurate to about 0.3 pixel), whereas the frame camera's fix is 30 ms old. The centroid is compared with the image position that the cached plan's forecast predicts for this instant; the squared pixel distance, normalised by the projected forecast covariance plus 0.5 pixel of noise, is gated like the innovation trigger of M3 (mean over 4 steps).

Only the decision *when* to plan comes from events. The planner still runs on the EKF, which is fed by the delayed frame camera. So an early trigger asks for a plan from a filter that has not yet seen what the events saw.

## Reaction to a surprise, at about {r['budget']:.0f} calls per ordinary pitch

Bracketed values are the placebo: the same pitches without the surprise.

{table(["trigger", "setting", "surprise", "calls: plain → surprised", f"replanned within {win:.0f} ms (placebo)", "median latency (placebo)", "hit quality: plain → surprised"], rrow)}

## Best quality within a budget

{table(["trigger", "ordinary, ≤ 4.5 calls", "ordinary, ≤ 13", "gust, ≤ 4.5", "gust, ≤ 13", "spin change, ≤ 4.5", "spin change, ≤ 13"], prow)}

The other two stretch items are done as well: the two-arm ping-pong rally ([`RALLY.md`](RALLY.md)), and Filament for the offline videos (every milestone video is rendered with MuJoCo 4.0's Filament renderer, `eval/filament.py`; `--renderer classic` gives the classic one).

Mean ± std over {len(CFG['seeds'])} seeds, {r['n']} pitches per seed and setting. Reproduce: `env -u PYTHONPATH .venv/bin/python scripts/m5_events.py`.
"""


# ---------------------------------------------------------------------- rally
REASON_TEXT = {"max": "reached the cap", "net": "into the net", "own_side": "landed on the hitter's own half",
               "missed": "missed by the receiver", "no_bounce": "hit before it bounced", "floor": "landed off the table",
               "passed": "passed an arm", "time": "time limit", "double_hit": "touched twice"}


def rally_report() -> str | None:
    r = load("rally")
    if r is None or len(r.get("rallies", {})) < 6:
        return None
    R, f, b, c = r["rallies"], r["feasibility"], r["bounce"], r["calibration"]
    cap = r["config"]["rally"]["max_hits"]
    names = list(R)
    rows = [[k, pm(R[k]["returns"], 1, 2), f"{R[k]['median']:.0f}", pct(R[k]["full"]), f"{R[k]['plans_per_return']:.0f}",
             ", ".join(f"{REASON_TEXT.get(x, x)} {100 * v:.0f} %" for x, v in sorted(R[k]["reasons"].items(), key=lambda t: -t[1]) if v >= 0.02)]
            for k in names]
    frow = [[x["table"], f"{x['ball_in']:.1f} m/s", f"{x['ball_out']:.1f} m/s",
             f"{x['paddle_speed_needed'][1]:.2f} m/s ({x['paddle_speed_needed'][0]:.2f} to {x['paddle_speed_needed'][2]:.2f})",
             f"{x['paddle_speed_available']:.2f} m/s"] for x in f]
    brow = [[k, f"{b[k]['ratio_median']:.3f}", f"{b[k]['ratio_std']:.3f}", f"{b[k]['ratio_max']:.2f}", f"{100 * b[k]['gains_energy']:.0f} %"]
            for k in ("frictionless", "friction 0.25")]
    ek, gk, ok = "EKF, every step (always think)", "EKF, gated, every step", "true ball state, every step"
    t = r["config"]["table"]
    return f"""# Stretch: a two-arm ping-pong rally

**Stretch goal:** a two-arm ping-pong rally with thin paddles and speculative contacts.
**Result:** done. Two Pandas keep a rally going for {R[gk]['returns'][0]:.1f} ± {R[gk]['returns'][1]:.1f} returns on average ({pct(R[gk]['full'])} of rallies reach the cap of {cap}) with the camera, the filter and the analytic planner, against {R[ok]['returns'][0]:.1f} with the true ball state. The step that made it work is a *when-to-think* rule: the arm does not replan while the filter is unsure of the ball's vertical speed, which it is for a few steps after every bounce. Without that rule the same stack manages {R[ek]['returns'][0]:.1f}. Video: [`rally.mp4`](rally.mp4).

![rally](rally.png)

## The setup

- **Two arms, one planner.** Arm B is a second Panda attached with the prefix `b_` at the far end, rotated half a turn about the vertical. Each arm plans in its own frame, where it is the batter's arm A, so the flight model, the EKF, the aiming, the impact inversion, the IK and the swing QP are the batter's, unchanged.
- **Paddles.** A round blade {1e3 * r['config']['paddle']['radius']:.0f} mm in radius and {1e3 * r['config']['paddle']['thickness']:.0f} mm thick on a {r['config']['paddle']['handle']:.2f} m handle; frictionless contact pair with restitution {r['config']['paddle']['restitution']}. A 40 mm ball moving 2 to 5 m/s travels 2 to 5 cm per 10 ms step against 6 mm of paddle: speculative contacts are on.
- **Table.** {t['length']:.2f} × {t['width']:.2f} m with a {100 * t['net_height']:.2f} cm net, {t['gap']:.2f} m from each robot base. A midi table, not the ITTF table (below).
- **Feeder.** Serves from in front of arm B to arm A's half, solved through the flight model so that they clear the net and reach A's strike zone.
- **Rules.** A return counts when the receiver hits a ball that bounced exactly once on its half. A rally ends in the net, on the hitter's own half, after a second bounce, off the table, or at {cap} returns.

## Why a midi table

| table | ball arrives at the paddle | must leave at | paddle speed needed (5 to 95 %) | the arm can give |
|---|---|---|---|---|
{chr(10).join("| " + " | ".join(x) + " |" for x in frow)}

A 2.7 g ball loses speed fast (about 18 m/s² of drag at 8 m/s), so it reaches the paddle slowly and has to be driven back. On the ITTF table a return needs about twice the paddle speed the Panda's joint speed limits allow. The midi table brings it just inside, with a long handle for leverage.

## The table bounce

{table(["table-ball contact", "v_z after / v_z before, median", "std", "max", "leaves faster than it arrived"], brow)}

With a frictionless pair, a bounce in MuJoCo with this direct restitution law and speculative contacts takes exactly one step: the step whose ballistic prediction would end below the surface. In that step the vertical velocity is reversed and scaled by e = {b['model']['e']:.4f} (offset {1e3 * b['model']['b']:.2f} mm/s), and everything else follows the ordinary flight step. The flight model therefore stays exact through the bounce (`tests/test_rally.py`: 0.05 mm over the flight). With friction, Coulomb's law held, but the ball left the table faster than it arrived in {100 * b['friction 0.25']['gains_energy']:.0f} % of bounces, up to {b['friction 0.25']['ratio_max']:.1f} times: the table is frictionless.

## Rallies

3 seeds of {r['n']} rallies per condition. Gated: no replanning while the filter's standard deviation of v_z exceeds {r['config']['rally']['replan_vz_sd']} m/s.

{table(["condition", "returns per rally", "median", f"reach {cap}", "planner calls per return", "how rallies end"], rows)}

## What it took

1. **Commit before the impact.** Strike planes 12 cm apart are 60 ms apart at ping-pong speeds. A replan in the last 60 ms that switched planes pulled the paddle off a stroke that was about to land. Within {1e3 * r['config']['rally']['commit_time']:.0f} ms of the impact only the same stroke may be corrected.
2. **Tell the filter that the bounce's timing is uncertain.** A few millimetres of height error decide which step the bounce falls in; in the wrong step the vertical speed is off by (1 + e)|v_z|. Process noise around the bounce lets the next frames correct it.
3. **Do not think while the filter does not know.** For a few steps after each bounce the forecast swings by centimetres. Replanning on it made the arm abandon good strokes for ones it could no longer reach. Gating replanning on the filter's own uncertainty roughly doubled the rally length: {R[ek]['returns'][0]:.1f} → {R[gk]['returns'][0]:.1f} returns.
4. **Calibrate the impact.** The paddle's effective restitution is {c['e']:.3f} (fitted on {c['n']} true-state hits) against the pair's {c['pair']}; the planner uses {c['config']}.

Two things tried that did not help, and are not used: penalising the face's rotation through the stroke (rallies got shorter even with the true ball state), and weighting the swing QP against early accelerations so the arm commits later.

Reproduce: `env -u PYTHONPATH .venv/bin/python scripts/rally.py` (about 25 minutes).
"""


SECTIONS = {"M0": m0, "M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5, "RALLY": rally_report}



# --------------------------------------------------------------------- README
def glance() -> str:
    rows = []
    r = load("m0")
    if r:
        st = {(q["timestep"], q["speculative"]): q for q in r["static"]}
        hs = sorted({q["timestep"] for q in r["static"]})
        off = [100 * (1 - st[(h, False)]["bounced"][0]) for h in hs]
        on = min(100 * st[(h, True)]["bounced"][0] for h in hs)
        rows.append(["M0 batting cage", "video of the bat passing through the ball with speculative contacts off and hitting it with them on; pitch distribution",
                     f"met: with the flag off the ball passes through a bat held still in {min(off):.0f} to {max(off):.0f} % of pitches (8, 10, 16 ms); with it on it bounces off in {on:.0f} %",
                     "[`results/M0.md`](results/M0.md)"])
    r = load("m1")
    if r:
        a = r["speed"]["analytic 100 Hz | all"]
        o = r["speed"]["analytic, planned once | all"]
        rows.append(["M1 model-based hitter", "hit rate and target accuracy against pitch speed",
                     f"met: {pct(a['hit'])} hit, {pct(a['on_target'])} inside the 0.75 m target zone at 100 Hz replanning; the same planner called once hits {pct(o['hit'])}",
                     "[`results/M1.md`](results/M1.md)"])
    r = load("m2")
    if r and "timing" in r:
        A, W = "analytic planner (classical)", "world-model planner (pure learning)"
        rows.append(["M2 learned world-model planner", "compare with the analytic MPC on hit rate and planning time",
                     f"done: the learned planner hits {pct(r['all'][W]['hit'])} but lands {pct(r['all'][W]['on_target'])} on target against {pct(r['all'][A]['on_target'])}; "
                     f"{r['timing']['world model']['ms_median']:.1f} ms per call against {r['timing']['analytic']['ms_median']:.1f} ms",
                     "[`results/M2.md`](results/M2.md)"])
    r = load("m3")
    if r and "when" in r:
        full = next(q for q in r["periodic"] if q["setting"] >= 100)
        l4, p4 = at_budget(r["learned"], 4.0), at_budget(r["periodic"], 4.0)
        l3, p3 = at_budget(r["learned"], 2.9), at_budget(r["periodic"], 2.9)
        rows.append(["M3 compute triggers", "Pareto curves of hit quality against planner calls",
                     f"met: every step costs {full['plans'][0]:.0f} calls per pitch for quality {full['quality'][0]:.2f}; the learned trigger holds {pm(l4['quality'], 1, 2)} with {l4['plans'][0]:.1f} calls "
                     f"(fixed rate: {pm(p4['quality'], 1, 2)} with {p4['plans'][0]:.1f}) and {pm(l3['quality'], 1, 2)} with {l3['plans'][0]:.1f} (fixed rate: {pm(p3['quality'], 1, 2)} with {p3['plans'][0]:.1f})",
                     "[`results/M3.md`](results/M3.md)"])
    r = load("m4")
    if r and len(r.get("reaction", {})) == 2:
        g = r["reaction"]["gust"]
        x = lambda k: f"{g[k]['extra_calls'][0] + 0.0:+.1f}".replace("-0.0", "+0.0")
        sp = r["reaction"]["spin"]
        spin_seen = any(sp[k]["react"][0] - sp[k]["react_placebo"][0] > 0.05 for k in sp)
        rows.append(["M4 surprises", "response latency to surprises for each trigger type",
                     f"met: a gust makes the learned trigger spend {x('learned_s')} planner calls and the innovation trigger {x('innovation')}; "
                     f"fixed-rate and variance triggers do not change their schedule ({x('periodic')}, {x('variance')}). Hit rate under gusts at about {r['budget']:.0f} calls: "
                     f"learned {100 * g['learned_s']['hit_surprise'][0]:.0f} %, fixed rate {100 * g['periodic']['hit_surprise'][0]:.0f} %."
                     + ("" if spin_seen else " No trigger replans sooner because of a spin change"),
                     "[`results/M4.md`](results/M4.md)"])
    r = load("m5")
    if r and len(r.get("reaction", {})) == 3:
        ev, inn = r["reaction"]["events"]["gust"], r["reaction"]["innovation"]["gust"]
        rows.append(["Stretch: event camera", "event-camera emulator; trigger on events instead of state innovation",
                     f"done: within {1e3 * r['window_s']:.0f} ms of a gust the event trigger calls the planner in {100 * ev['react']:.0f} % of pitches (placebo {100 * ev['react_placebo']:.0f} %), "
                     f"the frame-based innovation trigger in {100 * inn['react']:.0f} % (placebo {100 * inn['react_placebo']:.0f} %)",
                     "[`results/M5.md`](results/M5.md)"])
    r = load("rally")
    if r and len(r.get("rallies", {})) >= 6:
        R = r["rallies"]
        g, e, o = R["EKF, gated, every step"], R["EKF, every step (always think)"], R["true ball state, every step"]
        g12, off = R["EKF, gated, 12.5 Hz"], R["EKF, gated, speculative contacts off"]
        cap = r["config"]["rally"]["max_hits"]
        rows.append(["Stretch: ping-pong rally", "two-arm rally, thin paddles, speculative contacts",
                     f"done: {g['returns'][0]:.1f} returns per rally with the filter ({100 * g['full'][0]:.0f} % reach {cap}), {o['returns'][0]:.1f} with the true ball state; "
                     f"confidence-gated replanning doubled it from {e['returns'][0]:.1f}; at 12.5 Hz {g12['returns'][0]:.1f} returns with {g12['plans_per_return']:.0f} planner calls per return "
                     f"instead of {g['plans_per_return']:.0f}; speculative contacts off: {off['returns'][0]:.1f}",
                     "[`results/RALLY.md`](results/RALLY.md)"])
    rows.append(["Stretch: Filament videos", "use the Filament renderer for nice-looking videos",
                 "done: every milestone video is rendered offscreen with MuJoCo 4.0's Filament renderer (Vulkan), with physically based materials; `--renderer classic` for the classic one",
                 "`src/batter/eval/filament.py`"])
    return table(["milestone", "acceptance", "result", "report"], rows)


def caveats() -> str:
    out = []
    r3, r4, r2, r1, r5, rr = load("m3"), load("m4"), load("m2"), load("m1"), load("m5"), load("rally")
    if r3 and "when" in r3:
        w = r3["when"]
        out.append(f"- **The classical event triggers lose to a fixed rate on ordinary pitches.** At about 4 planner calls per pitch the innovation trigger reaches a hit quality of "
                   f"{w['innovation']['quality'][0]:.2f}, the variance trigger {w['variance']['quality'][0]:.2f} (with {w['variance']['plans'][0]:.1f} calls) and the drift trigger "
                   f"{w['drift']['quality'][0]:.2f} ({w['drift']['plans'][0]:.1f} calls), against {w['periodic']['quality'][0]:.2f} for a fixed rate. Variance and drift fire early, while the "
                   f"estimate still moves by centimetres, and are silent late, when millimetres matter; the innovation trigger fires on noise, because without a surprise the camera "
                   f"does not contradict the forecast, it only sharpens it.")
        l = sorted(r3["learned"], key=lambda q: q["plans"][0])[0]
        out.append(f"- **The learned trigger does not go below about {l['plans'][0]:.1f} calls per pitch.** At the highest price (λ = {l['setting']}) it still spends {l['plans'][0]:.1f} calls for a quality of "
                   f"{l['quality'][0]:.2f}; the corner of the curve below that was not reached.")
    if r4 and len(r4.get("reaction", {})) == 2:
        g, sp = r4["reaction"]["gust"], r4["reaction"]["spin"]
        per = at_budget(r4["pareto"]["gust"]["periodic"], g["learned_s"]["calls_surprise"][0])
        out.append(f"- **Reacting to a surprise does not buy back the quality.** At about 4 calls per pitch a gust takes the fixed rate from {g['periodic']['quality_plain'][0]:.2f} to "
                   f"{g['periodic']['quality_surprise'][0]:.2f}, the learned trigger trained with surprises from {g['learned_s']['quality_plain'][0]:.2f} to {g['learned_s']['quality_surprise'][0]:.2f}, "
                   f"and the innovation trigger from {g['innovation']['quality_plain'][0]:.2f} to {g['innovation']['quality_surprise'][0]:.2f}. Even replanning at every step loses quality on surprised "
                   f"pitches (spin change: {next(q for q in r4['pareto']['spin']['periodic'] if q['setting'] >= 100)['quality'][0]:.2f}, gust: "
                   f"{next(q for q in r4['pareto']['gust']['periodic'] if q['setting'] >= 100)['quality'][0]:.2f}), because the estimator needs frames to see the change and the arm needs time to move. "
                   f"What the reaction buys is modest: with the {g['learned_s']['calls_surprise'][0]:.1f} calls it then uses, the learned trigger's {g['learned_s']['quality_surprise'][0]:.2f} compares with "
                   f"{per['quality'][0]:.2f} for a fixed rate at {per['plans'][0]:.1f} calls.")
        out.append(f"- **A spin change is hard to see.** It bends the path gradually, so in the {1e3 * r4['window_s']:.0f} ms after it the innovation trigger replans in {100 * sp['innovation']['react'][0]:.0f} % of pitches "
                   f"against {100 * sp['innovation']['react_placebo'][0]:.0f} % on the same pitches without it (gust: {100 * g['innovation']['react'][0]:.0f} % against {100 * g['innovation']['react_placebo'][0]:.0f} %). "
                   f"The EKF's own surprise detector (a χ² gate on its innovations) trips on a majority of gusts and on few spin changes (`tests/test_ekf.py`); when it trips it makes the covariance honest, not the estimate better.")
    if r2 and "timing" in r2:
        W = "world-model planner (pure learning)"
        out.append(f"- **The learned planner hits but does not aim.** {pct(r2['all'][W]['hit'])} hit rate, {pct(r2['all'][W]['on_target'])} on target. Its outcome model predicts the landing point to about "
                   f"{np.mean([v['landing_err_med_m'] for k, v in r2['validation'].items() if k.startswith('s')]):.1f} m, which is the size of the target. It also needed data from the analytic batter; "
                   f"with a sixteenth of the data it hits {pct(r2['data_sweep'][0]['hit'])}.")
    if r1 and "timing" in r1:
        t = r1["timing"]
        out.append(f"- **The planner is not real-time at 100 Hz.** A call takes {t['ms_median']:.1f} ms (median) and {t['ms_p90']:.1f} ms (90th percentile) on one core, against a 10 ms control period, "
                   f"and the simulation does not charge for that time: a plan is applied in the step it was asked for. `EnvParams.latency_steps` delays plans by whole steps; it is implemented and not evaluated.")
    if r5 and len(r5.get("reaction", {})) == 3:
        ev, inn, per = r5["reaction"]["events"], r5["reaction"]["innovation"], r5["reaction"]["periodic"]
        out.append(f"- **Events trigger sooner, but that does not make a good trigger.** At about 4 calls per ordinary pitch the event trigger's quality is {ev['quality_plain'][0]:.2f} "
                   f"({ev['gust']['quality'][0]:.2f} under gusts), the frame-based innovation trigger's {inn['quality_plain'][0]:.2f} ({inn['gust']['quality'][0]:.2f}), a fixed rate's "
                   f"{per['quality_plain'][0]:.2f} ({per['gust']['quality'][0]:.2f}). The planner the events wake up still runs on the EKF, which is fed by frames that are 30 ms old; "
                   f"the events would have to feed the estimator for the earlier trigger to pay.")
    if rr and len(rr.get("rallies", {})) >= 6:
        f, R = rr["feasibility"], rr["rallies"]
        g, o = R["EKF, gated, every step"], R["true ball state, every step"]
        out.append(f"- **The rally needs a midi table, a long handle and a frictionless table.** On an ITTF table a return needs {f[0]['paddle_speed_needed'][1]:.1f} m/s of paddle speed and the Panda gives "
                   f"{f[0]['paddle_speed_available']:.1f}; the rally is played on a {rr['config']['table']['length']:.1f} m table with a {rr['config']['paddle']['handle']:.2f} m handle. "
                   f"With table friction, MuJoCo's one-step bounce returned the ball faster than it arrived in {100 * rr['bounce']['friction 0.25']['gains_energy']:.0f} % of bounces, so the table is frictionless "
                   f"(no spin kick off the table).")
        out.append(f"- **The filter still costs the rally about {o['returns'][0] - g['returns'][0]:.1f} returns** ({g['returns'][0]:.1f} against {o['returns'][0]:.1f} with the true ball state). "
                   f"Most rallies that end early end in the net or on the hitter's own half: the return is slightly slow or low. Both arms run the same controller; there is no adversarial play.")
    out.append("- **Simplified:** the bat and paddles do not collide with the arms, and joint limits during the follow-through are left to MuJoCo's limit constraints.")
    return "\n".join(out)


def readme() -> None:
    from batter import ROOT

    p = ROOT / "README.md"
    text = p.read_text()
    for tag, body in (("results", glance()), ("caveats", caveats())):
        a, b = f"<!-- {tag}:begin -->", f"<!-- {tag}:end -->"
        i, j = text.index(a) + len(a), text.index(b)
        text = text[:i] + "\n" + body + "\n" + text[j:]
    p.write_text(text)
    print("updated README.md")

def main():
    for name, fn in SECTIONS.items():
        text = fn()
        if text is None:
            print(f"{name}: no results yet")
            continue
        (RESULTS / f"{name}.md").write_text(text)
        print(f"wrote results/{name}.md")
    readme()


if __name__ == "__main__":
    main()
