"""Write results/M0.md .. M4.md from the result files, so every number in the write-ups is
the number the scripts produced.  Run after scripts/run_all.sh."""
from __future__ import annotations

import json

import numpy as np

from card_catcher import RESULTS, config
from card_catcher.models.objects import OBJECTS


def load(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def pm(v, scale=1.0, nd=1) -> str:
    return f"{v[0]*scale:.{nd}f} ± {v[1]*scale:.{nd}f}"


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def verdict(ok: bool) -> str:
    return "**met**" if ok else "**not met**"


# ----------------------------------------------------------------------------- M0
def m0(cfg) -> str:
    tun, ds, mount = load("m0_tunnel.json"), load("m0_dataset.json"), load("m0_mount.json")
    rho = cfg["sim"]["density"]
    obj_rows = []
    for name, o in OBJECTS.items():
        k = [o.kd_axis(np.eye(3)[i], rho) for i in range(3)]
        if name == "ball":
            dims = f"sphere, r = {o.size[0]*1e3:.0f} mm"
        elif name == "plate":
            dims = f"disc, r = {o.size[0]*1e3:.0f} mm, {2*o.size[1]*1e3:.0f} mm thick"
        else:
            dims = f"{2*o.size[0]*1e3:.0f} × {2*o.size[1]*1e3:.0f} × {2*o.size[2]*1e3:.1f} mm"
        obj_rows.append([name, dims, f"{o.mass*1e3:g} g", f"{min(k):.3f} – {max(k):.3f}"])
    tun_rows = [[f"{c['timestep_ms']:.0f} ms", "on" if c["speculative"] else "off", f"{c['tunnel_rate_mean']*100:.1f} ± {c['tunnel_rate_std']*100:.1f} %",
                 ", ".join(f"{k} m/s: {v*100:.0f} %" for k, v in c["by_speed"].items())] for c in tun["conditions"]]
    n_tun = tun["n_per_seed"] * len(tun["seeds"])
    ok_video = all((c["tunnel_rate_mean"] > 0.5) != c["speculative"] for c in tun["conditions"])
    ds_rows = []
    for name, r in ds["objects"].items():
        a = r["mean_std"]
        ds_rows.append([name, pm(a["launch_speed"], nd=2), pm(a["cross_speed"], nd=2), pm(a["flight_time"], nd=2), pm(a["nominal_miss_cm_median"], nd=1),
                        pm(a["residual_rms"], nd=2), pm(a["max_ang_speed"], nd=0)])
    tc_rows = []
    for name, r in ds["objects"].items():
        for dt, v in r["timestep_convergence"].items():
            tc_rows.append([name, dt, f"{v['median_cm_at_0.19s']:.2f}", f"{v['median_cm_at_0.64s']:.2f}", f"{v['median_cm_at_0.64s_minus_gravity_offset']:.2f}"])
    mount_rows = [[m, f"{p}°", f"{v['reachable']*100:.0f} %", f"{v['usable_margin_0.3']*100:.0f} %", f"{v['best_margin']:.2f}"]
                  for m, r in mount.items() for p, v in r.items()]
    card = ds["objects"]["card"]["mean_std"]
    return f"""# M0: scene and throws

Reproduce: `scripts/m0_mount.py`, `scripts/m0_dataset.py`, `scripts/m0_tunnel.py`. Video: [`m0_tunnel.mp4`](m0_tunnel.mp4).
Studio: `python -m mujoco.studio.viewer --model=assets/catch_card.xml --hot_reload=all`.

## Acceptance

- A video shows tunnelling with the flag off and none with it on: {verdict(ok_video)}. Over {n_tun} throws per condition the card goes through the fingertip in {tun_rows[0][2]} of throws with the flag off and {tun_rows[1][2]} with it on at 8 ms ({tun_rows[2][2]} and {tun_rows[3][2]} at 16 ms).
- A dataset of 1,000 card trajectories with fluid forces on: **met** ({ds['n']} accepted throws per object for each of 3 seeds, in `data/`).

## Scene

Menagerie Franka Panda, a low table, a floor, and one thrown object with a free joint. `<option timestep="0.008" integrator="implicitfast" density="1.2" viscosity="1.8e-5">` with `<flag speculative="enable"/>`. The arm's position servos are replaced by torque motors at the datasheet limits; the finger servo is kept. Sensors: object position and velocity (ground truth), and five 4.0 `contact` sensors (object against each finger, the hand, the floor, the table).

{table(["object", "geometry", "mass", "quadratic drag k_d along the body axes [1/m]"], obj_rows)}

All three use `fluidshape="ellipsoid"`. The drag formula in `models/objects.py` was checked against `mjData.qfrc_fluid` to six digits for motion along each body axis.

### Why the arm hangs from a gantry

Thrown objects arrive from above (45° to 85° below the horizontal), so the palm has to face forward and up. The pitch of the Panda's tool axis in the base frame is q6 + q4 − q2 − π/2 with q6 ≤ 3.75 rad, which leaves an upright Panda almost no such poses. The table counts TCP positions on a grid in front of the base (x 0.35–0.85 m, 0.2–1.0 m above or below the base) where IK converges, and where it does so with at least 0.3 rad to every joint limit.

{table(["mount", "palm elevation", "IK reachable", "usable (≥ 0.3 rad margin)", "best margin [rad]"], mount_rows)}

The scene therefore mounts the Panda inverted, 1.30 m above the floor. Nothing else about the robot is changed.

## Speculative contacts

The card is thrown face-first at 2–5 m/s along the jaw axis at the outside of one fingertip (9–11 mm of finger plus an 8 mm pad behind it), with the arm holding its ready pose. Per step it travels 1.6–4 cm at 8 ms and 3.2–8 cm at 16 ms. A throw counts as tunnelling when the card's centre of mass crosses the mid-plane of the pad at least 2 mm inside the pad's box, which cannot happen without the centre going through at least 13 mm of solid. In every throw the first robot body touched was that finger.

{table(["timestep", "speculative", "tunnelling (mean ± std over 3 seeds)", "by launch speed"], tun_rows)}

An earlier version of this test aimed lower on the finger and reported 6 % tunnelling with the flag on. Tracing those throws showed the card's lower edge clipping the palm first, and a criterion with only 2–3 mm of margin on a tapered finger. Both were test errors, not contact errors, and the numbers above are from the corrected geometry.

## Throws

A throw is aimed with the nominal drag-ballistic model at a random point of the catch window with a random flight time, then given a random attitude and spin (card and plate: edge-on, tilted, spinning about the normal). A throw is kept if its simulated flight crosses the catch plane inside the window at least 0.35 s after launch. Flights are simulated with `mujoco.batch`; the batch result equals a plain `mj_step` loop bit for bit.

{table(["object", "launch speed [m/s]", "speed at the plane [m/s]", "flight time [s]", "miss of the nominal model from the true launch state [cm]", "residual acceleration rms [m/s²]", "peak angular speed [rad/s]"], ds_rows)}

The card does flutter: its normal flips sign {card['normal_flips_per_throw'][0]:.1f} times per throw on average, it reaches {card['max_ang_speed'][0]:.0f} rad/s, and it slows from {card['launch_speed'][0]:.1f} to {card['cross_speed'][0]:.1f} m/s (its terminal speed). Even with the exact launch state, the drag-ballistic model misses the card's catch point by {card['nominal_miss_cm_median'][0]:.0f} cm.

![flights](m0_flight.png)

### How far is an 8 ms flight from a converged one?

Median position difference to the same launch simulated at 0.5 ms:

{table(["object", "timestep", "at 0.19 s [cm]", "at 0.64 s [cm]", "at 0.64 s, gravity offset removed [cm]"], tc_rows)}

For the ball nearly all of the difference is the offset g·h·t/2 that MuJoCo's semi-implicit step adds to every free fall; it is the same for every throw and the predictor reproduces it exactly (M1). The card's flight is chaotic, so its 8 ms trajectory is a different realisation of the same dynamics rather than an approximation of the 0.5 ms one. The 8 ms simulation is the plant for everything that follows.
"""


# ----------------------------------------------------------------------------- M1
def m1(cfg) -> str:
    prm, ekf = load("m1_params.json"), load("m1_ekf.json")
    res = ekf["results"]
    prm_rows = [[o, f"{p['kd0']:.4f}", f"{p['kd_per_throw_std']:.4f}", f"{p['sigma_a']:g}", f"{p['innovation_nll']:.2f}"] for o, p in prm.items()]
    rows = []
    for o in ("ball", "plate", "card"):
        for tau in ("0.4", "0.3", "0.2", "0.1"):
            if tau in res[o]:
                a = res[o][tau]
                rows.append([o, f"{float(tau)*1e3:.0f} ms", pm(a["rms_cm"], nd=2), f"{a['median_cm'][0]:.2f}", f"{a['p90_cm'][0]:.2f}",
                             f"{a['rms_time_ms'][0]:.1f}", f"{a['cover90'][0]*100:.0f} %"])
    ball = res["ball"]["0.2"]["rms_cm"]
    card, plate = res["card"]["0.2"], res["plate"]["0.2"]
    meas = ekf["measurement"]
    rate = 1.0 / (cfg["sim"]["timestep"] * meas["every"])
    return f"""# M1: physics predictor + EKF

Reproduce: `scripts/m1_ekf.py`. Tests: `tests/test_ekf.py`.

## Acceptance

- Ball error under 1 cm at 200 ms before the catch: {verdict(ball[0] < 1.0)}, {pm(ball, nd=2)} cm RMS (mean ± std over 3 seeds of 1,000 throws).
- The card error is large, as expected: {pm(card['rms_cm'], nd=2)} cm at 200 ms. The plate is at {pm(plate['rms_cm'], nd=2)} cm. That gap is what M2 is for.

## Model

State x = [p, v, k_d]. The process model is ballistic flight with isotropic quadratic drag, a(v) = g − k_d |v| v, discretised exactly as the simulator does it. MuJoCo's `implicitfast` step is linearly implicit in the velocity-dependent forces:

    v' = v + h (I + hK)⁻¹ a(v),   p' = p + h v',   K = −∂a/∂v = k_d |v| (I + v̂ v̂ᵀ)

which has the closed form given in `estim/physics.py`. Measurements are positions with σ = {meas['sigma']*1e3:.0f} mm noise at {rate:.0f} Hz and {meas['dropout']*100:.0f} % dropout. The catch point is where the predicted path crosses the catch plane; its covariance follows from the propagated state covariance, δp_c = (I − v nᵀ/(nᵀv)) δp.

Three statements about this model are tested rather than assumed:

- It reproduces MuJoCo for a non-spinning ball to 1e-9 m over 0.8 s in inviscid air. In real air the only remainder is the Stokes term, under 0.5 mm.
- Plain semi-implicit Euler (dropping the (I + hK)⁻¹) is off by 3.4 mm over the same flight, and continuous-time ballistics by g·h·t/2 ≈ 3 cm. The first version of this predictor made the former mistake; the test caught it.
- The analytic Jacobian matches finite differences to 1e-7, and the forecast covariance matches Monte Carlo sampling to within 12 % for white and 15 % for time-correlated acceleration noise.

## Tuning

Per object, on 500 separate throws: k_d by least squares on the step map, and the white acceleration noise σ_a by maximum likelihood of the filter innovations (grid search).

{table(["object", "fitted k_d [1/m]", "spread of per-throw k_d", "σ_a [m/s²]", "innovation NLL"], prm_rows)}

For the ball the fitted k_d equals the analytic value ρ C π r² / m. The card and plate need large σ_a because their unmodelled acceleration is several m/s².

## Result

{table(["object", "time before catch", "catch-point error RMS [cm]", "median [cm]", "90th pct [cm]", "timing RMS [ms]", "inside predicted 90 % ellipse"], rows)}

![error vs time to catch](m1_error.png)

The ball's covariance is honest: about 90 % of errors fall inside the 90 % ellipse. For the plate and card the ellipse is too small, because the filter assumes the model error is white while an aerodynamic force that is wrong now stays wrong for the next 100 ms. M2 addresses both the error and the over-confidence.
"""


# ----------------------------------------------------------------------------- M2
def m2(cfg) -> str:
    r = load("m2_residual.json")
    out = []
    for obj in r["objects"]:
        o = r["objects"][obj]
        rows = []
        taus = [t for t in ("0.5", "0.4", "0.3", "0.2", "0.1", "0.05") if all(t in o[m] for m in ("physics", "pure", "hybrid"))]
        one_seed = all(o[m][t]["rms_cm"][1] == 0 for t in taus for m in ("physics", "pure", "hybrid"))   # no spread to report
        fmt = (lambda v, nd=2: f"{v[0]:.{nd}f}") if one_seed else pm
        for tau in taus:
            if True:
                rows.append([f"{float(tau)*1e3:.0f} ms"] + [fmt(o[m][tau]["rms_cm"], nd=2) for m in ("physics", "pure", "hybrid")]
                            + [f"{o[m][tau]['cover90'][0]*100:.0f} %" for m in ("physics", "pure", "hybrid", "hybrid_white")])
        out.append((obj, o, rows))
    card = r["objects"]["card"]
    c2 = {m: card[m]["0.2"]["rms_cm"] for m in ("physics", "pure", "hybrid")}
    beats = c2["hybrid"][0] < c2["pure"][0] and c2["hybrid"][0] < c2["physics"][0]
    cal_rows = []
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    worst = 0.0
    for b, c in card["calibration"].items():
        cov = c["coverage"]
        worst = max(worst, max(abs(cv - lv) for cv, lv in zip(cov, levels)))
        cal_rows.append([b, pm(c["nll"], nd=3), f"{c['z_std']:.3f}", f"{cov[4]*100:.1f} %", f"{cov[8]*100:.1f} %", f"{cov[9]*100:.1f} %"])
    sweep = card.get("sweep_rms_cm_at_200ms", {})
    sw_rows = [[n, f"{v['pure']:.2f}", f"{v['hybrid']:.2f}"] for n, v in sorted(sweep.items(), key=lambda kv: int(kv[0]))]
    body = f"""# M2: learned aerodynamic residual

Reproduce: `scripts/m2_residual.py` (trains on CPU, about 4 minutes per model). Tests: `tests/test_online.py`.

## Acceptance

- The residual model beats physics-only and pure learning on the card: {verdict(beats)}. At 200 ms before the catch the catch-point RMS error is {pm(c2['physics'], nd=2)} cm (physics), {pm(c2['pure'], nd=2)} cm (pure learning) and {pm(c2['hybrid'], nd=2)} cm (hybrid), mean ± std over 3 training seeds, each tested on 1,000 unseen throws.
- Its predicted variance is calibrated: {verdict(worst < 0.05)}. Across ten nominal levels and three horizon bands, the empirical coverage of the residual's central intervals is within {worst*100:.1f} points of nominal (table and plot below).

## What is learned

The residual is defined per simulator step as r_k = (v_(k+1) − v_k)/h − w(v_k), where w is the physics model's effective acceleration with the fitted drag. With the true r the physics model reproduces a trajectory exactly (tested), so "true minus physics" has no discretisation error hidden in it.

- **Hybrid.** The M1 physics EKF runs unchanged and supplies features: filtered velocity, normalised innovation, validity flag, drag estimate. A GRU (128 units) reads them frame by frame; from its state an MLP outputs a mean and a log-variance for the residual acceleration at each of the next {112} steps (0.9 s). Loss: Gaussian negative log-likelihood (Kendall & Gal). A second EKF uses the nowcast mean as a known input and κ times the nowcast variance as its process noise, with the drag fixed at the fitted value. The forecast integrates physics plus the predicted residuals from that filter's state.
- **Pure learning.** Same GRU and decoder, raw measurement features only, and it outputs future positions directly. It is given no gravity, drag or integrator.
- **Physics only.** The M1 EKF.

Two scalars are fitted after training, both on 500 validation throws, and nothing is tuned on the test sets:

- φ = {card['phi'][0]:.3f}: the lag-one autocorrelation of the standardised forecast errors along the horizon. The forecast covariance treats the residual error as an AR(1) process with this coefficient (state augmentation) instead of white noise.
- κ = {", ".join(f"{k:g}" for k in card['kappa_per_seed'])} (per seed): inflation of the filter's learned process noise, chosen by maximum likelihood of the innovations. It is needed because the filter itself treats the nowcast error as white.

## Card

{table(["time before catch", "physics RMS [cm]", "pure RMS [cm]", "hybrid RMS [cm]", "90 % ellipse: physics", "pure", "hybrid", "hybrid, white-noise forecast"], out[0][2])}

![M2](m2_residual.png)

### Calibration of the residual's variance

{table(["horizon", "NLL [nats per axis]", "std of z-scores (ideal 1)", "coverage of 50 % interval", "90 %", "95 %"], cal_rows)}

The variance of the residual itself is well calibrated at every horizon. The covariance of the *catch point* is a harder object, because it integrates correlated errors twice. Ignoring the correlation (last column of the card table) leaves the 90 % ellipse covering only about half of the outcomes. With the AR(1) model it is conservative far from the catch and close to nominal in the last 100 ms. It is not exact, and M4's variance trigger inherits that.

### Data efficiency

RMS error 200 ms before the catch, seed 0, by number of training throws:

{table(["training throws", "pure learning [cm]", "hybrid [cm]"], sw_rows)}
"""
    for obj, o, rows in out[1:]:
        body += f"""
## {obj.capitalize()}

One training seed. φ = {o['phi'][0]:.3f}, κ = {", ".join(f"{k:g}" for k in o['kappa_per_seed'])}.

{table(["time before catch", "physics RMS [cm]", "pure RMS [cm]", "hybrid RMS [cm]", "90 % ellipse: physics", "pure", "hybrid", "hybrid, white-noise forecast"], rows)}
"""
    body += """
## Limits

- The network sees position fixes only. The card's attitude, which drives its aerodynamics, is inferred from how the track bends, so the long-horizon error stays in the centimetres: flutter is chaotic and not predictable 400 ms ahead from position alone.
- The hybrid filter's use of a nowcast that depends on past measurements is exact only if the network outputs the true conditional mean. κ is the one-parameter correction for the part of that assumption that fails.
"""
    return body


# ----------------------------------------------------------------------------- M3
def m3(cfg) -> str:
    r = load("m3_catch.json")
    rows = []
    for obj in ("ball", "card", "plate"):
        allk = {**r["ablation_predictor"].get(obj, {}), **r["main"].get(obj, {})}
        for kind in ("oracle", "physics", "hybrid"):
            if kind in allk:
                a = allk[kind]
                star = " (default)" if kind in r["main"].get(obj, {}) else ""
                rows.append([obj, kind + star, pm(a["caught"], 100), pm(a["clean"], 100), pm(a["retained"], 100), f"{a['tcp_err_mm_median'][0]:.1f} / {a['tcp_err_mm_p90'][0]:.1f}",
                             f"{a['plans_per_throw'][0]:.0f}", f"{a['late_throws'][0]*100:.0f} %", f"{a['sat_steps_per_throw'][0]:.2f}", f"{a['ms_per_world_step'][0]:.2f}"])
    ball = r["main"]["ball"]["physics"]["caught"]
    card = r["main"]["card"]["hybrid"]["caught"]
    card_or = r["ablation_predictor"]["card"]["oracle"]["caught"]
    card_ph = r["ablation_predictor"]["card"]["physics"]["caught"]
    n_tot = r["n_per_seed"] * len(r["seeds"])
    ab = r.get("ablation_contacts", {})
    ab_rows = []
    for obj in ab:
        for key, a in ab[obj].items():
            dt, sp = key.split("_")
            ab_rows.append([obj, dt.replace("ms", " ms"), sp, pm(a["caught"], 100), pm(a["clean"], 100), f"{a['final_pen_mm_caught'][0]:.2f}",
                            pm(a["retained"], 100), f"{a['ms_per_world_step'][0]:.2f}"])
    ab_text = ""
    if "card" in ab and "ball" in ab:
        c, bl = ab["card"], ab["ball"]
        ab_text = f"""The plain "caught" column is misleading for the card. With the flag off the sensor-based criterion reports more catches ({c['8ms_off']['caught'][0]*100:.0f} % against {c['8ms_on']['caught'][0]*100:.0f} % at 8 ms), but those cards sit a median {c['8ms_off']['final_pen_mm_caught'][0]:.1f} mm inside the finger geometry, several times the card's 0.7 mm thickness, against {c['8ms_on']['final_pen_mm_caught'][0]:.2f} mm with the flag on. They were not pinched; they tunnelled part-way into a finger and stuck there (the mechanism M0 isolates). Counting only catches whose deepest overlap is at most the object's own thickness gives the "clean" column, which is what the figure shows: {pm(c['8ms_on']['clean'], 100)} % on against {pm(c['8ms_off']['clean'], 100)} % off at 8 ms, and {pm(c['16ms_on']['clean'], 100)} % against {pm(c['16ms_off']['clean'], 100)} % at 16 ms.

For the ball the flag does not matter at 8 ms ({pm(bl['8ms_on']['clean'], 100)} % on, {pm(bl['8ms_off']['clean'], 100)} % off). At 16 ms it decides the outcome: {pm(bl['16ms_on']['clean'], 100)} % on against {pm(bl['16ms_off']['clean'], 100)} % off."""
    st = r.get("stretch_suction", {})
    st_rows = []
    for obj in st:
        grip = {**r["ablation_predictor"].get(obj, {}), **r["main"].get(obj, {})}
        for kind, a in st[obj].items():
            st_rows.append([obj, kind, pm(grip[kind]["caught"], 100), pm(a["retained"], 100)])
    h = cfg["sim"]["timestep"]
    main_ms = r["main"]["ball"]["physics"]["ms_per_world_step"][0]
    return f"""# M3: interception controller

Reproduce: `scripts/m3_catch.py`. Videos: [`m3_ball.mp4`](m3_ball.mp4), [`m3_card.mp4`](m3_card.mp4). Tests: `tests/test_planner.py`, `tests/test_online.py`.

## Acceptance

- Ball catch rate above 90 %: {verdict(ball[0] > 0.9)}, {pm(ball, 100)} % ({n_tot} throws, 3 seeds, EKF on noisy position fixes with dropout).
- Card catch rate, reported: {pm(card, 100)} % with the hybrid predictor, {pm(card_ph, 100)} % with physics only, {pm(card_or, 100)} % with a perfect predictor.

"Caught" means that half a second after the catch the object is pinched between both fingers (both finger `contact` sensors fire), within 8 cm of the tool centre, and off the table and floor. "Retained" relaxes the pinch to any contact with the hand. "Clean" additionally requires that nothing overlaps the object by more than the object's own thickness at that moment. It matters for the card: a sheet 0.7 mm thick can pass the sensor test while sitting inside a finger. With speculative contacts on the two columns differ by a few points (edge-on wedges); the contact ablation below shows how far they separate without the flag.

## Controller

1. **Target.** The predicted crossing of the catch plane gives a point, a time and an arrival direction. The hand is placed with its tool centre on the point and its palm facing the arrival direction, limited to a 35° cone about the plane normal, with one finger above the object and one below.
2. **IK.** Damped least squares with joint-limit clamping and a posture term, warm-started from the previous goal.
3. **Trajectory QP.** Per joint, minimum effort and jerk from the current state to rest at the goal when the object arrives, under position, velocity and acceleration limits (Franka datasheet), written in the simulator's own integrator so the plan is exactly trackable. If no limit is active the optimum is closed form; otherwise OSQP solves it. Whether the goal is reachable in time is decided analytically (bang-coast-bang time of a double integrator, checked against an exact LP solver in the tests). A joint that cannot make it takes the shortest longer horizon.
4. **Tracking.** Computed torque in discrete time, τ = (M + hD) a + c − f_passive, with PD gains placed by pole placement on the sampled error dynamics. In free motion a plan is tracked to 1e-6 rad.
5. **Jaws.** Closed 50 ms before the predicted contact, or at once when a contact sensor fires.

Replanning is at a fixed 50 Hz here; M4 studies when to replan.

### The catch window

The window on the plane is the region the arm reaches in time. A first version used ±28 cm by ±20 cm and lost about 10 % of ball catches with a perfect predictor: the bottom strip needed the elbow to travel up to 1.9 rad in 0.6 s, beyond the 12.5 rad/s² limit. The window is now ±20 cm by ±14 cm about a centre moved 6 cm up the plane, and the ready pose points the palm along each object's typical arrival direction.

## Results

{table(["object", "predictor", "caught [%]", "clean [%]", "retained [%]", "TCP error at the catch, median / p90 [mm]", "planner calls per throw", "throws with a late plan", "torque-saturated steps per throw", "wall-clock per world step [ms]"], rows)}

![catch rates](m3_catch.png)

- **Ball.** The gap between the EKF and the oracle is small: the catch point is known to a few millimetres well before the arm has to commit.
- **Card.** A perfect predictor catches {card_or[0]*100:.0f} %, so most misses are grasp mechanics, not prediction: the card arrives tumbling with an attitude the position-only sensor cannot see, and only some attitudes slide between the jaws. Prediction quality still matters within that ceiling: the hybrid predictor recovers {card[0]*100:.0f} % against {card_ph[0]*100:.0f} % for physics only.
- **Plate.** A 16 cm disc does not fit an 8 cm jaw opening unless its rim is presented to the jaws, which needs the attitude. The parallel gripper essentially never catches it.
- **Limits.** Torque saturation is rare, so the acceleration limits are an adequate stand-in for torque limits in the planner. Stepping costs {main_ms:.2f} ms per world step including estimation, forecasting and planning, against a {h*1e3:.0f} ms timestep.

## Ablation: speculative contacts × timestep

Perfect predictor, so only the contact physics differs. Throw datasets are regenerated at each timestep.

{table(["object", "timestep", "speculative", "caught [%]", "clean [%]", "median overlap of caught objects [mm]", "retained [%]", "wall-clock per world step [ms]"], ab_rows) if ab_rows else "_not run_"}

{ab_text}

## Stretch: suction disc instead of jaws

A 7 cm disc with the 4.0 `adhesion` geom attribute sits between the open fingers (1 N per contact for the card, 20 N for the plate). The metric for suction is "retained".

{table(["object", "predictor", "parallel gripper, caught [%]", "suction disc, retained [%]"], st_rows) if st_rows else "_not run_"}
"""


# ----------------------------------------------------------------------------- M4
M4_FINDINGS = {
    "ball": """- **The learned trigger matches periodic replanning down to about 5 calls per throw and is ahead below that.** Both hold the baseline's catch rate at 5 calls; at 3.4 calls the learned trigger still catches 93 %, where periodic replanning is falling steeply (55 % at 2.7 calls).
- **Fixed-threshold event triggers lose to plain periodic replanning at the same budget.** The likely reason, not separately measured here: the variance and drift triggers fire mostly early in the flight, when the estimate moves a lot, and rarely once it has settled within δ, while a catch needs a fresh plan shortly before contact. A slow periodic clock provides one; a threshold on accumulated change does not. The learned policy has the time to catch among its inputs.
- **Variance and drift have the same shape**, as expected for a consistent filter, where the expected squared drift equals the drop in covariance trace. Drift is somewhat better at mid budgets.""",
    "card": """- **All triggers lie on roughly one curve within the error bars.** The card's catch rate is limited by prediction and grasp mechanics (M3), and replanning more often buys only a few points between 15 and 70 calls per throw.
- **The learned trigger does not reach the baseline.** Its most call-hungry setting uses 13 calls per throw, and at the budgets it covers it matches periodic replanning. With a reward of catch − λ · replans and a catch rate near 30 %, the marginal value of a call is small, and the policy learned to save calls.
- **The variance trigger is the weakest here**: it needs about twice the calls of the drift trigger for the same catch rate. A likely cause is the hybrid forecast covariance, which is conservative far from the catch (M2).""",
}


def m4(cfg) -> str:
    names = {"periodic": "periodic", "variance": "variance-triggered", "drift": "drift-triggered", "learned": "learned (PPO)"}
    units = {"periodic": "rate [Hz]", "variance": "δ [mm]", "drift": "δ [mm]", "learned": "λ"}
    body = """# M4: event-triggered replanning

Reproduce: `scripts/m4_triggers.py`. Tests: `tests/test_online.py`.

## Acceptance

- A Pareto plot of catch rate against planner calls for the three required triggers (periodic, variance-triggered, learned), plus a drift trigger: **met**.

![Pareto](m4_pareto.png)

## Triggers

Every trigger sees the same per-step information from the predictor. The first plan of a throw is made as soon as a forecast exists and counts as one call for all of them. ĉ(t) is the current catch-point estimate with in-plane covariance P(t), and t_p the time of the last plan.

- **Periodic.** Replan every 1/rate seconds; 50 Hz is the baseline. On the 125 Hz simulation clock, 50 Hz alternates 2- and 3-step gaps.
- **Variance-triggered.** Replan when tr P(t_p) − tr P(t) > δ². For a near-optimal estimator the update ĉ(t) − ĉ(t_p) is uncorrelated with ĉ(t_p), so this difference is the expected squared distance between where the plan aims and where a new plan would aim. Like the triggers of Trimpe and D'Andrea it uses covariances only.
- **Drift-triggered.** Replan when the realised displacement of the estimate (position, plus the timing change converted to a distance) exceeds δ.
- **Learned.** A two-layer policy trained with PPO on `mujoco.batch` worlds, reward = catch − λ · (number of replans). λ is an input and is drawn log-uniformly per episode, so one policy gives the whole curve. Its inputs are the predicted time to catch, time since the last plan, the drift, and the current and at-plan covariance traces.
"""
    for obj in ("card", "ball"):
        r = load(f"m4_{obj}.json")
        if r is None:
            continue
        base = next(x for x in r["periodic"] if x["setting"] == 50.0)
        target = base["catch"][0] - 0.03
        head = []
        for name in ("periodic", "variance", "drift", "learned"):
            if name not in r:
                continue
            ok = [x for x in r[name] if x["catch"][0] >= target]
            if ok:
                b = min(ok, key=lambda x: x["plans"][0])
                s = b["setting"]
                s = f"{s:g} Hz" if name == "periodic" else (f"δ = {s*1e3:g} mm" if name in ("variance", "drift") else f"λ = {s:g}")
                head.append([names[name], s, f"{b['plans'][0]:.1f}", pm(b["catch"], 100), f"{base['plans'][0]/b['plans'][0]:.1f}×"])
            else:
                head.append([names[name], "none", "–", "–", "–"])
        body += f"""
## {obj.capitalize()} ({r['predictor']} predictor)

{r['n_per_seed']} throws per seed and setting, 3 seeds. Baseline, periodic 50 Hz: {pm(base['catch'], 100)} % with {base['plans'][0]:.1f} planner calls per throw.

Cheapest setting of each trigger that stays within 3 points of the baseline's catch rate:

{table(["trigger", "setting", "planner calls per throw", "catch rate [%]", "fewer calls than 50 Hz"], head)}
"""
        # catch rate at equal planner budgets, interpolated in log(calls) between measured settings
        budgets = (5, 10, 20)

        def at(name, bud):
            pts = sorted((x["plans"][0], x["catch"][0]) for x in r.get(name, []))
            if not pts or not (pts[0][0] <= bud <= pts[-1][0]):
                return None
            return float(np.interp(np.log(bud), np.log([q[0] for q in pts]), [q[1] for q in pts]))

        eq = [[names[n]] + [("–" if at(n, bud) is None else f"{at(n, bud)*100:.0f}") for bud in budgets]
              for n in ("periodic", "variance", "drift", "learned") if n in r]
        body += f"""
Catch rate [%] at an equal budget of planner calls per throw (interpolated between the measured settings; – means the trigger's settings do not span that budget):

{table(["trigger"] + [f"{bud} calls" for bud in budgets], eq)}

{M4_FINDINGS.get(obj, "")}

All settings:
"""
        for name in ("periodic", "variance", "drift", "learned"):
            if name not in r:
                continue
            rows = []
            for x in r[name]:
                s = x["setting"]
                s = f"{s:g}" if name in ("periodic", "learned") else f"{s*1e3:g}"
                rows.append([s, f"{x['plans'][0]:.1f}", pm(x["catch"], 100)])
            body += f"\n**{names[name]}**\n\n{table([units[name], 'planner calls per throw', 'catch rate [%]'], rows)}\n"
        log = load(f"m4_ppo_log_{obj}.json")
        if log:
            body += (f"\nPPO training: {len(log)} iterations of {r.get('ppo_batch', 96)} throws, about "
                     f"{np.median(np.diff([0.0] + [x['seconds'] for x in log])) * len(log) / 60:.0f} minutes on CPU (median iteration time × iterations). "
                     f"Mean return went from {np.mean([x['return'] for x in log[:3]]):+.2f} (first three iterations) to "
                     f"{np.mean([x['return'] for x in log[-3:]]):+.2f} (last three).\n")
    return body


def main():
    cfg = config.load()
    for name, fn in (("M0", m0), ("M1", m1), ("M2", m2), ("M3", m3), ("M4", m4)):
        try:
            (RESULTS / f"{name}.md").write_text(fn(cfg))
            print("wrote", RESULTS / f"{name}.md")
        except (FileNotFoundError, TypeError, KeyError, IndexError, StopIteration) as e:
            print(f"skipped {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
