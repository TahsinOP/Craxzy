"""Figures for the documentation site, one SVG per theme (docs/public/fig/<name>-{light,dark}.svg).

    env -u PYTHONPATH .venv/bin/python docs/figures/make_figures.py [name ...]

Every figure is drawn from the project's code or its result files (results/*.json,
data/*.npz); nothing is sketched by hand.  Backgrounds are transparent so the
figure sits on the page's own surface in both themes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from card_catcher import DATA, RESULTS, config  # noqa: E402
from card_catcher.config import Plane  # noqa: E402
from card_catcher.control.arm import discrete_pd_gains  # noqa: E402
from card_catcher.control.planner import TrajectoryQP, min_transfer_time  # noqa: E402
from card_catcher.envs.flight import FlightData  # noqa: E402
from card_catcher.estim import physics  # noqa: E402
from card_catcher.models.objects import OBJECTS  # noqa: E402
from card_catcher.models.scene import QD_MAX, QDD_MAX, build_flight_model  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "public" / "fig"
CFG = config.load()
H = CFG["sim"]["timestep"]

# validated categorical steps of the dataviz palette (light / dark surface)
THEMES = {
    "light": dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", violet="#6250d6",
                  ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7"),
    "dark": dict(blue="#3987e5", orange="#d95926", aqua="#199e70", yellow="#c98500", violet="#9085e9",
                 ink="#f2f1ec", ink2="#c3c2b7", muted="#8f8d86", grid="#33322f", axis="#55544f"),
}
T: dict = {}


def style(theme: str) -> None:
    global T
    T = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none", "savefig.transparent": True,
        "axes.edgecolor": T["axis"], "axes.linewidth": 0.8, "axes.grid": True, "grid.color": T["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.labelcolor": T["ink2"], "axes.titlecolor": T["ink"], "text.color": T["ink"], "xtick.color": T["muted"],
        "ytick.color": T["muted"], "xtick.labelcolor": T["ink2"], "ytick.labelcolor": T["ink2"], "axes.titlesize": 10.5,
        "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "legend.frameon": False, "legend.labelcolor": T["ink2"], "lines.linewidth": 2.0, "lines.markersize": 5.0,
        "font.family": "DejaVu Sans", "svg.fonttype": "none", "savefig.bbox": "tight", "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    })


def load(name: str):
    return json.loads((RESULTS / name).read_text())


def method():
    return {"physics": T["blue"], "pure": T["orange"], "hybrid": T["aqua"]}


def objcol():
    return {"ball": T["blue"], "plate": T["orange"], "card": T["aqua"]}


METHOD_NAME = {"physics": "physics only (EKF)", "pure": "pure learning (GRU)", "hybrid": "hybrid (physics + residual)"}


# ------------------------------------------------------------------ simulation
def fig_flights():
    """Side view of real flights against the nominal (drag-ballistic) model from the true launch state."""
    plane = Plane.from_cfg(CFG)
    kd = load("m1_params.json")
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [2.9, 1.5]})
    for a, obj in zip(ax, ("ball", "card")):
        D = FlightData.load(DATA / f"flight_{obj}_seed0.npz")
        for n, i in enumerate(range(0, 12)):
            k = int(D.cross_k[i]) + 2
            P = D.p[i, :k]
            Pn, _ = physics.rollout_nominal(D.p[i, 0], D.v[i, 0], kd[obj]["kd0"], D.dt, k - 1)
            a.plot(Pn[:, 0], Pn[:, 2], color=T["muted"], lw=1.0, ls="--", label="nominal model from the launch state" if n == 0 else None)
            a.plot(P[:, 0], P[:, 2], color=objcol()[obj], lw=1.4, alpha=0.9, label="simulated flight" if n == 0 else None)
        w = np.array(plane.w_range)
        pts = plane.c0[None] + w[:, None] * plane.ew[None]
        a.plot(pts[:, 0], pts[:, 2], color=T["ink"], lw=3, solid_capstyle="round", label="catch window (plane tilted 30°)")
        a.set(xlabel="x [m]", ylabel="z [m]", ylim=(0.4, 2.0), title=f"{obj}: 12 throws, side view")
        a.set_aspect("equal")
    hd, lb = ax[0].get_legend_handles_labels()
    hd2, lb2 = ax[1].get_legend_handles_labels()
    fig.legend([hd[1], hd2[1], hd[0], hd[2]], ["ball, simulated flight", "card, simulated flight", lb[0], lb[2]], loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.04))
    return fig


def fig_integrator():
    """Three discretisations of the same drag-ballistic model against MuJoCo, for a non-spinning ball."""
    m = build_flight_model("ball", CFG["sim"])
    d = mujoco.MjData(m)
    p0, v0 = np.array([2.8, 0.1, 1.2]), np.array([-3.5, 0.2, 3.0])
    d.qpos[:3], d.qvel[:3] = p0, v0
    n = 100
    P = [p0.copy()]
    for _ in range(n):
        mujoco.mj_step(m, d)
        P.append(d.qpos[:3].copy())
    P = np.array(P)
    kd = OBJECTS["ball"].kd_axis(np.array([1.0, 0, 0]), CFG["sim"]["density"])
    t = np.arange(n + 1) * H
    Pi, _ = physics.rollout_nominal(p0, v0, kd, H, n)
    # plain semi-implicit Euler: explicit drag
    p, v, Pe = p0.copy(), v0.copy(), [p0.copy()]
    for _ in range(n):
        v = v + H * (physics.GRAVITY + physics.drag_accel(v[None], np.array([kd]))[0])
        p = p + H * v
        Pe.append(p.copy())
    Pe = np.array(Pe)
    # continuous time: fine RK4 of the ODE
    def f(s):
        return np.r_[s[3:], physics.GRAVITY - kd * np.linalg.norm(s[3:]) * s[3:]]
    s, Pc, sub = np.r_[p0, v0], [p0.copy()], 20
    for _ in range(n):
        for _ in range(sub):
            hh = H / sub
            k1 = f(s); k2 = f(s + hh / 2 * k1); k3 = f(s + hh / 2 * k2); k4 = f(s + hh * k3)
            s = s + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Pc.append(s[:3].copy())
    Pc = np.array(Pc)
    fig, a = plt.subplots(figsize=(6.4, 3.6))
    for Pm, c, lab in ((Pc, T["orange"], "continuous-time ODE"), (Pe, T["yellow"], "semi-implicit Euler, explicit drag"),
                       (Pi, T["aqua"], "linearly-implicit step (used here)")):
        e = np.linalg.norm(Pm - P, axis=1) * 1e3
        a.semilogy(t[1:], np.maximum(e[1:], 1e-7), color=c, label=f"{lab}: {e[-1]:.2g} mm at 0.8 s")
    a.set(xlabel="flight time [s]", ylabel="distance from MuJoCo [mm]", title="Ball: model error against the simulator (h = 8 ms)")
    a.legend(loc="lower right")
    return fig


def fig_residual():
    """What the physics model leaves out: per-step residual acceleration along single throws."""
    kd = load("m1_params.json")
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.3))
    for obj in ("ball", "plate", "card"):
        D = FlightData.load(DATA / f"flight_{obj}_seed0.npz")
        r = physics.residual_accel(D.v, kd[obj]["kd0"], D.dt)
        i = 0
        k = int(D.cross_k[i])
        ax[0].plot(np.arange(k) * D.dt, np.linalg.norm(r[i, :k], axis=1), color=objcol()[obj], label=obj)
        mask = np.arange(r.shape[1])[None] < D.cross_k[:, None]
        mag = np.linalg.norm(r, axis=2)[mask]
        ax[1].hist(mag, bins=np.linspace(0, 16, 65), histtype="step", lw=1.8, density=True, color=objcol()[obj], label=obj)
    ax[0].set(xlabel="flight time [s]", ylabel="|residual acceleration| [m/s²]", title="One throw of each object")
    ax[1].set(xlabel="|residual acceleration| [m/s²]", ylabel="density", title="All steps of 1,000 throws", yscale="log")
    ax[0].legend(); ax[1].legend()
    return fig


def fig_mount():
    r = load("m0_mount.json")
    fig, a = plt.subplots(figsize=(5.6, 3.3))
    x = np.arange(3)
    for j, (mnt, c) in enumerate((("upright", T["blue"]), ("inverted", T["aqua"]))):
        y = [r[mnt][e]["usable_margin_0.3"] * 100 for e in ("20", "40", "60")]
        a.bar(x + (j - 0.5) * 0.38, y, 0.35, color=c, label=f"{mnt} mount")
        for xi, yi in zip(x + (j - 0.5) * 0.38, y):
            a.text(xi, yi + 1.5, f"{yi:.0f}", ha="center", fontsize=8.5, color=T["ink2"])
    a.set_xticks(x, ["20°", "40°", "60°"])
    a.set(xlabel="palm elevation above the horizontal", ylabel="window positions with a usable pose [%]", ylim=(0, 75),
          title="Why the arm hangs from a gantry")
    a.grid(axis="x", visible=False)
    a.legend(loc="upper left")
    return fig


def fig_tunnel():
    r = load("m0_tunnel.json")["conditions"]
    fig, a = plt.subplots(figsize=(5.6, 3.2))
    keys = [f"{c['timestep_ms']:.0f} ms\n{'on' if c['speculative'] else 'off'}" for c in r]
    vals = [c["tunnel_rate_mean"] * 100 for c in r]
    cols = [T["aqua"] if c["speculative"] else T["orange"] for c in r]
    a.bar(range(len(keys)), vals, 0.6, color=cols)
    for i, v in enumerate(vals):
        a.text(i, v + 2, f"{v:.0f} %", ha="center", fontsize=9, color=T["ink2"])
    a.set_xticks(range(len(keys)), keys)
    a.set_xlabel("timestep / speculative contacts")
    a.set(ylabel="throws that pass through the fingertip [%]", ylim=(0, 112), title="Card against a fingertip, 600 throws per bar")
    a.grid(axis="x", visible=False)
    return fig


# ------------------------------------------------------------------ estimation
def fig_m1():
    r = load("m1_ekf.json")["results"]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for obj in ("ball", "plate", "card"):
        taus = sorted(float(t) for t in r[obj])
        e = [r[obj][f"{t:g}"]["rms_cm"][0] for t in taus]
        c = [r[obj][f"{t:g}"]["cover90"][0] * 100 for t in taus]
        ax[0].semilogy(np.array(taus) * 1e3, e, "o-", color=objcol()[obj], label=obj)
        ax[1].plot(np.array(taus) * 1e3, c, "o-", color=objcol()[obj], label=obj)
    ax[0].axhline(1.0, color=T["ink2"], lw=1, ls="--")
    ax[0].text(640, 1.08, "1 cm target for the ball at 200 ms", fontsize=8.5, color=T["ink2"])
    ax[1].axhline(90, color=T["ink2"], lw=1, ls="--")
    for a in ax:
        a.invert_xaxis()
        a.set_xlabel("time before the catch [ms]")
        a.legend()
    ax[0].set(ylabel="catch-point error, RMS [cm]", title="Physics + EKF: error against time to catch")
    ax[1].set(ylabel="truth inside the predicted 90 % ellipse [%]", ylim=(0, 100), title="Is the covariance honest?")
    return fig


def _m2(obj="card"):
    return load("m2_residual.json")["objects"][obj]


def fig_m2_error():
    o = _m2()
    fig, a = plt.subplots(figsize=(6.4, 3.6))
    for m in ("physics", "pure", "hybrid"):
        taus = sorted(float(t) for t in o[m])
        y = np.array([o[m][f"{t:g}"]["rms_cm"][0] for t in taus])
        s = np.array([o[m][f"{t:g}"]["rms_cm"][1] for t in taus])
        a.plot(np.array(taus) * 1e3, y, "o-", color=method()[m], label=METHOD_NAME[m])
        a.fill_between(np.array(taus) * 1e3, y - s, y + s, color=method()[m], alpha=0.2, lw=0)
    a.invert_xaxis()
    a.set(xlabel="time before the catch [ms]", ylabel="catch-point error, RMS [cm]", ylim=(0, None),
          title="Card: prediction error against time to catch (3 seeds)")
    a.legend()
    return fig


def fig_m2_calibration():
    o = _m2()
    fig, a = plt.subplots(figsize=(4.6, 4.0))
    lev = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
    a.plot([0, 100], [0, 100], color=T["ink2"], lw=1, ls="--", label="ideal")
    for (band, c), col in zip(o["calibration"].items(), (T["aqua"], T["violet"], T["yellow"])):
        a.plot(lev, np.array(c["coverage"]) * 100, "o-", color=col, label=f"horizon {band}")
    a.set(xlabel="nominal central interval [%]", ylabel="residuals inside the interval [%]", xlim=(0, 100), ylim=(0, 100),
          title="Residual variance: reliability")
    a.set_aspect("equal")
    a.legend(loc="upper left")
    return fig


def fig_m2_coverage():
    o = _m2()
    fig, a = plt.subplots(figsize=(6.4, 3.6))
    for m in ("physics", "pure", "hybrid"):
        taus = sorted(float(t) for t in o[m])
        a.plot(np.array(taus) * 1e3, [o[m][f"{t:g}"]["cover90"][0] * 100 for t in taus], "o-", color=method()[m], label=METHOD_NAME[m])
    taus = sorted(float(t) for t in o["hybrid_white"])
    a.plot(np.array(taus) * 1e3, [o["hybrid_white"][f"{t:g}"]["cover90"][0] * 100 for t in taus], "--", color=method()["hybrid"],
           label="hybrid, error treated as white (φ = 0)")
    a.axhline(90, color=T["ink2"], lw=1, ls="--")
    a.invert_xaxis()
    a.set(xlabel="time before the catch [ms]", ylabel="catch points inside the 90 % ellipse [%]", ylim=(0, 100),
          title="Card: coverage of the catch-point covariance")
    a.legend(loc="lower left")
    return fig


def fig_m2_data():
    o = _m2()
    sw = o["sweep_rms_cm_at_200ms"]
    ns = sorted(int(n) for n in sw)
    fig, a = plt.subplots(figsize=(5.6, 3.4))
    a.axhline(o["physics"]["0.2"]["rms_cm"][0], color=method()["physics"], label=METHOD_NAME["physics"])
    for m in ("pure", "hybrid"):
        a.plot(ns, [sw[str(n)][m] for n in ns], "o-", color=method()[m], label=METHOD_NAME[m])
    a.set_xscale("log")
    a.set_xticks(ns, [str(n) for n in ns])
    a.minorticks_off()
    a.set(xlabel="training throws", ylabel="RMS error 200 ms before the catch [cm]", ylim=(0, 7.5), title="Card: data efficiency (seed 0)")
    a.legend(loc="lower left")
    return fig


def fig_ellipses():
    e = load("docs_data.json")["ellipses"]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True, sharey=True)
    s90 = np.sqrt(4.605)                       # chi-square(2 dof) 90 % quantile
    cols = [T["violet"], T["blue"], T["yellow"], T["orange"]]
    for a, m in zip(ax, ("physics", "hybrid")):
        for (tau, c) in zip(("0.4", "0.3", "0.2", "0.1"), cols):
            uw, cov = np.array(e[m][tau]["uw"]) * 100, np.array(e[m][tau]["cov"]) * 1e4
            w, V = np.linalg.eigh(cov)
            ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
            a.add_patch(Ellipse(uw, 2 * s90 * np.sqrt(w[1]), 2 * s90 * np.sqrt(w[0]), angle=ang, fill=False, ec=c, lw=1.8))
            a.plot(*uw, "o", color=c, label=f"{float(tau)*1e3:.0f} ms before the catch")
        a.plot(*(np.array(e["truth_uw"]) * 100), "*", color=T["ink"], ms=13, label="where it actually crossed")
        a.set(xlabel="u on the catch plane [cm]", title=f"{METHOD_NAME[m]}")
        a.set_aspect("equal")
    ax[0].set_ylabel("w on the catch plane [cm]")
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return fig


# --------------------------------------------------------------------- control
def fig_qp():
    """One joint: the closed-form minimum-effort plan, and the same transfer with less time (limits active, OSQP)."""
    j = 3
    qp = TrajectoryQP(H, np.full(7, -2.9), np.full(7, 2.9))
    q0, v0, qg = 0.0, 0.0, 1.2
    tmin = float(min_transfer_time(np.array(qg - q0), np.array(v0), qp.v_max[j], qp.a_max[j]))
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.1))
    for N, c in ((150, T["blue"]), (int(np.ceil(tmin / H)) + 4, T["orange"])):
        a = qp.closed_form(q0, v0, qg, N)
        lab = f"{N*H:.2f} s: no limit active, closed form"
        if not qp.within_limits(j, a, q0, v0):
            a = qp.solve_osqp(j, q0, v0, qg, N)
            while a is None:                       # OSQP's iteration budget is small; take the next horizon that converges
                N += 1
                a = qp.solve_osqp(j, q0, v0, qg, N)
            lab = None
        lab = lab or f"{N*H:.2f} s: limits active, OSQP"
        v = v0 + H * np.cumsum(a)
        q = q0 + H * np.cumsum(v)
        t = np.arange(1, N + 1) * H
        ax[0].plot(np.r_[0, t], np.r_[q0, q], color=c, label=lab)
        ax[1].plot(np.r_[0, t], np.r_[v0, v], color=c)
        ax[2].step(np.r_[0, t], np.r_[a, a[-1]], where="post", color=c)
    ax[1].axhline(qp.v_max[j], color=T["ink2"], lw=1, ls="--")
    ax[2].axhline(qp.a_max[j], color=T["ink2"], lw=1, ls="--")
    ax[2].axhline(-qp.a_max[j], color=T["ink2"], lw=1, ls="--")
    ax[0].set(xlabel="time [s]", ylabel="joint angle [rad]", title="Position")
    ax[1].set(xlabel="time [s]", ylabel="velocity [rad/s]", title="Velocity (dashed: limit)")
    ax[2].set(xlabel="time [s]", ylabel="acceleration [rad/s²]", title="Acceleration (dashed: limits)")
    ax[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"Joint 4, 1.2 rad from rest to rest; time-optimal transfer {tmin:.3f} s", x=0.01, ha="left", fontsize=10.5,
                 fontweight="bold", color=T["ink"])
    fig.tight_layout()
    return fig


def fig_pd():
    """Sampled error dynamics e_{k+1} = (2 - h Kd - h^2 Kp) e_k - (1 - h Kd) e_{k-1}."""
    w = 25.0
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.3), sharey=True)
    for a, h in zip(ax, (0.008, 0.032)):
        n = int(0.4 / h)
        for (kp, kd), c, lab in ((discrete_pd_gains(w, h), T["aqua"], "pole placement on the sampled system"),
                                 ((w * w, 2 * w), T["orange"], "continuous-time gains ω², 2ω")):
            e = [1.0, 1.0]
            for _ in range(n):
                e.append((2 - h * kd - h * h * kp) * e[-1] - (1 - h * kd) * e[-2])
            a.plot(np.arange(n + 1) * h, e[1:], "o-" if h > 0.02 else "-", ms=3, color=c, label=lab)
        t = np.linspace(0, 0.4, 200)
        a.plot(t, (1 + w * t) * np.exp(-w * t), color=T["ink2"], lw=1, ls="--", label="critically damped target (1 + ωt)e^(−ωt)")
        a.set(xlabel="time [s]", title=f"h = {h*1e3:.0f} ms, ω = 25 rad/s")
    ax[0].set_ylabel("tracking error (initial error = 1)")
    ax[0].legend(loc="upper right", fontsize=7.5)
    return fig


# ------------------------------------------------------------------ evaluation
def _bars(a, groups, series, get, colors, labels, fmt="{:.0f}"):
    w = 0.8 / len(series)
    for j, s in enumerate(series):
        xs, ys, es = [], [], []
        for i, g in enumerate(groups):
            v = get(g, s)
            if v is None:
                continue
            xs.append(i + (j - (len(series) - 1) / 2) * w); ys.append(v[0] * 100); es.append(v[1] * 100)
        a.bar(xs, ys, w * 0.92, yerr=es, color=colors[j], label=labels[j], error_kw={"ecolor": T["ink2"], "lw": 1})
        for x, y in zip(xs, ys):
            a.text(x, y + 2.5, fmt.format(y), ha="center", fontsize=8.5, color=T["ink2"])
    a.set_xticks(range(len(groups)), groups)
    a.grid(axis="x", visible=False)


def fig_m3_main():
    r = load("m3_catch.json")
    allk = lambda o: {**r["ablation_predictor"].get(o, {}), **r["main"].get(o, {})}  # noqa: E731
    fig, a = plt.subplots(figsize=(6.4, 3.6))
    _bars(a, ["ball", "card", "plate"], ["oracle", "physics", "hybrid"], lambda o, k: allk(o).get(k, {}).get("caught"),
          [T["muted"], T["blue"], T["aqua"]], ["perfect predictor", METHOD_NAME["physics"], METHOD_NAME["hybrid"]])
    a.axhline(90, color=T["ink2"], lw=1, ls="--")
    a.set(ylabel="catch rate [%]", ylim=(0, 110), title="Catch rate by object and predictor (parallel gripper, 50 Hz)")
    a.legend(loc="center right")
    return fig


def fig_m3_contacts():
    ab = load("m3_catch.json")["ablation_contacts"]
    keys = ["8ms_on", "8ms_off", "16ms_on", "16ms_off"]
    labels = ["8 ms\non", "8 ms\noff", "16 ms\non", "16 ms\noff"]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5), sharey=True)
    for a, obj in zip(ax, ("ball", "card")):
        _bars(a, keys, ["caught", "clean"], lambda k, s: ab[obj][k][s], [T["yellow"], T["aqua"]],
              ["caught (contact sensors)", "clean (and no interpenetration)"])
        a.set_xticks(range(4), labels)
        a.set(xlabel="timestep / speculative contacts", title=f"{obj}, perfect predictor")
    ax[0].set(ylabel="throws [%]", ylim=(0, 112))
    ax[1].legend(loc="upper right")
    return fig


def fig_m3_suction():
    r = load("m3_catch.json")
    st = r["stretch_suction"]
    grip = lambda o: {**r["ablation_predictor"].get(o, {}), **r["main"].get(o, {})}  # noqa: E731
    fig, a = plt.subplots(figsize=(5.8, 3.5))
    def get(o, s):
        return grip(o)["hybrid"]["retained"] if s == "gripper" else st[o]["hybrid"]["retained"]
    _bars(a, ["plate", "card"], ["gripper", "suction"], get, [T["blue"], T["aqua"]], ["parallel gripper", "suction disc (adhesion)"])
    a.set(ylabel="retained in the hand [%]", ylim=(0, 105), title="Hybrid predictor: gripper against suction disc")
    a.legend(loc="upper right")
    return fig


def fig_m4_pareto():
    names = {"periodic": "periodic (fixed rate)", "variance": "variance-triggered", "drift": "drift-triggered", "learned": "learned (PPO)"}
    cols = {"periodic": T["blue"], "variance": T["yellow"], "drift": T["violet"], "learned": T["aqua"]}
    marks = {"periodic": "o", "variance": "s", "drift": "D", "learned": "^"}
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9))
    for a, obj in zip(ax, ("ball", "card")):
        r = load(f"m4_{obj}.json")
        for name in names:
            rows = sorted(r[name], key=lambda x: x["plans"][0])
            a.errorbar([x["plans"][0] for x in rows], [x["catch"][0] * 100 for x in rows], yerr=[x["catch"][1] * 100 for x in rows],
                       marker=marks[name], color=cols[name], label=names[name], capsize=2, elinewidth=1)
        b = next(x for x in r["periodic"] if x["setting"] == 50.0)
        a.annotate("50 Hz baseline", (b["plans"][0], b["catch"][0] * 100), textcoords="offset points", xytext=(0, -16), ha="center",
                   fontsize=8.5, color=T["ink2"])
        a.set_xscale("log")
        a.set(xlabel="planner calls per throw", ylabel="catch rate [%]", title=f"{obj} ({r['predictor']} predictor)")
        a.legend(loc="lower right")
    return fig


def fig_m4_ppo():
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for obj in ("ball", "card"):
        log = load(f"m4_ppo_log_{obj}.json")
        it = np.array([x["iter"] for x in log])
        k = np.ones(5) / 5
        sm = lambda y: np.convolve(np.pad(y, 2, mode="edge"), k, mode="valid")  # noqa: E731
        ax[0].plot(it, sm(np.array([x["return"] for x in log])), color=objcol()[obj], label=obj)
        ax[1].plot(it, sm(np.array([x["plans"] for x in log])), color=objcol()[obj], label=obj)
    ax[0].set(xlabel="PPO iteration (96 throws each)", ylabel="return = catch − λ · replans", title="Training return (5-iteration mean)")
    ax[1].set(xlabel="PPO iteration", ylabel="planner calls per throw", title="Calls the policy spends")
    ax[0].legend(); ax[1].legend()
    return fig


def fig_trigger_times():
    tt = load("docs_data.json")["trigger_times"]
    cols = [T["blue"], T["yellow"], T["violet"], T["aqua"]]
    fig, ax = plt.subplots(len(tt), 1, figsize=(6.8, 5.6), sharex=True)
    bins = np.linspace(0, 0.8, 41)
    for a, (name, d), c in zip(ax, tt.items(), cols):
        a.hist(np.array(d["tau"]) * 1e3, bins=bins * 1e3, color=c, weights=np.full(len(d["tau"]), 1 / 128))
        a.set_title(f"{name}: {d['catch']*100:.0f} % caught with {d['plans']:.1f} calls per throw", fontsize=9.5)
        a.set_ylabel("replans\nper throw")
        a.axvspan(0, 150, color=T["ink2"], alpha=0.12, lw=0)
    ax[-1].invert_xaxis()
    ax[-1].set_xlabel("time before the catch at which the planner is called [ms]   (shaded: the last 150 ms)")
    fig.tight_layout()
    return fig


FIGS = {
    "flights": fig_flights, "integrator": fig_integrator, "residual": fig_residual, "mount": fig_mount, "tunnel": fig_tunnel,
    "m1": fig_m1, "m2-error": fig_m2_error, "m2-calibration": fig_m2_calibration, "m2-coverage": fig_m2_coverage,
    "m2-data": fig_m2_data, "ellipses": fig_ellipses, "qp": fig_qp, "pd": fig_pd, "m3-main": fig_m3_main,
    "m3-contacts": fig_m3_contacts, "m3-suction": fig_m3_suction, "m4-pareto": fig_m4_pareto, "m4-ppo": fig_m4_ppo,
    "trigger-times": fig_trigger_times,
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or FIGS:
        for theme in THEMES:
            style(theme)
            fig = FIGS[name]()
            fig.savefig(OUT / f"{name}-{theme}.svg")
            plt.close(fig)
        print("fig:", name, flush=True)


if __name__ == "__main__":
    main()
