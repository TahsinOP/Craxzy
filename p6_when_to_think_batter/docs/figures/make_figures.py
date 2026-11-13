"""Figures for the documentation site, one SVG per theme (docs/public/fig/<name>-{light,dark}.svg).

    env -u PYTHONPATH .venv/bin/python docs/figures/make_figures.py [name ...]

Every figure is drawn from the project's code or its result files (results/*.json,
data/models/*.pt); nothing is sketched by hand.  Backgrounds are transparent so the
figure sits on the page's own surface in both themes.  The one figure that needs raw
per-pitch data the JSON does not keep (when each trigger replans) recomputes it once
and caches it in results/docs_when.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from batter import DATA, RESULTS, config  # noqa: E402
from batter.control.planner import crossing  # noqa: E402
from batter.estim.flight import FlightModel  # noqa: E402
from batter.models.pitch import generate  # noqa: E402
from batter.models.scene import build_cage_model  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "public" / "fig"
CFG = config.load()

# validated categorical steps of the dataviz palette (light / dark surface), the same as P1's site
THEMES = {
    "light": dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", violet="#6250d6", magenta="#e87ba4",
                  ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", fill="#e9e8e2"),
    "dark": dict(blue="#3987e5", orange="#d95926", aqua="#199e70", yellow="#c98500", violet="#9085e9", magenta="#d06e95",
                 ink="#f2f1ec", ink2="#c3c2b7", muted="#8f8d86", grid="#33322f", axis="#55544f", fill="#2b2a27"),
}
T: dict = {}
# colour follows the entity in every figure
TRIG = {"periodic": "blue", "innovation": "yellow", "variance": "magenta", "drift": "violet", "learned": "aqua", "wm": "orange",
        "events": "aqua", "learned_s": "aqua"}
LABEL = {"periodic": "fixed rate", "innovation": "innovation (χ² gate)", "variance": "variance (Trimpe)",
         "drift": "drift (send-on-delta)", "learned": "learned (PPO)", "wm": "world-model planner, fixed rate"}
MARK = {"periodic": "o", "innovation": "s", "variance": "^", "drift": "v", "learned": "D", "wm": "P", "learned_s": "D"}


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


def c(name: str) -> str:
    return T[TRIG.get(name, name)]


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def save(fig, name: str, theme: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{theme}.svg")
    plt.close(fig)


def errorbar(ax, x, rows, key, color, scale=1.0, **kw):
    y = np.array([r[key] for r in rows])
    ax.errorbar(x, scale * y[:, 0], scale * y[:, 1], color=color, capsize=2, **kw)


# --------------------------------------------------------------------------- figures
def pitches(theme: str) -> None:
    fm = FlightModel.identify(build_cage_model(CFG))
    P = generate(fm, CFG, 400, seed=0)
    traj = fm.rollout(P.x0, int(P.cross_t.max() / fm.h) + 6)
    x0 = P.x0.copy()
    x0[:, 6:9] = 0.0
    traj0 = fm.rollout(x0, traj.shape[0] - 1)
    sx = CFG["pitch"]["strike_x"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    a = ax[0]
    for i in range(0, 400, 7):
        k = int(np.ceil(crossing(traj[:, i], sx)[0])) + 1
        a.plot(traj[:k, i, 0], traj[:k, i, 2], color=T["orange"] if P.x0[i, 7] > 0 else T["blue"], lw=0.8, alpha=0.6)
    a.plot([], [], color=T["orange"], label="backspin: floats")
    a.plot([], [], color=T["blue"], label="topspin: dives")
    a.axvline(sx, color=T["ink2"], lw=0.8, ls=":")
    a.text(sx + 0.08, 0.72, "hitting plane", color=T["ink2"], fontsize=8)
    a.set(xlabel="distance from the robot [m]", ylabel="height [m]", title="Pitches, side view")
    a.invert_xaxis()
    a.legend(loc="lower left")
    a = ax[1]
    dz = np.array([crossing(traj[:, i], sx)[1][2] - crossing(traj0[:, i], sx)[1][2] for i in range(400)])
    a.axhspan(-0.6, 0.6, color=T["orange"], alpha=0.3, lw=0, label="bat thickness")
    a.scatter(P.x0[:, 7], 100 * dz, s=6, color=T["blue"], alpha=0.6)
    a.set(xlabel="spin about the horizontal axis [rad/s]", ylabel="[cm]", title="Height change at the plate caused by spin")
    a.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "pitches", theme)


def tunnel(theme: str) -> None:
    r = load("m0")
    hs = sorted({q["timestep"] for q in r["static"]})
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for a, rows, key, title in [(ax[0], r["static"], "bounced", "Bat held still: the ball bounces off"),
                                (ax[1], r["swing"], "hit", "Swinging batter: the ball is returned")]:
        x = np.arange(len(hs))
        for spec, col, lab, dx in [(False, "orange", "speculative contacts off", -0.19), (True, "blue", "speculative contacts on", 0.19)]:
            q = [next(z for z in rows if z["timestep"] == h and z["speculative"] == spec) for h in hs]
            y = np.array([100 * z[key][0] for z in q])
            e = np.array([100 * z[key][1] for z in q])
            a.bar(x + dx, y, 0.36, yerr=e, color=T[col], label=lab, capsize=3, error_kw={"ecolor": T["ink2"]})
            for xi, yi in zip(x + dx, y):
                a.text(xi, yi + 2.5, f"{yi:.0f}", ha="center", fontsize=8, color=T["ink2"])
        a.set_xticks(x, [f"{1e3 * h:.0f} ms" for h in hs])
        a.set(xlabel="simulator timestep", title=title, ylim=(0, 114))
        a.grid(axis="x", visible=False)
    ax[0].set_ylabel("pitches [%]")
    ax[1].legend(loc="center right")
    fig.tight_layout()
    save(fig, "tunnel", theme)


def estimator(theme: str) -> None:
    e = load("m1")["estimator"]
    tau = 1e3 * np.array(e["tau"])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    a = ax[0]
    med, p90, sd = (np.array(e[k]) for k in ("err_med", "err_p90", "sd_pred"))
    a.errorbar(tau, 100 * med[:, 0], 100 * med[:, 1], color=T["blue"], marker="o", capsize=2, label="error, median")
    a.plot(tau, 100 * p90[:, 0], color=T["blue"], ls=":", label="error, 90th percentile")
    a.plot(tau, 100 * sd[:, 0], color=T["orange"], marker="s", label="the filter's own standard deviation")
    a.axhline(4.5, color=T["ink2"], lw=0.8, ls="--")
    a.text(tau.max(), 5.3, "half the bat's width", ha="right", fontsize=8, color=T["ink2"])
    a.set(xlabel="time left to the plate [ms]", ylabel="[cm]", yscale="log", title="Forecast of where the ball crosses the plate")
    a.invert_xaxis()
    a.legend(loc="lower left")
    a = ax[1]
    cov = np.array(e["cover90"])
    a.errorbar(tau, 100 * cov[:, 0], 100 * cov[:, 1], color=T["blue"], marker="o", capsize=2)
    a.axhline(90, color=T["ink2"], lw=0.8, ls="--")
    a.set(xlabel="time left to the plate [ms]", ylabel="truth inside the 90 % ellipse [%]", ylim=(70, 100),
          title="Is the covariance honest?")
    a.invert_xaxis()
    fig.tight_layout()
    save(fig, "estimator", theme)


def calibration(theme: str) -> None:
    cb = load("m1")["calibration"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    e, y = np.array(cb["e_grid"]), np.array(cb["radial_error_m"])
    ax.plot(e, y, color=T["blue"], marker="o")
    ax.axhline(0, color=T["ink2"], lw=0.8)
    ax.axvline(cb["restitution_landing"], color=T["aqua"], ls="--", lw=1.2)
    ax.text(cb["restitution_landing"] + 0.003, y.max() * 0.8, f"e = {cb['restitution_landing']:.3f}", color=T["aqua"], fontsize=9)
    ax.set(xlabel="restitution in the planner's impact model", ylabel="mean landing error along the range [m]",
           title="Fitting the effective restitution")
    fig.tight_layout()
    save(fig, "calibration", theme)


def speed(theme: str) -> None:
    s = load("m1")["speed"]
    m2 = load("m2")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    variants = [(s["analytic 100 Hz"], "blue", "o", "-", "analytic planner, 100 Hz"),
                (s["analytic 100 Hz, true ball state"], "muted", "s", "--", "the same with the true ball state"),
                (m2["speed"]["world-model planner (pure learning)"], "orange", "P", "-", "world-model planner, 100 Hz"),
                (s["analytic, planned once"], "violet", "^", ":", "analytic planner, called once")]
    for rows, col, mk, ls, lab in variants:
        v = [r["speed"] for r in rows]
        errorbar(ax[0], v, rows, "hit", T[col], 100, marker=mk, ls=ls, label=lab)
        errorbar(ax[1], v, rows, "on_target", T[col], 100, marker=mk, ls=ls, label=lab)
    ax[0].set(xlabel="pitch speed at release [m/s]", ylabel="[%]", title="Hit rate", ylim=(-3, 104))
    ax[1].set(xlabel="pitch speed at release [m/s]", title="Ball lands in the target zone")
    ax[0].legend(loc="center left", fontsize=7.8)
    fig.tight_layout()
    save(fig, "speed", theme)


def wm_compare(theme: str) -> None:
    r = load("m2")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.5), gridspec_kw={"width_ratios": [1, 1]})
    t = r["timing"]
    names = [("analytic", "analytic\nplanner", "blue"), ("world model", "world model\n+ MPPI", "orange"),
             ("world model prior only", "world-model\nprior only", "orange")]
    a = ax[0]
    for i, (k, lab, col) in enumerate(names):
        a.bar(i, t[k]["ms_median"], 0.6, color=T[col], alpha=1.0 if i < 2 else 0.55)
        a.plot([i, i], [t[k]["ms_median"], t[k]["ms_p90"]], color=T["ink2"], lw=1.2)
        a.text(i, t[k]["ms_p90"] + 0.6, f"{t[k]['ms_median']:.1f} ms", ha="center", fontsize=8, color=T["ink2"])
    a.axhline(10, color=T["ink2"], ls="--", lw=0.8)
    a.text(2.4, 10.4, "control period", ha="right", fontsize=8, color=T["ink2"])
    a.set_xticks(range(3), [n[1] for n in names])
    a.set(ylabel="ms per call (bar: median, line: to p90)", title="Planning time, alone on one core")
    a.grid(axis="x", visible=False)
    a = ax[1]
    d = r["data_sweep"]
    x = [q["pitches"] for q in d]
    errorbar(a, x, d, "hit", T["orange"], 100, marker="o", label="world model: hit rate")
    errorbar(a, x, d, "on_target", T["orange"], 100, marker="s", ls="--", label="world model: on target")
    an = r["all"]["analytic planner (classical)"]
    a.axhline(100 * an["on_target"][0], color=T["blue"], ls="--", label="analytic planner: on target")
    a.set(xscale="log", xlabel="pitches of batting data", ylabel="[%]", ylim=(0, 104), title="The learned planner against data")
    a.legend(loc="lower right", fontsize=7.8)
    fig.tight_layout()
    save(fig, "wm", theme)


def pareto(theme: str) -> None:
    r = load("m3")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for kind in ("wm", "variance", "drift", "innovation", "periodic", "learned"):
        rows = sorted(r[kind], key=lambda q: q["plans"][0])
        x = [q["plans"][0] for q in rows]
        lw = 2.5 if kind in ("learned", "periodic") else 1.5
        errorbar(ax[0], x, rows, "quality", c(kind), marker=MARK[kind], label=LABEL[kind], lw=lw)
        errorbar(ax[1], [q["ms_per_pitch"][0] for q in rows], rows, "quality", c(kind), marker=MARK[kind], lw=lw)
    ax[0].set(xscale="log", xlabel="planner calls per pitch", ylabel="hit quality", title="Hit quality against planner calls")
    ax[1].set(xscale="log", xlabel="planning time per pitch [ms, wall-clock]", title="The same against measured compute")
    legend_below(fig, ax[0], ncol=3)
    save(fig, "pareto", theme)


def when_data() -> dict:
    """Times to the plate at which each trigger replans, at about 4 calls per pitch (cached)."""
    path = RESULTS / "docs_when.json"
    if path.exists():
        return json.loads(path.read_text())
    from batter.eval.runner import evaluate
    from batter.eval.sweeps import at_budget, policy_path

    r = load("m3")
    fm = FlightModel.identify(build_cage_model(CFG))
    P = generate(fm, CFG, 300, 777)
    out = {}
    for kind in ("periodic", "innovation", "variance", "drift", "learned"):
        row = at_budget(r[kind], 4.0)
        spec = ("learned", (str(policy_path("nominal")), row["setting"])) if kind == "learned" else (kind, row["setting"])
        o = evaluate(CFG, spec, P, 0)
        tau = np.concatenate([P.cross_t[i] - np.asarray(s, float)[1:] * fm.h for i, s in enumerate(o["plan_steps"])])
        out[kind] = {"tau": tau.tolist(), "plans": row["plans"][0], "quality": row["quality"][0], "n": 300}
    path.write_text(json.dumps(out))
    return out


def when(theme: str) -> None:
    w = when_data()
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    bins = np.linspace(0.0, 0.7, 29)
    for kind in ("variance", "drift", "innovation", "periodic", "learned"):
        h, _ = np.histogram(w[kind]["tau"], bins)
        ax.plot(1e3 * 0.5 * (bins[:-1] + bins[1:]), h / w[kind]["n"], color=c(kind), marker=".",
                lw=2.4 if kind in ("learned", "periodic") else 1.4,
                label=f"{LABEL[kind]}: {w[kind]['plans']:.1f} calls, quality {w[kind]['quality']:.2f}")
    ax.invert_xaxis()
    ax.set(xlabel="time left to the plate [ms]   (the pitch runs left to right)", ylabel="replans per pitch per 25 ms",
           title="When each trigger thinks, at about 4 planner calls per pitch")
    legend_below(fig, ax, ncol=2)
    save(fig, "when", theme)


def ppo(theme: str) -> None:
    import torch

    log = torch.load(DATA / "models" / "trigger_nominal.pt", map_location="cpu")["log"]
    it = np.array([q["iter"] for q in log])
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9))
    for a, key, title in [(ax[0], "return", "return = quality − λ · calls"), (ax[1], "quality", "hit quality"),
                          (ax[2], "plans", "planner calls per pitch")]:
        y = np.array([q[key] for q in log], float)
        a.plot(it, y, color=T["aqua"], alpha=0.35, lw=1.0)
        k = 8
        a.plot(it[k - 1:], np.convolve(y, np.ones(k) / k, mode="valid"), color=T["aqua"])
        a.set(xlabel="PPO iteration", title=title)
    fig.tight_layout()
    save(fig, "ppo", theme)


def surprise_pareto(theme: str) -> None:
    r = load("m4")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for a, kind, title in [(ax[0], "spin", "Pitches with a spin change"), (ax[1], "gust", "Pitches with a gust")]:
        for name in ("variance", "drift", "innovation", "periodic", "learned", "learned_s"):
            rows = sorted(r["pareto"][kind][name], key=lambda q: q["plans"][0])
            lab = {"learned": "learned (never saw a surprise)", "learned_s": "learned (trained with surprises)"}.get(name, LABEL.get(name))
            errorbar(a, [q["plans"][0] for q in rows], rows, "quality", c(name), marker=MARK[name], label=lab,
                     ls="--" if name == "learned" else "-", lw=2.4 if name in ("learned_s", "periodic") else 1.4)
        a.set(xscale="log", xlabel="planner calls per pitch", title=title)
    ax[0].set_ylabel("hit quality")
    legend_below(fig, ax[0], ncol=3)
    save(fig, "surprise-pareto", theme)


def reaction(theme: str) -> None:
    r = load("m4")
    g = r["reaction"]["gust"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.7))
    a = ax[0]
    grid = np.linspace(0, 0.4, 81)
    for name in ("variance", "drift", "innovation", "periodic", "learned_s"):
        ls_, lp_ = np.array(g[name]["latency_s"]), np.array(g[name]["latency_p"])
        lab = "learned (PPO)" if name == "learned_s" else LABEL[name]
        a.plot(1e3 * grid, [100 * np.mean(ls_ <= x) for x in grid], color=c(name), label=lab,
               lw=2.4 if name in ("learned_s", "innovation") else 1.4)
        a.plot(1e3 * grid, [100 * np.mean(lp_ <= x) for x in grid], color=c(name), ls=":", lw=1.0)
    a.plot([], [], color=T["ink2"], ls=":", lw=1.0, label="same pitches, no gust")
    a.set(xlabel="time since the gust began [ms]", ylabel="pitches with a planner call by then [%]",
          title="Latency to the next planner call")
    a.legend(fontsize=7.4, loc="center right")
    a = ax[1]
    order = ["periodic", "variance", "drift", "innovation", "learned_s"]
    x = np.arange(len(order))
    plain = np.array([g[k]["quality_plain"] for k in order])
    sur = np.array([g[k]["quality_surprise"] for k in order])
    a.bar(x - 0.2, plain[:, 0], 0.38, color=T["fill"], edgecolor=T["axis"], label="same pitches, no gust")
    a.bar(x + 0.2, sur[:, 0], 0.38, yerr=sur[:, 1], color=[c(k) for k in order], label="with the gust", capsize=2,
          error_kw={"ecolor": T["ink2"]})
    for xi, k in zip(x, order):
        a.text(xi, 1.03, f"{g[k]['calls_plain'][0]:.1f}→{g[k]['calls_surprise'][0]:.1f}", ha="center", fontsize=7.5, color=T["ink2"])
    a.set_xticks(x, ["fixed\nrate", "variance", "drift", "innovation", "learned"], fontsize=8)
    a.set(ylabel="hit quality", ylim=(0, 1.12), title="Quality at ~4 calls (numbers: calls without → with)")
    a.grid(axis="x", visible=False)
    a.legend(loc="lower left", fontsize=7.4)
    fig.tight_layout()
    save(fig, "reaction", theme)


EV_LABEL = {"events": "event trigger (event camera)", "innovation": "innovation trigger (frame camera, 30 ms late)", "periodic": "fixed rate"}
EV_COL = {"events": "aqua", "innovation": "yellow", "periodic": "blue"}


def events(theme: str) -> None:
    r = load("m5")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for a, sur, title in [(ax[0], "none", "Ordinary pitches"), (ax[1], "gust", "Pitches with a gust")]:
        for k in ("periodic", "innovation", "events"):
            rows = sorted(r["pareto"][sur][k], key=lambda q: q["plans"][0])
            errorbar(a, [q["plans"][0] for q in rows], rows, "quality", T[EV_COL[k]], marker="o", label=EV_LABEL[k])
        a.set(xscale="log", xlabel="planner calls per pitch", ylim=(0, 1.02), title=title)
    ax[0].set_ylabel("hit quality")
    legend_below(fig, ax[0], ncol=3)
    save(fig, "events", theme)


def events_latency(theme: str) -> None:
    r = load("m5")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    grid = np.linspace(0, 0.4, 81)
    for k in ("periodic", "innovation", "events"):
        g = r["reaction"][k]["gust"]
        ax.plot(1e3 * grid, [100 * np.mean(np.array(g["latency"]) <= x) for x in grid], color=T[EV_COL[k]], label=EV_LABEL[k])
        ax.plot(1e3 * grid, [100 * np.mean(np.array(g["latency_placebo"]) <= x) for x in grid], color=T[EV_COL[k]], ls=":", lw=1.0)
    ax.plot([], [], color=T["ink2"], ls=":", lw=1.0, label="same pitches, no gust")
    ax.set(xlabel="time since the gust began [ms]", ylabel="pitches with a planner call by then [%]",
           title="Latency to the next planner call after a gust")
    legend_below(fig, ax, ncol=2)
    save(fig, "events-latency", theme)


def rally_fig(theme: str) -> None:
    r = load("rally")
    R = r["rallies"]
    names = list(R)
    cap = r["config"]["rally"]["max_hits"]
    col = [T["muted"], T["blue"], T["aqua"], T["aqua"], T["aqua"], T["orange"]]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    y = np.array([R[k]["returns"] for k in names])
    a.barh(range(len(names)), y[:, 0], xerr=y[:, 1], color=col, capsize=3, error_kw={"ecolor": T["ink2"]})
    for i in range(len(names)):
        a.text(y[i, 0] + 0.25, i, f"{y[i, 0]:.1f}", va="center", fontsize=8.5, color=T["ink2"])
    a.set_yticks(range(len(names)), names, fontsize=8.2)
    a.invert_yaxis()
    a.grid(axis="y", visible=False)
    a.set(xlabel=f"returns per rally (capped at {cap})", title="Rally length, 3 seeds")
    a = ax[1]
    for k, c_, ls in [(names[0], T["muted"], "--"), (names[1], T["blue"], "-"), (names[2], T["aqua"], "-")]:
        h = np.array(R[k]["hist"], float)
        surv = 1.0 - np.cumsum(h) / h.sum()
        a.step(np.arange(len(h)), np.r_[1.0, surv[:-1]] * 100, where="post", color=c_, ls=ls, label=k)
    a.set(xlabel="returns", ylabel="rallies still going [%]", title="How long rallies last", ylim=(0, 104))
    legend_below(fig, a, ncol=1)
    save(fig, "rally", theme)


def rally_feasibility(theme: str) -> None:
    r = load("rally")
    f = r["feasibility"]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for i, row in enumerate(f):
        lo, med, hi = row["paddle_speed_needed"]
        ax.plot([lo, hi], [i, i], color=T["blue"], lw=6, solid_capstyle="round", alpha=0.45)
        ax.plot(med, i, "o", color=T["blue"])
    ax.axvline(f[0]["paddle_speed_available"], color=T["orange"], lw=1.6, ls="--")
    ax.text(f[0]["paddle_speed_available"] + 0.04, len(f) - 0.6, "the arm's limit", color=T["orange"], fontsize=8.5)
    ax.set_yticks(range(len(f)), [row["table"] for row in f])
    ax.set_ylim(-0.6, len(f) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="paddle speed a return needs [m/s] (median, 5-95 %)", title="Why the rally is on a midi table")
    fig.tight_layout()
    save(fig, "rally-feasibility", theme)


def legend_below(fig, ax, ncol: int) -> None:
    """One legend under the whole figure, so it never sits on the curves."""
    h, lab = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=ncol, fontsize=8.2)


FIGURES = {f.__name__.replace("_", "-"): f for f in (pitches, tunnel, estimator, calibration, speed, wm_compare, pareto, when,
                                                       ppo, surprise_pareto, reaction, events, events_latency, rally_fig,
                                                       rally_feasibility)}


def main():
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        for theme in THEMES:
            style(theme)
            FIGURES[name](theme)
        print(f"  {name}")


if __name__ == "__main__":
    main()
