"""Figures.  Every function takes the output path last and draws with the current
``plotstyle`` theme, so the same code makes the PNGs of ``results/`` (light) and
the light/dark SVG pairs of the documentation site."""
from __future__ import annotations

import numpy as np

from clay_sculptor.eval import plotstyle as ps
from clay_sculptor.eval.plotstyle import plt

MM = 1e3
ps.apply()          # light theme by default; the docs figure script switches themes explicitly


def volume(geom, nodes: np.ndarray) -> float:
    """Volume of the cage [m^3]: every hexahedral cell split into six tetrahedra
    around its main diagonal (a diagnostic; exact when the cell faces are planar)."""
    n = np.asarray(nodes, dtype=float)
    c = lambda dx, dy, dz: n[dx:n.shape[0] - 1 + dx, dy:n.shape[1] - 1 + dy, dz:n.shape[2] - 1 + dz]
    p = {k: c(*k) for k in np.ndindex(2, 2, 2)}
    a, g = p[(0, 0, 0)], p[(1, 1, 1)]
    ring = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1)]
    vol = 0.0
    for i in range(6):
        b, d = p[ring[i]], p[ring[(i + 1) % 6]]
        vol = vol + np.einsum("...i,...i->...", np.cross(b - a, d - a), g - a) / 6.0
    return float(np.abs(vol).sum())


def _map_axes(ax, title: str | None = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=9.5)


def show_depth(ax, h: np.ndarray, top: float, vmax: float, title: str | None = None):
    """Depth below the undeformed top [mm]; rows = x (away from the robot upwards), columns = y."""
    im = ax.imshow(MM * (top - h), origin="lower", cmap=ps.depth_cmap(), vmin=0, vmax=vmax, interpolation="nearest")
    _map_axes(ax, title)
    return im


def show_diff(ax, d: np.ndarray, vmax: float, title: str | None = None):
    im = ax.imshow(MM * d, origin="lower", cmap=ps.diff_cmap(), vmin=-vmax, vmax=vmax, interpolation="nearest")
    _map_axes(ax, title)
    return im


def maps_row(maps: list, top: float, path, vmax: float | None = None, suptitle: str | None = None, ncol: int | None = None,
             label: str = "depth below the undeformed top [mm]") -> None:
    """A row (or grid) of depth maps sharing one colour scale."""
    vmax = vmax or max(1.0, max(float(MM * (top - h).max()) for _, h in maps))
    ncol = ncol or len(maps)
    nrow = int(np.ceil(len(maps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol + 0.7, 2.2 * nrow + (0.3 if suptitle else 0)), squeeze=False, layout="constrained")
    im = None
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, h) in zip(axes.ravel(), maps):
        ax.axis("on")
        im = show_depth(ax, h, top, vmax, title)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label(label, fontsize=8.5)
    cb.outline.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, x=0.02, ha="left", fontsize=10.5, fontweight="bold")
    ps.save(fig, path)


def sequence_maps(geom, h0: np.ndarray, finals: dict, path) -> None:
    top = float(h0.max())
    maps_row([("untouched", h0)] + [(f"after {k}", v) for k, v in finals.items()], top, path,
             suptitle="Scripted sequences: top-surface height maps")


def permanence(t: np.ndarray, plastic: np.ndarray, elastic: np.ndarray, t_prim: float, cfg, path) -> None:
    c = ps.colors()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(t, MM * (plastic[0] - plastic), color=c["aqua"], label="plastic (yield strain %.2g)" % cfg.clay.yield_strain)
    ax.plot(t, MM * (elastic[0] - elastic), color=c["blue"], linestyle=(0, (5, 2)), label="elastic (yield off)")
    ax.axvspan(0, cfg.prim.t_down + cfg.prim.t_dwell, color=c["grid"], alpha=0.6, lw=0)
    ax.axvline(t_prim, color=c["muted"], lw=0.8)
    ymax = MM * (plastic[0] - plastic).max()
    ax.text((cfg.prim.t_down + cfg.prim.t_dwell) / 2, ymax * 1.04, "tool pressing", ha="center", fontsize=8, color=c["ink2"])
    ax.text(t_prim + 0.03, ymax * 1.04, "tool away, primitive over", fontsize=8, color=c["ink2"])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("depth at the press centre [mm]")
    ax.set_ylim(-0.3, ymax * 1.15)
    ax.set_title("A pressed dent stays in plastic clay and vanishes in elastic clay")
    ax.legend(loc="center right")
    ps.save(fig, path)


def targets_figure(geom, h0: np.ndarray, shapes: dict, targets, seeds, path, who: str) -> None:
    """Per target: target map, the map reached (first seed) and their signed difference."""
    top = float(h0.max())
    fig, axes = plt.subplots(len(targets), 3, figsize=(7.4, 2.25 * len(targets) + 0.3), squeeze=False, layout="constrained")
    im = im2 = None
    for r, name in enumerate(targets):
        ht = geom.heightmap(shapes[f"{name}_target"].astype(np.float64))
        hr = geom.heightmap(shapes[f"{name}_seed{seeds[0]}"].astype(np.float64))
        vmax = max(1.0, float(MM * (top - ht).max()))
        im = show_depth(axes[r, 0], ht, top, 12.0, f"{name}: target")
        show_depth(axes[r, 1], hr, top, 12.0, f"reached ({who})")
        im2 = show_diff(axes[r, 2], hr - ht, 3.0, "reached - target")
    cb = fig.colorbar(im, ax=axes[:, :2], fraction=0.035, pad=0.03)
    cb.set_label("depth below the undeformed top [mm]", fontsize=8.5)
    cb.outline.set_visible(False)
    cb2 = fig.colorbar(im2, ax=axes[:, 2], fraction=0.08, pad=0.05)
    cb2.set_label("height difference [mm]", fontsize=8.5)
    cb2.outline.set_visible(False)
    ps.save(fig, path)


def cost_curves(curves: dict, path, title: str = "Oracle MPC: error after each action") -> None:
    """Height-map RMSE against the number of executed actions, one panel per target, one line per seed."""
    c = ps.colors()
    fig, axes = plt.subplots(1, len(curves), figsize=(3.1 * len(curves), 2.9), squeeze=False)
    for ax, (name, runs) in zip(axes[0], curves.items()):
        for run in runs:
            ax.plot(range(len(run)), run, color=c["ink2"], marker="D", markersize=4.5, markeredgecolor=c["surface"], lw=1.6, alpha=0.85)
        ax.set_title(name)
        ax.set_xlabel("actions executed")
        ax.set_ylim(bottom=0)
        ax.xaxis.get_major_locator().set_params(integer=True)
    axes[0, 0].set_ylabel("height-map RMSE to target [mm]")
    fig.suptitle(title, x=0.02, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    ps.save(fig, path)


# ---- M2 ------------------------------------------------------------------------------------
M2_ORDER = ("prior", "gnn", "structured", "unet")


def m2_errors(res: dict, path) -> None:
    """Left: one-step height-map RMSE per model.  Right: open-loop error against the horizon."""
    c = ps.colors()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.3), gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    names = ["no_change"] + [k for k in M2_ORDER if k in res["one_step"]]
    vals = [res["one_step"][k]["map_rmse_mm"] for k in names]
    cols = [c["muted"]] + [c[ps.SLOT[k]] for k in names[1:]]
    labels = ["no change"] + [ps.LABEL[k].replace(" (prior + GNN)", "") for k in names[1:]]
    y = np.arange(len(names))[::-1]
    a.barh(y, [v[0] for v in vals], xerr=[v[1] for v in vals], color=cols, height=0.62, error_kw=dict(ecolor=c["ink2"], lw=1, capsize=2))
    for yi, v in zip(y, vals):
        a.text(v[0] + v[1] + 0.02 * max(x[0] for x in vals), yi, f"{v[0]:.2f}", va="center", fontsize=8.5, color=c["ink"])
    a.set_yticks(y, labels)
    a.grid(axis="y", visible=False)
    a.set_xlabel("one-step height-map RMSE [mm]")
    a.set_title("One primitive ahead")
    h = np.arange(1, len(res["multi_step"]["no_change"]) + 1)
    b.plot(h, [v[0] for v in res["multi_step"]["no_change"]], color=c["muted"], marker="x", linestyle=(0, (1, 2)), label="no change")
    for k in M2_ORDER:
        if k not in res["multi_step"]:
            continue
        m, s = np.array([v[0] for v in res["multi_step"][k]]), np.array([v[1] for v in res["multi_step"][k]])
        kw = ps.method(k)
        b.plot(h, m, **kw)
        b.fill_between(h, m - s, m + s, color=kw["color"], alpha=0.18, lw=0)
    b.set_xlabel("primitives predicted open loop")
    b.set_ylabel("height-map RMSE [mm]")
    b.set_title("Fed its own predictions")
    b.set_ylim(bottom=0)
    b.legend(loc="upper left", ncol=1)
    fig.tight_layout()
    ps.save(fig, path)


def m2_examples(ex: dict, top: float, path, n: int = 4) -> None:
    """Held-out transitions: shape before, true shape after and each model's prediction."""
    cols = [("before", "before"), ("after", "true after")] + [(k, ps.LABEL[k].replace(" (prior + GNN)", "").replace("height-map ", ""))
                                                                for k in M2_ORDER if k in ex]
    tools = ("press", "roller", "stamp")
    fig, axes = plt.subplots(n, len(cols), figsize=(1.55 * len(cols) + 0.8, 1.6 * n + 0.2), squeeze=False, layout="constrained")
    im = None
    for r in range(n):
        for cidx, (key, title) in enumerate(cols):
            im = show_depth(axes[r, cidx], ex[key][r], top, 14.0, title if r == 0 else None)
        axes[r, 0].set_ylabel(tools[int(ex["tool"][r])], fontsize=9)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("depth below the undeformed top [mm]", fontsize=8.5)
    cb.outline.set_visible(False)
    ps.save(fig, path)


def m2_shift(res: dict, path) -> None:
    """One-step error on clay whose material differs from the training clay."""
    c = ps.colors()
    names = [n for n in ("young_lo", "yield_lo", "nominal", "yield_hi", "young_hi") if n == "nominal" or n in res["shift"]]
    pretty = dict(young_lo="Young's\nmodulus x0.5", young_hi="Young's\nmodulus x2", yield_lo="yield\nstrain x0.5",
                  yield_hi="yield\nstrain x2", nominal="training\nmaterial")
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    kinds = ["no_change"] + [k for k in M2_ORDER if k in res["one_step"]]
    w = 0.8 / len(kinds)
    for i, k in enumerate(kinds):
        vals = [(res["one_step"][k]["map_rmse_mm"] if n == "nominal" else res["shift"][n][k]) for n in names]
        col = c["muted"] if k == "no_change" else c[ps.SLOT[k]]
        lab = "no change" if k == "no_change" else ps.LABEL[k]
        ax.bar(np.arange(len(names)) + (i - (len(kinds) - 1) / 2) * w, [v[0] for v in vals], width=w * 0.86, color=col, label=lab,
               yerr=[v[1] for v in vals], error_kw=dict(ecolor=c["ink2"], lw=0.8, capsize=1.5))
    ax.set_xticks(np.arange(len(names)), [pretty[n] for n in names])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("one-step height-map RMSE [mm]")
    ax.set_title("Models trained on one clay, tested on others")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ps.save(fig, path)


# ---- M3 ------------------------------------------------------------------------------------
M3_ORDER = ("oracle", "prior", "gnn", "structured", "unet")


def m3_tradeoff(res: dict, targets, path) -> None:
    """Shape error left after MPC against planning time per decision, one panel per target and one overall."""
    panels = list(targets) + ["all targets"]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2), squeeze=False)
    c = ps.colors()
    for ax, name in zip(axes[0], panels):
        for k in M3_ORDER:
            if k not in res["methods"]:
                continue
            r = res["methods"][k]
            if name == "all targets":
                y, t = r["overall"]["remaining"], r["overall"]["plan_time_per_decision_s"]
                y = (100 * y[0], 100 * y[1])
            else:
                y, t = r[name]["summary"]["map_rmse_mm"], r[name]["summary"]["plan_time_per_decision_s"]
            kw = ps.method(k)
            kw["linestyle"] = "none"
            ax.errorbar(t[0], y[0], yerr=y[1], ecolor=kw["color"], elinewidth=1.2, capsize=2, markersize=8, **kw)
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("planning time per decision [s]")
        ax.set_ylim(bottom=0)
    axes[0, 0].set_ylabel("height-map RMSE to target [mm]")
    axes[0, -1].set_ylabel("initial error left [%]")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    ps.save(fig, path)


def m3_shapes(geom, shapes: dict, targets, kinds, seed: int, path, oracle_npz=None) -> None:
    """Target and the shape each planner reached (first seed), one row per target."""
    cols = [("target", "target")]
    oracle = None
    if oracle_npz is not None and oracle_npz.exists():
        oracle = np.load(oracle_npz)
        cols.append(("oracle", "oracle"))
    cols += [(k, ps.LABEL[k].replace(" (prior + GNN)", "").replace("height-map ", "")) for k in kinds]
    top = geom.clay.top
    fig, axes = plt.subplots(len(targets), len(cols), figsize=(1.6 * len(cols) + 0.8, 1.7 * len(targets) + 0.2), squeeze=False, layout="constrained")
    im = None
    for r, name in enumerate(targets):
        for cidx, (key, title) in enumerate(cols):
            if key == "target":
                n = shapes[f"{name}_target"]
            elif key == "oracle":
                n = oracle[f"{name}_seed{seed}"]
            else:
                n = shapes[f"{key}_{name}_seed{seed}"]
            im = show_depth(axes[r, cidx], geom.heightmap(n.astype(np.float64)), top, 12.0, title if r == 0 else None)
        axes[r, 0].set_ylabel(name, fontsize=9)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("depth below the undeformed top [mm]", fontsize=8.5)
    cb.outline.set_visible(False)
    ps.save(fig, path)


# ---- M4 ------------------------------------------------------------------------------------
def m4_pareto(res: dict, path) -> None:
    """Paired difference to the pure learned planner ('never') against oracle calls per episode.

    Each episode is paired with the 'never' episode of the same target and planner seed; the bars are
    95 % t-intervals of the mean difference. Seed-to-seed spreads (several points) would hide differences
    the pairing resolves, so they are not drawn here (they are in results/M4.md)."""
    from scipy import stats
    c = ps.colors()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    cfgs = res["configs"]

    def rem(k):
        ep = cfgs[k]["episodes"]
        return np.array([e["remaining"] for t in ep for e in sorted(ep[t], key=lambda e: e["seed"])])

    def diff(k):
        d = 100 * (rem(k) - rem("never"))
        return d.mean(), stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))

    ax.axhline(0, color=c["ink2"], lw=0.8)
    ax.annotate("never asks (pure learned)", (0, 0), textcoords="offset points", xytext=(4, 5), fontsize=8, color=c["ink2"])
    labels = dict(random="re-rank at random decisions", fallback="re-rank when the members' maps disagree",
                  regret="re-rank when the members doubt the ranking")
    for prefix, name in (("random", "random"), ("disagreement", "fallback"), ("misrank", "regret")):
        ks = sorted((k for k in cfgs if k.startswith(prefix)), key=lambda k: cfgs[k]["summary"]["calls_per_episode"][0])
        if not ks:
            continue
        x = np.array([cfgs[k]["summary"]["calls_per_episode"][0] for k in ks])
        y, e = np.array([diff(k) for k in ks]).T
        kw = ps.method(name)
        kw["label"] = labels[name]
        ax.errorbar(x, y, yerr=e, ecolor=kw["color"], elinewidth=1.0, capsize=2, **kw)
    if "always" in cfgs:
        kw = ps.method("fallback")
        kw.update(linestyle="none", label=None, marker="s", color=c["ink"])
        x = cfgs["always"]["summary"]["calls_per_episode"][0]
        y, e = diff("always")
        ax.errorbar(x, y, yerr=e, ecolor=c["ink"], elinewidth=1.0, capsize=2, markersize=6, **kw)
        ax.annotate("always", (x, y), textcoords="offset points", xytext=(6, -12), fontsize=8, color=c["ink2"])
    ax.set_xlim(-1, 34)
    ax.set_xlabel("oracle rollouts used for planning, per episode")
    ax.set_ylabel("initial error left, minus never [points]")
    ax.set_title("Paired gain of asking the simulator")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    ps.save(fig, path)


def m4_calibration(spread: np.ndarray, err: np.ndarray, cal: dict, path) -> None:
    """Ensemble spread against the true one-step error on held-out transitions."""
    c = ps.colors()
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.scatter(spread, err, s=9, color=c["aqua"], alpha=0.35, lw=0)
    ax.plot(cal["spread_mm"], cal["error_mm"], color=c["ink"], marker="o", markersize=5, lw=1.6, label="mean error in 8 equal-count bins")
    ax.set_xlabel("ensemble spread [mm]")
    ax.set_ylabel("true error of the ensemble mean [mm]")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(f"Disagreement tracks error (Spearman {cal['spearman']:.2f})")
    ax.legend(loc="upper left")
    ps.save(fig, path)


def stretch_figure(geom, shapes: dict, targets, planners, seed: int, path) -> None:
    """Stretch targets: the target and what each planner reached (first seed)."""
    names = dict(structured="structured", fallback="structured + fallback", oracle="oracle")
    top = geom.clay.top
    fig, axes = plt.subplots(len(targets), 1 + len(planners), figsize=(1.9 * (1 + len(planners)) + 0.8, 2.0 * len(targets) + 0.2),
                             squeeze=False, layout="constrained")
    im = None
    for r, t in enumerate(targets):
        im = show_depth(axes[r, 0], geom.heightmap(shapes[f"{t}_target"].astype(np.float64)), top, 12.0, "target" if r == 0 else None)
        axes[r, 0].set_ylabel(dict(t="letter T", combo="three tools").get(t, t), fontsize=9)
        for k, p in enumerate(planners):
            show_depth(axes[r, k + 1], geom.heightmap(shapes[f"{p}_{t}_seed{seed}"].astype(np.float64)), top, 12.0, names[p] if r == 0 else None)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("depth below the undeformed top [mm]", fontsize=8.5)
    cb.outline.set_visible(False)
    ps.save(fig, path)


def discretisation(rows: list, path) -> None:
    """Design check of M0: share of the stroke that stays, and creep after unloading, per discretisation."""
    c = ps.colors()
    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])
    labels = [l for l in labels if "yield strain" not in l]
    depths = sorted({r["depth_cmd_mm"] for r in rows})
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.1))
    y = np.arange(len(labels))[::-1]
    hgt = 0.36
    for k, (dep, col) in enumerate(zip(depths, (c["blue"], c["orange"]))):
        sel = {r["label"]: r for r in rows if r["depth_cmd_mm"] == dep}
        a.barh(y + (0.5 - k) * hgt, [100 * sel[l]["retained"] for l in labels], height=hgt * 0.9, color=col, label=f"{dep:.0f} mm press")
        creep = [max(sel[l]["creep_mm"], 1e-4) for l in labels]
        b.barh(y + (0.5 - k) * hgt, creep, height=hgt * 0.9, color=col)
        for yi, l in zip(y + (0.5 - k) * hgt, labels):
            if sel[l]["inverted"]:
                b.text(max(sel[l]["creep_mm"], 1e-4) * 1.25, yi, f"{sel[l]['inverted']} inverted", va="center", fontsize=8, color=c["ink"])
    a.set_yticks(y, labels)
    b.set_yticks(y, [""] * len(labels))
    a.set_xlabel("share of the stroke left as a dent [%]")
    b.set_xscale("log")
    b.set_xlabel("largest vertex motion, 1.5 s after the primitive [mm]")
    a.set_title("How much of a press stays")
    b.set_title("Does the clay come to rest?")
    for ax in (a, b):
        ax.grid(axis="y", visible=False)
    a.set_xlim(0, 105)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(w_pad=2.5)
    ps.save(fig, path)
