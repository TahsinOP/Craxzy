"""Figures of the documentation site, one transparent SVG per theme.

    env -u PYTHONPATH .venv/bin/python docs/figures/make_figures.py [name ...]

Reads results/*.json and results/*.npz (written by the milestone scripts) and
draws with the same functions as the PNGs in results/ (clay_sculptor.eval.plots),
so a number in a figure here is the number in the corresponding report.
Output: docs/public/fig/<name>-{light,dark}.svg, plus the renders as PNG.
"""
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
from clay_sculptor import RESULTS, config  # noqa: E402
from clay_sculptor.envs.shape import Geometry  # noqa: E402
from clay_sculptor.envs.clay_env import quiet_build  # noqa: E402
from clay_sculptor.eval import plots  # noqa: E402
from clay_sculptor.eval import plotstyle as ps  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "public" / "fig"
cfg = config.load()
geom = Geometry(quiet_build("press", cfg.clay, cfg.sim), cfg.clay, cfg.map)
J = lambda name: json.loads((RESULTS / f"{name}.json").read_text())
Z = lambda name: dict(np.load(RESULTS / f"{name}.npz"))
hm = lambda nodes: geom.heightmap(np.asarray(nodes, dtype=np.float64))


def m0_shapes(path):
    z = Z("m0_shapes")
    plots.sequence_maps(geom, hm(z["untouched"]), {k: hm(z[k]) for k in ("press", "roll", "stamp")}, path)


def m0_permanence(path):
    z = Z("m0_shapes")
    t = np.arange(len(z["trace_plastic"])) * cfg.sim.timestep
    plots.permanence(t, z["trace_plastic"], z["trace_elastic"], cfg.prim.nsteps("press", cfg.sim.timestep) * cfg.sim.timestep, cfg, path)


def m0_discretisation(path):
    plots.discretisation(J("m0_discretisation"), path)


def m1_targets(path):
    r, z = J("m1_oracle"), Z("m1_oracle_shapes")
    plots.targets_figure(geom, hm(Z("m0_shapes")["untouched"]), z, list(r["targets"]), r["seeds"], path, "oracle MPC")


def m1_curves(path):
    r = J("m1_oracle")
    plots.cost_curves({n: [e["costs_mm"] for e in t["episodes"]] for n, t in r["targets"].items()}, path)


def m2_errors(path):
    plots.m2_errors(J("m2_eval"), path)


def m2_examples(path):
    plots.m2_examples(Z("m2_examples"), float(cfg.clay.top), path)


def m2_shift(path):
    plots.m2_shift(J("m2_eval"), path)


def m3_tradeoff(path):
    r = J("m3_mpc")
    plots.m3_tradeoff(r, [t for t in r["methods"]["structured"] if t != "overall"], path)


def m3_shapes(path):
    r = J("m3_mpc")
    targets = [t for t in r["methods"]["structured"] if t != "overall"]
    plots.m3_shapes(geom, Z("m3_mpc_shapes"), targets, [k for k in ("prior", "gnn", "structured", "unet") if k in r["methods"]],
                    r["seeds"][0], path, RESULTS / "m1_oracle_shapes.npz")


def m4_pareto(path):
    plots.m4_pareto(J("m4_fallback"), path)


def m4_calibration(path):
    z = Z("m4_fallback_calibration")
    plots.m4_calibration(z["spread"], z["err"], J("m4_fallback")["calibration"], path)


def stretch(path):
    r, z = J("stretch"), Z("stretch_shapes")
    done = [p for p in ("structured", "fallback", "oracle") if all(p in t["planners"] for t in r["targets"].values())]
    plots.stretch_figure(geom, z, list(r["targets"]), done, r["seeds"][0], path)


FIGS = {
    "m0-shapes": m0_shapes, "m0-permanence": m0_permanence, "m0-discretisation": m0_discretisation,
    "m1-targets": m1_targets, "m1-curves": m1_curves,
    "m2-errors": m2_errors, "m2-examples": m2_examples, "m2-shift": m2_shift,
    "m3-tradeoff": m3_tradeoff, "m3-shapes": m3_shapes,
    "m4-pareto": m4_pareto, "m4-calibration": m4_calibration, "stretch": stretch,
}
RENDERS = ["m0_render.png", "targets_render.png"]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(FIGS)
    for name in names:
        for theme in ("light", "dark"):
            ps.apply(theme, transparent=True)
            try:
                FIGS[name](OUT / f"{name}-{theme}.svg")
            except FileNotFoundError as e:
                print(f"skip {name}: {e.filename} is missing")
                break
        else:
            print("wrote", name)
    for r in RENDERS:
        if (RESULTS / r).exists():
            shutil.copy(RESULTS / r, OUT / r)
    ps.apply("light")
