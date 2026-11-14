"""Shared matplotlib style, light and dark.

Colours are the reference palette of the dataviz guidance (categorical slots in
their validated order, each with its own step for the dark surface).  Colour
follows the *entity*, never its rank, and is the same in every figure:

    oracle (MuJoCo in the loop)        -> neutral ink (the reference, not a competitor)
    prior (classical, no learning)     -> blue    (slot 1)
    gnn (pure learning, nodes)         -> orange  (slot 2)
    structured (hybrid: prior + net)   -> aqua    (slot 3)
    unet (pure learning, height map)   -> yellow  (slot 4)
    fallback (hybrid + oracle checks)  -> violet  (slot 7); random trigger magenta, ranking trigger red

Orange and yellow are the weak pair of this palette, and three slots have < 3:1
contrast on the light surface, so every series also has its own marker and line
style, figures carry a legend or direct labels, and the numbers are repeated in a
table in the results markdown.

Height maps use the sequential blue ramp (deeper = darker); signed differences use
the diverging blue-red pair with a neutral grey midpoint.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

THEMES = {
    "light": dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", violet="#6250d6",
                  red="#e34948", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  surface="#fcfcfb", mid="#f0efec"),
    "dark": dict(blue="#3987e5", orange="#d95926", aqua="#199e70", yellow="#c98500", magenta="#d55181", violet="#9085e9",
                 red="#e66767", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 surface="#1a1a19", mid="#383835"),
}
SLOT = dict(oracle="ink2", prior="blue", gnn="orange", structured="aqua", unet="yellow", fallback="violet", random="magenta", regret="red")
MARKER = dict(oracle="D", prior="s", gnn="^", structured="o", unet="v", fallback="P", random="X", regret="*")
DASH = dict(oracle="-", prior="-", gnn="-", structured="-", unet=(0, (5, 2)), fallback="-", random=(0, (2, 2)), regret=(0, (6, 2, 1, 2)))
LABEL = dict(oracle="oracle (MuJoCo)", prior="prior only", gnn="GNN", structured="structured (prior + GNN)",
             unet="height-map U-Net", fallback="structured + oracle fallback", random="random fallback",
             regret="structured + ranking fallback")
_theme = "light"


def apply(theme: str = "light", transparent: bool = False) -> dict:
    """Set rcParams for ``theme`` and return its colour table."""
    global _theme
    _theme = theme
    c = THEMES[theme]
    face = "none" if transparent else c["surface"]
    plt.rcParams.update({
        "figure.facecolor": face, "axes.facecolor": face, "savefig.facecolor": face,
        "axes.edgecolor": c["axis"], "axes.linewidth": 0.8, "axes.grid": True, "grid.color": c["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.labelcolor": c["ink2"], "axes.titlecolor": c["ink"], "text.color": c["ink"], "xtick.color": c["muted"],
        "ytick.color": c["muted"], "xtick.labelcolor": c["ink2"], "ytick.labelcolor": c["ink2"], "axes.titlesize": 10.5,
        "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.markersize": 6.0, "font.family": "DejaVu Sans", "figure.dpi": 110,
        "savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlelocation": "left",
        "svg.fonttype": "none",
    })
    return c


def colors() -> dict:
    return THEMES[_theme]


def method(name: str) -> dict:
    """Line/marker keyword arguments of a method (colour, marker, dash, label)."""
    c = colors()
    return dict(color=c[SLOT[name]], marker=MARKER[name], linestyle=DASH[name], label=LABEL[name],
                markeredgecolor=c["surface"], markeredgewidth=1.0)


def depth_cmap() -> LinearSegmentedColormap:
    """Sequential blue ramp: 0 (untouched) light, deep dark.  On the dark theme the ramp is reversed."""
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    if _theme == "dark":
        ramp = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
    return LinearSegmentedColormap.from_list("depth", ramp)


def diff_cmap() -> LinearSegmentedColormap:
    """Diverging blue-red with a neutral midpoint."""
    c = colors()
    return LinearSegmentedColormap.from_list("diff", ["#184f95", "#3987e5", c["mid"], "#e34948", "#a32a2a"])


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
