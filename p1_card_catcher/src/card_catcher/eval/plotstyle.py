"""Shared matplotlib style.

Colours are the validated categorical palette of the dataviz guidance (light
surface; `validate_palette.js` passes lightness band, chroma floor, CVD
separation and normal-vision floor for slots 1-4).  Colour follows the *entity*,
never its rank, and is the same in every figure of the project:

    classical / physics-only baseline -> blue
    pure-learning baseline            -> orange
    hybrid (physics + learned)        -> aqua
    a fourth series                   -> yellow

Two of the slots have < 3:1 contrast on the surface, so every figure carries a
legend and the numbers are repeated in a table in the results markdown.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

METHOD = {"physics": BLUE, "pure": ORANGE, "hybrid": AQUA, "extra": YELLOW}
OBJECT = {"ball": BLUE, "plate": ORANGE, "card": AQUA}


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.markersize": 5.0, "font.family": "DejaVu Sans", "figure.dpi": 110,
        "savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
