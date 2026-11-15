"""Dots drawn as plain filled shapes, so every PDF viewer shows their colors.

matplotlib writes markers (from scatter and from plot) as shared PDF objects whose fill
color is set outside them, and some viewers (Adobe on phones among them) paint those
black. Here each dot is its own filled path with its color written next to it.
"""

import numpy as np
from matplotlib.markers import MarkerStyle
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D, ScaledTranslation


def dots(ax, x, y, s=36, color=None, edgecolor="none", linewidth=0, marker="o", alpha=1.0, zorder=2, label=None):
    x, y = np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
    colors = list(color) if isinstance(color, (list, tuple, np.ndarray)) else [color] * len(x)
    markers = list(marker) if isinstance(marker, (list, tuple, np.ndarray)) else [marker] * len(x)
    size = float(np.sqrt(s))  # marker size in points, as in plot(markersize=...)
    if label is not None:  # an empty line gives the legend its entry
        ax.plot([], [], linestyle="none", marker=markers[0] if markers else marker, markersize=size, color=colors[0] if colors else color,
                markeredgecolor=edgecolor, markeredgewidth=linewidth, label=label)
    if len(x) == 0:
        return
    ax.plot(x, y, linestyle="none", marker=None)  # keeps the axis limits around the dots, draws nothing
    def path_of(m):
        ms = MarkerStyle(m)
        return ms.get_path().transformed(ms.get_transform())
    shape = Affine2D().scale(size / 72) + ax.figure.dpi_scale_trans
    for xi, yi, c, m in zip(x, y, colors, markers):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        ax.add_artist(PathPatch(path_of(m), transform=shape + ScaledTranslation(xi, yi, ax.transData), facecolor=c,
                                edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder))
