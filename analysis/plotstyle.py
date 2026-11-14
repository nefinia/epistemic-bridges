"""Dots drawn as plain line markers, so every PDF viewer shows their colors.

matplotlib's scatter writes markers as shared PDF objects, and some viewers (Adobe on
phones among them) paint those black. Line markers carry their color with each point.
"""

import numpy as np


def dots(ax, x, y, s=36, color=None, edgecolor="none", linewidth=0, marker="o", alpha=1.0, zorder=2, label=None):
    x, y = np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
    colors = color if isinstance(color, (list, tuple, np.ndarray)) else [color] * len(x)
    size = float(np.sqrt(s))
    if len(x) == 0:
        ax.plot([], [], linestyle="none", marker=marker, markersize=size, color=color, label=label)
        return
    first = True
    for c in dict.fromkeys(colors):
        m = np.array([cc == c for cc in colors])
        ax.plot(x[m], y[m], linestyle="none", marker=marker, markersize=size, color=c, markerfacecolor=c,
                markeredgecolor=edgecolor, markeredgewidth=linewidth, alpha=alpha, zorder=zorder,
                label=label if first else None)
        first = False
