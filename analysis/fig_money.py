"""Figure 1: every model with no persona on the precision-recall plane of the map of ideas.

Recall and precision come from voice_content.py (voice removed, content words), the voice
(repetition of sentence patterns) is the color, and people are the crosshair, with one
standard error over players as a band. The corner beyond people on both counts is shaded.

    python analysis/fig_money.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import dots  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
X, Y, V = "map_reached_ideas", "on_map_ideas", "cr_pos"


def draw(ax, fig, plain, p, jk, label_size=6.8):
    cmap = LinearSegmentedColormap.from_list("voice", ["#f7d9c4", "#eb6834", "#7a2408"])
    norm = Normalize(vmin=3.6, vmax=6.0)
    ex, ey = jk.loc["people", X + "_jk"], jk.loc["people", Y + "_jk"]
    ax.axvspan(p[X] - ex, p[X] + ex, color=GRID, alpha=0.6, zorder=0, lw=0)
    ax.axhspan(p[Y] - ey, p[Y] + ey, color=GRID, alpha=0.6, zorder=0, lw=0)
    ax.axvline(p[X], color=INK, lw=0.8, zorder=1)
    ax.axhline(p[Y], color=INK, lw=0.8, zorder=1)
    # Label offsets in points, placed by hand so that no label covers a dot or another label.
    OFF = {"Qwen3 32B": (6, 4, "left"), "Qwen3 8B": (0, -13, "center"), "Qwen2.5 14B": (7, 4, "left"),
           "Qwen2.5 7B": (-7, -6, "right"), "Vicuna 7B": (7, 7, "left"), "Zephyr 7B": (7, -1, "left"),
           "Qwen2.5 3B": (-7, -9, "right"), "gpt-oss 20B": (7, -9, "left"), "OLMo 2 7B": (0, -13, "center"),
           "Qwen2.5 7B base": (-8, -6, "right"), "Phi-3 mini": (0, 8, "center"), "Qwen2.5 1.5B": (-7, -4, "right"),
           "GPT-2 XL": (7, -4, "left"), "GPT-J 6B": (7, -4, "left"), "Qwen2.5 0.5B": (-7, -4, "right")}
    for _, r in plain.iterrows():
        dots(ax, [r[X]], [r[Y]], s=60, marker="s" if r["kind"] == "base" else "o", color=to_hex(cmap(norm(r[V]))),
             edgecolor=INK, linewidth=0.5, zorder=3)
        dx, dy, ha = OFF.get(r["name"], (7, -4, "left"))
        ax.annotate(f"{r['name'].replace(' base', ' (base)')}, {r['year']}", (r[X], r[Y]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=label_size, color=MUTED, zorder=5)
    dots(ax, [p[X]], [p[Y]], s=220, marker="*", color=to_hex(cmap(norm(p[V]))), edgecolor=INK, linewidth=0.7, zorder=4)
    ax.annotate("people", (p[X], p[Y]), xytext=(0, 12), textcoords="offset points", ha="center", va="center",
                fontsize=label_size + 1.7, color=INK, zorder=5)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Recall: share of the human map of ideas reached")
    ax.set_ylabel("Precision: share of answers on the human map")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Voice: repetition of sentence patterns", fontsize=8)
    cb.set_ticks([4.0, 4.5, 5.0, 5.5, 6.0])
    cb.ax.tick_params(labelsize=7)
    ax.legend(handles=[Line2D([], [], ls="none", marker="s", color=MUTED, label="base model"),
                       Line2D([], [], ls="none", marker="o", color=MUTED, label="chat model")],
              frameon=False, fontsize=7.5, loc="lower left")


def main():
    res = pd.read_csv(ROOT / "results" / "paper" / "voice_content.csv")
    jk = pd.read_csv(ROOT / "results" / "paper" / "players_uncertainty.csv", index_col=0)
    meta = pd.read_csv(ROOT / "results" / "paper" / "models_baseline.csv").set_index("model")
    plain = res[~res["model"].str.contains("|", regex=False) & (res["model"] != "people")].copy()
    plain["year"] = [int(meta.loc[m, "year"]) for m in plain["model"]]
    p = res.set_index("model").loc["people"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "svg.fonttype": "none"})
    figs = ROOT / "paper" / "figs"
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    draw(ax, fig, plain, p, jk)
    fig.tight_layout()
    fig.savefig(figs / "fig1_money.pdf"); fig.savefig(figs / "fig1_money.svg"); fig.savefig(figs / "fig1_money.png", dpi=150)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    draw(ax, fig, plain, p, jk, label_size=6.5)
    fig.tight_layout()
    fig.savefig(figs / "web" / "fig1.svg")


if __name__ == "__main__":
    main()
