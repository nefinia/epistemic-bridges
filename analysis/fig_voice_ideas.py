"""Figures for voice and ideas, from results/paper/voice_content.csv (voice_content.py).

fig0_voice_ideas: left, every group by how much of the human map of ideas it reaches and how many of its
answers land on it; right, how much its voice repeats. Earlier version placed groups by ideas reached and
how repetitive its voice is. fig3_conditions: change from no persona, in ideas and in voice,
for each persona and creative condition.

    python analysis/fig_voice_ideas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import dots  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FOCUS, BASE = "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B"
CREATIVE = {"ordinary_life": "an ordinary life", "memory": "a memory", "story": "a story", "playful": "playful",
            "step_by_step": "step by step"}


def deltas(res: pd.DataFrame) -> pd.DataFrame:
    """Change from no persona for each condition and model: reasoning personas against the main
    run with no persona, creative conditions against the no-persona answers of the creative run."""
    rows = []
    for base, g in res.groupby("base"):
        for c in g["cond"].dropna():
            if c.startswith("T "):
                continue
            run, cond = c.split(":")
            if cond == "baseline":
                continue
            ref = g[g["cond"] == f"{run}:baseline"] if run == "creative" else g[g["cond"].isna()]
            if ref.empty:
                continue
            x, y = g[g["cond"] == c].iloc[0], ref.iloc[0]
            rows.append({"run": run, "cond": cond, "model": base, "d_ideas": x["map_reached_ideas"] - y["map_reached_ideas"],
                         "d_onmap": x["on_map_ideas"] - y["on_map_ideas"], "d_voice": x["cr_pos"] - y["cr_pos"]})
    return pd.DataFrame(rows)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, INK, MUTED, FAINT, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#b9b8b2", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "svg.fonttype": "none"})
    res = pd.read_csv(ROOT / "results" / "paper" / "voice_content.csv")
    res["base"] = res["model"].str.split(" | ", regex=False).str[0]
    res["cond"] = res["model"].str.split(" | ", regex=False).str[1]
    g = res.set_index("model")
    X, Y = "map_reached_ideas", "cr_pos"
    figs = ROOT / "paper" / "figs"

    # Figure 1: ideas (answers on the map against map reached) and voice, side by side
    X, Y = "map_reached_ideas", "on_map_ideas"
    fig, (ax, av) = plt.subplots(1, 2, figsize=(7.6, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})
    plain = res[res["cond"].isna() & (res["model"] != "people")]
    jk_path = ROOT / "results" / "paper" / "players_uncertainty.csv"
    jk = pd.read_csv(jk_path, index_col=0) if jk_path.exists() else None
    for kind, color, label in (("base", BLUE, "Base models"), ("chat", ORANGE, "Chat models")):
        sub = plain[plain["kind"] == kind]
        if jk is not None:
            for _, r in sub.iterrows():
                if r["model"] in jk.index:
                    ax.errorbar(r[X], r[Y], xerr=jk.loc[r["model"], "map_reached_ideas_jk"], fmt="none", ecolor=GRID,
                                elinewidth=0.8, zorder=1)
        dots(ax, sub[X], sub[Y], s=34, color=color, edgecolor="white", linewidth=1, zorder=3, label=label + ", no persona")
    for _, r in plain.iterrows():
        if r["name"] in ("GPT-2 XL", "GPT-J 6B", "Qwen3 8B", "Qwen3 32B", "gpt-oss 20B", "Qwen2.5 14B"):
            off, ha = ((-5, 5), "right") if r["name"] == "Qwen3 8B" else ((5, -3), "left")
            ax.annotate(r["name"], (r[X], r[Y]), xytext=off, textcoords="offset points", fontsize=6.5, color=MUTED, ha=ha)
    t = res[(res["base"] == FOCUS) & res["cond"].fillna("").str.startswith("T ")].copy()
    t["T"] = t["cond"].str[2:].astype(float)
    t = t.sort_values("T")
    ax.plot(t[X], t[Y], color=MUTED, linewidth=1.1, linestyle=(0, (3, 2)), zorder=2)
    dots(ax, t[X], t[Y], s=9, color=MUTED, zorder=3)
    hot = t.iloc[-1]
    ax.annotate("more randomness", (hot[X], hot[Y]), xytext=(-6, -2), textcoords="offset points", fontsize=7, color=MUTED, ha="right")

    def arrow(a, b, color):
        ax.annotate("", xy=(g.loc[b, X], g.loc[b, Y]), xytext=(g.loc[a, X], g.loc[a, Y]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, shrinkA=6, shrinkB=6, mutation_scale=10), zorder=4)

    arrow(BASE, FOCUS, ORANGE)
    key = f"{FOCUS} | creative:ordinary_life"
    arrow(FOCUS, key, AQUA)
    dots(ax, [g.loc[key, X]], [g.loc[key, Y]], s=40, color=AQUA, edgecolor="white", linewidth=1.2, zorder=5)
    ax.annotate("an ordinary life", (g.loc[key, X], g.loc[key, Y]), xytext=(-4, -12), textcoords="offset points", fontsize=7.5, color=INK, ha="right")
    dots(ax, [g.loc[BASE, X]], [g.loc[BASE, Y]], s=60, color=BLUE, edgecolor="white", linewidth=1.5, zorder=5)
    ax.annotate("Qwen2.5 7B base", (g.loc[BASE, X], g.loc[BASE, Y]), xytext=(6, -10), textcoords="offset points", fontsize=7.5, color=INK)
    dots(ax, [g.loc[FOCUS, X]], [g.loc[FOCUS, Y]], s=60, color=ORANGE, edgecolor="white", linewidth=1.5, zorder=5)
    ax.annotate("after chat training", (g.loc[FOCUS, X], g.loc[FOCUS, Y]), xytext=(-6, -13), textcoords="offset points",
                fontsize=7.5, color=INK, ha="right")
    p = g.loc["people"]
    dots(ax, [p[X]], [p[Y]], s=170, marker="*", color=INK, zorder=6)
    ax.annotate("people", (p[X], p[Y]), xytext=(6, 5), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("Share of the human map of ideas reached")
    ax.set_ylabel("Share of answers on the human map")
    ax.set_title("Ideas (voice removed)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    # voice: repetition of sentence patterns, one row per group
    v = pd.concat([res[res["model"] == "people"], plain]).sort_values("cr_pos")
    y = np.arange(len(v))
    cols = [INK if k == "people" else BLUE if k == "base" else ORANGE for k in v["kind"]]
    av.hlines(y, 3.2, v["cr_pos"], color=GRID, linewidth=1)
    dots(av, v["cr_pos"], y, color=cols, s=26, zorder=3)
    av.axvline(g.loc["people", "cr_pos"], color=INK, linewidth=0.8, linestyle=(0, (2, 2)))
    av.set_yticks(y)
    av.set_yticklabels(["People" if m == "people" else n.replace(" base", " (base)") for m, n in zip(v["model"], v["name"])], fontsize=6.5)
    av.set_xlim(3.2, 5.8)
    av.set_xlabel("Repetition of sentence patterns")
    av.set_title("Voice", fontsize=9, color=INK)
    av.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(figs / "fig0_voice_ideas.pdf"); fig.savefig(figs / "fig0_voice_ideas.svg")

    # Figure 2: change from no persona, ideas and voice
    d = deltas(res)
    d.to_csv(ROOT / "results" / "paper" / "voice_content_deltas.csv", index=False)
    s = d.groupby(["run", "cond"]).agg(ideas=("d_ideas", "mean"), lo=("d_ideas", "min"), hi=("d_ideas", "max"),
                                       voice=("d_voice", "mean"), vlo=("d_voice", "min"), vhi=("d_voice", "max")).reset_index()
    s = s.sort_values("ideas")
    y = np.arange(len(s))
    colors = [BLUE if r == "fer" else ORANGE for r in s["run"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 0.25 * len(s) + 1.1), sharey=True)
    for ax, col, lo, hi, title in ((axes[0], "ideas", "lo", "hi", "Ideas: change in places reached"),
                                   (axes[1], "voice", "vlo", "vhi", "Voice: change in repetition")):
        ax.hlines(y, s[lo], s[hi], color=GRID, linewidth=3)
        dots(ax, s[col], y, color=colors, s=28, zorder=3, edgecolor="white", linewidth=1)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("\u2190 narrower ideas    wider ideas \u2192", fontsize=7.5, color=MUTED)
    axes[1].set_xlabel("\u2190 more varied voice    more repetitive \u2192", fontsize=7.5, color=MUTED)
    axes[0].set_yticks(y); axes[0].set_yticklabels(s["cond"].str.replace("_", " ").str.replace(" siev", "").str.replace(" tmk", ""))
    dots(axes[1], [], [], color=BLUE, label="Reasoning (FER)"); dots(axes[1], [], [], color=ORANGE, label="Creative")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    
    fig.savefig(figs / "fig3_conditions.pdf"); fig.savefig(figs / "fig3_conditions.svg")


if __name__ == "__main__":
    main()
