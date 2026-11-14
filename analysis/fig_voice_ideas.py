"""Figures for voice and ideas, from results/paper/voice_content.csv (voice_content.py).

fig0_voice_ideas: every group placed by how much of the human map of ideas it reaches and
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

    # Figure 1
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    plain = res[res["cond"].isna() & (res["model"] != "people")]
    others = plain[~plain["model"].isin([FOCUS, BASE])]
    dots(ax, others.loc[others["kind"] == "base", X], others.loc[others["kind"] == "base", Y], s=30, color=BLUE, zorder=3,
         edgecolor="white", linewidth=1, label="Base models, no persona")
    dots(ax, others.loc[others["kind"] == "chat", X], others.loc[others["kind"] == "chat", Y], s=30, color=ORANGE, zorder=3,
         edgecolor="white", linewidth=1, label="Chat models, no persona")
    for _, r in others[others["name"].isin(["Qwen3 8B", "Qwen3 32B", "gpt-oss 20B", "Qwen2.5 14B", "OLMo 2 7B", "GPT-2 XL", "GPT-J 6B"])].iterrows():
        ax.annotate(r["name"], (r[X], r[Y]), xytext=(4, -9) if r["name"] == "gpt-oss 20B" else (4, 3),
                    textcoords="offset points", fontsize=6.5, color=MUTED)
    jk_path = ROOT / "results" / "paper" / "players_uncertainty.csv"
    if jk_path.exists():  # standard error from leaving out one player at a time
        jk = pd.read_csv(jk_path, index_col=0)
        for m, r in res[res["cond"].isna()].iterrows():
            if r["model"] in jk.index:
                ax.errorbar(r[X], r[Y], xerr=jk.loc[r["model"], "map_reached_ideas_jk"], fmt="none", ecolor=FAINT,
                            elinewidth=0.8, capsize=0, zorder=1)
    t = res[(res["base"] == FOCUS) & res["cond"].fillna("").str.startswith("T ")].copy()
    t["T"] = t["cond"].str[2:].astype(float)
    t = t.sort_values("T")
    ax.plot(t[X], t[Y], color=MUTED, linewidth=1.1, linestyle=(0, (3, 2)), zorder=2)
    dots(ax, t[X], t[Y], s=9, color=MUTED, zorder=3)
    for _, r in t.iloc[[0, -1]].iterrows():
        ax.annotate(f"T {r['T']:.1f}", (r[X], r[Y]), xytext=(6, 0), textcoords="offset points", fontsize=7, color=MUTED)

    def arrow(a, b, color):
        ax.annotate("", xy=(g.loc[b, X], g.loc[b, Y]), xytext=(g.loc[a, X], g.loc[a, Y]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, shrinkA=6, shrinkB=6, mutation_scale=10), zorder=4)

    arrow(BASE, FOCUS, ORANGE)
    cbase = f"{FOCUS} | creative:baseline"
    for c, label in CREATIVE.items():
        key = f"{FOCUS} | creative:{c}"
        arrow(FOCUS, key, AQUA)
        dots(ax, [g.loc[key, X]], [g.loc[key, Y]], s=34, color=AQUA, edgecolor="white", linewidth=1.2, zorder=5)
        off = {"memory": (-5, 4), "playful": (-5, -10)}.get(c, (-5, -3))
        ax.annotate(label, (g.loc[key, X], g.loc[key, Y]), xytext=off, textcoords="offset points", fontsize=7.5,
                    color=INK, ha="right")
    dots(ax, [g.loc[BASE, X]], [g.loc[BASE, Y]], s=60, color=BLUE, edgecolor="white", linewidth=1.5, zorder=5)
    ax.annotate("Qwen2.5 7B\nbefore chat training", (g.loc[BASE, X], g.loc[BASE, Y]), xytext=(8, 2),
                textcoords="offset points", fontsize=8, color=INK)
    dots(ax, [g.loc[FOCUS, X]], [g.loc[FOCUS, Y]], s=60, color=ORANGE, edgecolor="white", linewidth=1.5, zorder=5)
    ax.annotate("after chat training", (g.loc[FOCUS, X], g.loc[FOCUS, Y]), xytext=(8, 3), textcoords="offset points",
                fontsize=8, color=INK, ha="left")
    p = g.loc["people"]
    dots(ax, [p[X]], [p[Y]], s=150, marker="*", color=INK, zorder=6)
    ax.annotate("half of the players,\nagainst the other half", (p[X], p[Y]), xytext=(8, -14), textcoords="offset points",
                fontsize=8, color=INK, ha="left")
    ax.text(0.98, 0.97, "chat training", transform=ax.transAxes, fontsize=8, color=ORANGE, va="top", ha="right")
    ax.text(0.98, 0.91, "creative conditions, same chat model", transform=ax.transAxes, fontsize=8, color=AQUA, va="top", ha="right")
    ax.text(0.98, 0.85, "more randomness (temperature)", transform=ax.transAxes, fontsize=8, color=MUTED, va="top", ha="right")
    ax.set_xlabel("Share of the human map of ideas reached (recall, voice removed)")
    if jk_path.exists():
        ax.text(0.02, 0.17, "grey bars: standard error from\nleaving out one player at a time", transform=ax.transAxes,
                fontsize=7, color=MUTED, va="bottom")
    ax.set_ylabel("Repetition of the voice\n(compression of sentence patterns)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
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
    for ax, col, lo, hi, title in ((axes[0], "ideas", "lo", "hi", "Ideas: change in map reached"),
                                   (axes[1], "voice", "vlo", "vhi", "Voice: change in repetition")):
        ax.hlines(y, s[lo], s[hi], color=GRID, linewidth=3)
        dots(ax, s[col], y, color=colors, s=28, zorder=3, edgecolor="white", linewidth=1)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y); axes[0].set_yticklabels(s["cond"].str.replace("_", " ").str.replace(" siev", "").str.replace(" tmk", ""))
    dots(axes[1], [], [], color=BLUE, label="Reasoning (FER)"); dots(axes[1], [], [], color=ORANGE, label="Creative")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(figs / "fig3_conditions.pdf"); fig.savefig(figs / "fig3_conditions.svg")


if __name__ == "__main__":
    main()
