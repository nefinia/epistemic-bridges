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
    plain = res[res["cond"].isna() & (res["model"] != "people")]
    jk_path = ROOT / "results" / "paper" / "players_uncertainty.csv"
    jk = pd.read_csv(jk_path, index_col=0) if jk_path.exists() else None

    # One row per model, ordered by year and size, the same rows in every panel. People are the shaded band:
    # their value with one standard error over players on each side.
    key = f"{FOCUS} | creative:ordinary_life"
    meta = pd.read_csv(ROOT / "results" / "paper" / "models_baseline.csv").set_index("model")
    rows = plain.copy()
    rows["year"] = [meta.loc[m.split(" | ")[0], "year"] for m in rows["model"]]
    rows["size"] = [meta.loc[m.split(" | ")[0], "size_b"] + (0.02 if m == key else 0.01 if m == FOCUS else 0.03 if "OLMo" in m else 0) for m in rows["model"]]
    rows = rows.sort_values(["year", "size"]).reset_index(drop=True)
    yy = np.arange(len(rows))

    def look(m, kind):
        if m == key:
            return AQUA, "o", 34
        return (BLUE, "s", 30) if kind == "base" else (ORANGE, "o", 30)

    def label(m, n, year):
        if m == key:
            return "Qwen2.5 7B, ordinary life"
        return f"{n.replace(' base', ' (base)')}, {int(year)}"

    def row_panel(ax, col, xlabel, title, err=None, labels=True):
        pv = g.loc["people", col]
        pe = jk.loc["people", f"{col}_jk"] if jk is not None and "people" in jk.index else 0
        ax.axvspan(pv - pe, pv + pe, color=GRID, alpha=0.9, zorder=0, lw=0)
        ax.axvline(pv, color=INK, linewidth=1, zorder=1)
        for yi, (_, r) in zip(yy, rows.iterrows()):
            c, mk, sz = look(r["model"], r["kind"])
            if err is not None and r["model"] in err.index:
                e = err.loc[r["model"], f"{col}_jk"]
                ax.hlines(yi, r[col] - e, r[col] + e, color=c, alpha=0.35, linewidth=1.5, zorder=2)
            dots(ax, [r[col]], [yi], color=c, marker=mk, s=sz, zorder=3)
        ax.set_yticks(yy)
        if labels:
            ax.set_yticklabels([label(m, n, y) for m, n, y in zip(rows["model"], rows["name"], rows["year"])], fontsize=7.5)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel(xlabel); ax.set_title(title, fontsize=9, color=INK)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(len(rows) - 0.3, -1.6)
        ax.text(pv, -1.1, "people", ha="center", va="center", fontsize=7.5, color=INK)

    def ideas_panel(ax, labels=True):
        row_panel(ax, "map_reached_ideas", "Share of the human\nmap of ideas reached", "Ideas: recall", err=jk, labels=labels)

    def onmap_panel(ax, labels=True):
        row_panel(ax, "on_map_ideas", "Share of answers\non the human map", "Ideas: precision", err=jk, labels=labels)

    def voice_panel(av, labels=True):
        row_panel(av, "cr_pos", "Repetition of\nsentence patterns", "Voice", err=jk, labels=labels)

    fig, (ax, ao, av) = plt.subplots(1, 3, figsize=(6.5, 3.9), sharey=True)
    ideas_panel(ax); onmap_panel(ao, labels=False); voice_panel(av, labels=False)
    fig.tight_layout()
    fig.savefig(figs / "fig0_voice_ideas.pdf"); fig.savefig(figs / "fig0_voice_ideas.svg")
    web = figs / "web"
    web.mkdir(exist_ok=True)
    for name, panel in (("fig0a", ideas_panel), ("fig0c", onmap_panel), ("fig0b", voice_panel)):
        f1, a1 = plt.subplots(figsize=(4.6, 4.4)); panel(a1); f1.tight_layout(); f1.savefig(web / f"{name}.svg"); plt.close(f1)

    # Figure 2: change from no persona, ideas and voice
    d = deltas(res)
    d.to_csv(ROOT / "results" / "paper" / "voice_content_deltas.csv", index=False)
    s = d.groupby(["run", "cond"]).agg(ideas=("d_ideas", "mean"), lo=("d_ideas", "min"), hi=("d_ideas", "max"),
                                       voice=("d_voice", "mean"), vlo=("d_voice", "min"), vhi=("d_voice", "max")).reset_index()
    s = s[s["run"].isin(["fer", "creative", "lifeway"])].sort_values("ideas")   # other framings are in the appendix
    y = np.arange(len(s))
    colors = [BLUE if r == "fer" else AQUA if r == "lifeway" else ORANGE for r in s["run"]]
    names = s["cond"].str.replace("_", " ").str.replace(" siev", "").str.replace(" tmk", "").str.replace("ordinary life+", "life + ").str.replace("life way+", "life + ")
    PANELS = {"ideas": ("lo", "hi", "Ideas: change in recall", "\u2190 narrower    wider \u2192"),
              "voice": ("vlo", "vhi", "Voice: repetition", "\u2190 more varied    more repetitive \u2192")}

    def cond_panel(ax, col, labels=True, legend=False):
        lo, hi, title, xlabel = PANELS[col]
        ax.hlines(y, s[lo], s[hi], color=GRID, linewidth=3)
        dots(ax, s[col], y, color=colors, s=28, zorder=3, edgecolor="white", linewidth=1, marker=["s" if r == "fer" else "D" if r == "lifeway" else "o" for r in s["run"]])
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_title(title, fontsize=9, color=INK)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
        ax.set_yticks(y)
        if labels:
            ax.set_yticklabels(names)
        if legend:
            dots(ax, [], [], color=BLUE, marker="s", label="Reasoning (FER)"); dots(ax, [], [], color=ORANGE, label="Creative")
            if (s["run"] == "lifeway").any():
                dots(ax, [], [], color=AQUA, marker="D", label="Ordinary life + reasoning")
            ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 0.17 * len(s) + 1.1), sharey=True)
    cond_panel(axes[0], "ideas"); cond_panel(axes[1], "voice", labels=False)
    axes[0].tick_params(axis="y", labelsize=8); axes[0].tick_params(axis="x", labelsize=8); axes[1].tick_params(axis="x", labelsize=8)
    dots(axes[1], [], [], color=BLUE, marker="s", label="Reasoning (FER)"); dots(axes[1], [], [], color=ORANGE, label="Creative")
    if (s["run"] == "lifeway").any():
        dots(axes[1], [], [], color=AQUA, marker="D", label="Ordinary life + reasoning")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figs / "fig3_conditions.pdf"); fig.savefig(figs / "fig3_conditions.svg")
    for name, col in (("fig3a", "ideas"), ("fig3b", "voice")):
        f1, a1 = plt.subplots(figsize=(4.2, 0.25 * len(s) + 1.5)); cond_panel(a1, col, legend=True)
        f1.tight_layout(); f1.savefig(web / f"{name}.svg"); plt.close(f1)




if __name__ == "__main__":
    main()
