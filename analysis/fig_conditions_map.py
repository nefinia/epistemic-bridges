"""Figure 2: what each persona and creative condition changes, as one map.

Across, the change in recall on the map of ideas; up and down, the change in how much the voice
repeats its sentence patterns; both against the same model with no persona, averaged over models
(voice_content_deltas.csv, from uncertainty.py), on the five models that answered every condition,
with one standard error over models as bars. Each reasoning persona is drawn alone and, with an
arrow, held by an ordinary life drawn at random.

    python analysis/fig_conditions_map.py
"""

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import dots  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
# label offsets in points, placed by hand so that no label covers a point or another label
OFF = {"dreamy": (-5, 3, "right"), "association": (-5, -6, "right"), "wordplay": (5, 4, "left"), "child": (-5, 4, "right"),
       "memory": (5, 3, "left"), "story": (-5, 3, "right"), "senses": (-5, 3, "right"), "playful": (5, 0, "left"),
       "step by step": (5, -3, "left"), "ordinary life": (0, 9, "center"), "random constraint": (5, 0, "left"),
       "teleological": (6, 0, "left"), "analogical thinker": (-4, 9, "right"), "causal mechanist": (6, -8, "left"),
       "dialectical": (-6, 0, "right")}


def summarise(per_model):
    """Mean and standard error over the five models that answered every condition."""
    five = per_model[per_model.run == "creative"].model.unique()
    p = per_model[per_model.run.isin(["fer", "creative", "lifeway"]) & per_model.model.isin(five)]
    g = p.groupby(["run", "cond"])
    return g.agg(ideas=("d_ideas", "mean"), ideas_se=("d_ideas", "sem"), voice=("d_voice", "mean"),
                 voice_se=("d_voice", "sem"), lo=("d_ideas", "min"), hi=("d_ideas", "max"),
                 positive=("d_ideas", lambda x: int((x > 0).sum())), models=("model", "nunique")).reset_index()


def draw(ax, d, fs=7.5):
    fer = d[d.run == "fer"].set_index("cond")
    way = d[d.run == "lifeway"].assign(cond=lambda x: x.cond.str.replace("life_way+", "", regex=False)).set_index("cond")
    cre = d[d.run == "creative"].set_index("cond")
    nm = lambda c: c.replace("_", " ").replace(" siev", "").replace(" tmk", "")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axhline(0, color=MUTED, lw=0.8)

    def label(text, x, y, color):
        dx, dy, ha = OFF.get(text, (5, 3, "left"))
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fs, color=color, ha=ha, va="center", zorder=5)

    def bars(r, color):
        ax.errorbar(r.ideas, r.voice, xerr=r.ideas_se, yerr=r.voice_se, fmt="none", ecolor=color, elinewidth=0.8,
                    alpha=0.35, capsize=0, zorder=1)

    for c, r in cre.iterrows():
        bars(r, ORANGE)
        dots(ax, [r.ideas], [r.voice], color=ORANGE, s=34, edgecolor="white", linewidth=1, zorder=3)
        label(nm(c), r.ideas, r.voice, MUTED)
    for c, r in fer.iterrows():
        if c in way.index:
            w = way.loc[c]
            bars(w, AQUA)
            ax.annotate("", xy=(w.ideas, w.voice), xytext=(r.ideas, r.voice), zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=1, alpha=0.55, linestyle=(0, (4, 2)),
                                        shrinkA=4, shrinkB=4))
            dots(ax, [w.ideas], [w.voice], color=AQUA, marker="D", s=34, edgecolor="white", linewidth=1, zorder=3)
        bars(r, BLUE)
        dots(ax, [r.ideas], [r.voice], color=BLUE, marker="s", s=34, edgecolor="white", linewidth=1, zorder=3)
        label(nm(c), r.ideas, r.voice, BLUE)
    ax.set_xlabel("Ideas: change in recall   (← narrower    wider →)")
    ax.set_ylabel("Voice: change in repetition\n(↓ more varied    more repetitive ↑)")
    ax.legend(handles=[Line2D([], [], ls="none", marker="s", color=BLUE, label="reasoning persona"),
                       Line2D([], [], ls="none", marker="D", color=AQUA, label="held by an ordinary life"),
                       Line2D([], [], ls="none", marker="o", color=ORANGE, label="creative condition")],
              frameon=False, fontsize=fs, loc="lower left")


def table(d):
    """The numbers behind the figure, with the range over models, for the appendix."""
    d = d[d.run.isin(["fer", "creative", "lifeway"])].copy()
    lab = {"fer": "reasoning", "creative": "creative", "lifeway": "life + reasoning"}
    d["name"] = (d.cond.str.replace("life_way+", "", regex=False).str.replace("_", " ")
                 .str.replace(" siev", "").str.replace(" tmk", ""))
    d = d.sort_values(["run", "ideas"])
    rows = [f"{lab[r.run]} & {r['name']} & {r.ideas:+.2f} ({r.ideas_se:.2f}) & {r.lo:+.2f} to {r.hi:+.2f} & {int(r.positive)} of {int(r.models)} & {r.voice:+.2f} ({r.voice_se:.2f})\\\\"
            for _, r in d.iterrows()]
    t = ("\\begin{table}[H]\\centering\\footnotesize\n\\caption{The conditions of Figure~\\ref{fig:conditions}, change against the same model "
         "with no persona, on the five models that answered every condition: recall on the map of ideas (mean with standard error over "
         "models, range, and the number of models it widens), and repetition of the voice (mean with standard error).}\n\\label{tab:conditions}\n\\begin{tabular}{llrrrr}\n\\toprule\n"
         "Kind & Condition & Recall & Range & Wider & Voice\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (ROOT / "paper" / "appendix_conditions_table.tex").write_text(t)


def main():
    d = summarise(pd.read_csv(ROOT / "results" / "paper" / "voice_content_deltas.csv"))
    table(d)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "svg.fonttype": "none"})
    figs = ROOT / "paper" / "figs"
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    draw(ax, d)
    fig.tight_layout()
    fig.savefig(figs / "fig3_conditions.pdf"); fig.savefig(figs / "fig3_conditions.svg"); fig.savefig(figs / "fig3_conditions.png", dpi=150)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    draw(ax, d, fs=7)
    fig.tight_layout()
    fig.savefig(figs / "web" / "fig3.svg")


if __name__ == "__main__":
    main()
