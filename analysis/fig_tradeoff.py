"""Figure 5: creative conditions against the temperature dial, on the same two models.
Reads results/paper/{conditions,temperature}.csv written by paper_results.py."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
res, figs = ROOT / "results" / "paper", ROOT / "paper" / "figs"
cond = pd.read_csv(res / "conditions.csv")
temps = pd.read_csv(res / "temperature.csv")
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
for ax, model in zip(axes, ["Qwen2.5 7B", "Qwen3 8B"]):
    t = temps[temps["model"] == model].sort_values("temperature")
    ax.plot(t["recall"], t["precision"], color=MUTED, linewidth=1.5, marker="o", markersize=4, label="No persona, temperature 0.4 to 2.0")
    for _, r in t[t["temperature"].isin([0.4, 1.0, 1.6, 2.0])].iterrows():
        ax.annotate(f"T {r['temperature']:.1f}", (r["recall"], r["precision"]), xytext=(4, -9), textcoords="offset points", fontsize=6.5, color=MUTED)
    c = cond[(cond["model"] == model) & (cond["run"] == "creative") & (cond["condition"] != "baseline")]
    ax.scatter(c["recall"], c["precision"], color=ORANGE, s=30, edgecolor="white", linewidth=1, zorder=3, label="Creative condition, temperature 1.0")
    for _, r in c.iterrows():
        if r["condition"] in ("ordinary_life", "story", "memory", "step_by_step", "playful", "dreamy"):
            ax.annotate(r["condition"].replace("_", " "), (r["recall"], r["precision"]), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK)
    ax.set_title(model, fontsize=9, color=INK)
    ax.set_xlabel("Share of the human map reached (recall)")
    ax.set_xlim(-0.02, 0.75); ax.set_ylim(0, 1)
axes[0].set_ylabel("Answers on the human map (precision)")
axes[0].legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(figs / "fig5_tradeoff.pdf"); fig.savefig(figs / "fig5_tradeoff.svg")
