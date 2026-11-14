"""The summary figure: how one model's map closes with chat training, and how
creative personas and temperature open it again, next to another group of people.

Everything is measured the same way here, half against half: for each pair, a
random half of the players draws the map, and each group (the other half of the
players, or as many answers from a model) is placed against it. Averaged over
draws and pairs. English, core pairs, answers read between their two ideas and
cut to 40 words, all-MiniLM-L6-v2.

    python analysis/money_plot.py --humans data/human_bridges.json --tools /path/to/tools
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage import precision_recall  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FOCUS = "Qwen/Qwen2.5-7B-Instruct"
BASE = "Qwen/Qwen2.5-7B"
CREATIVE = {"ordinary_life": "an ordinary life", "memory": "a memory", "story": "a story",
            "playful": "playful", "step_by_step": "step by step"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--humans", type=Path, default=ROOT / "data" / "human_bridges.json")
    ap.add_argument("--ai", type=Path, default=ROOT / "data" / "ai")
    ap.add_argument("--tools", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=200)
    args = ap.parse_args()

    h = pd.DataFrame(json.loads(args.humans.read_text()))
    h = h[(h["lang"] == "en") & (h["set"] == "core")].assign(group="people")
    rows = [json.loads(l) for f in sorted(args.ai.glob("*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    a = pd.DataFrame(rows)
    a = a[(a["task"] == "bridge") & (a["lang"] == "en") & (a["set"] == "core")].copy()
    a["run"] = a["run"].fillna("fer")
    a["temperature"] = a["temperature"].fillna(1.0)
    groups = []
    base1 = (a["run"] == "fer") & (a["condition"] == "baseline")
    for model in a.loc[base1, "model"].unique():
        groups.append((model, "model", a[base1 & (a["model"] == model)]))
    for c in CREATIVE:
        groups.append((c, "creative", a[(a["run"] == "creative") & (a["model"] == FOCUS) & (a["condition"] == c)]))
    for t in sorted(a.loc[a["run"] == "temps", "temperature"].unique()):
        groups.append((f"T {t:.1f}", "temp", a[(a["run"] == "temps") & (a["model"] == FOCUS) & (a["temperature"] == t)]))

    every = pd.concat([h] + [g for _, _, g in groups], ignore_index=False, keys=range(len(groups) + 1))
    text = (every["from"] + " " + every["final"].fillna("").astype(str).apply(lambda s: " ".join(re.findall(r"\S+", s)[:40]))
            + " " + every["to"])
    from sentence_transformers import SentenceTransformer
    E_all = SentenceTransformer(str(args.tools / "all-MiniLM-L6-v2")).encode(text.tolist(), normalize_embeddings=True, batch_size=128)
    E = {i: E_all[np.where(every.index.get_level_values(0) == i)[0]] for i in range(len(groups) + 1)}
    pair_of = {i: every.loc[i, "pair"].to_numpy() for i in range(len(groups) + 1)}

    rng = np.random.default_rng(0)
    out = []
    labels = [("people", "people")] + [(name, kind) for name, kind, _ in groups]
    for gi, (name, kind) in enumerate(labels):
        P, R = [], []
        for pair in sorted(set(pair_of[0])):
            hi = np.where(pair_of[0] == pair)[0]
            gidx = np.where(pair_of[gi] == pair)[0]
            ps, rs = [], []
            for _ in range(args.draws):
                perm = rng.permutation(hi)
                ref, other = perm[: len(hi) // 2], perm[len(hi) // 2:]
                if gi == 0:
                    pick = other
                    p, r = precision_recall(E[0][ref], E[0][pick], 2)
                else:
                    if len(gidx) < len(other):
                        continue
                    pick = rng.choice(gidx, size=len(other), replace=False)
                    p, r = precision_recall(E[0][ref], E[gi][pick], 2)
                ps.append(p); rs.append(r)
            if ps:
                P.append(np.mean(ps)); R.append(np.mean(rs))
        out.append({"group": name, "kind": kind, "precision": np.mean(P), "recall": np.mean(R)})
        print(f"{name:40s} on the map {np.mean(P):.2f}  map reached {np.mean(R):.2f}")
    res = pd.DataFrame(out)
    (ROOT / "results" / "paper").mkdir(parents=True, exist_ok=True)
    res.to_csv(ROOT / "results" / "paper" / "money_plot.csv", index=False)
    draw(res)


def draw(res: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, INK, MUTED, FAINT, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#b9b8b2", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "svg.fonttype": "none"})
    g = res.set_index("group")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    others = res[(res["kind"] == "model") & ~res["group"].isin([FOCUS, BASE])]
    ax.scatter(others["recall"], others["precision"], s=18, color=FAINT, zorder=2, label="Other models, no persona")

    t = res[res["kind"] == "temp"].copy()
    t["T"] = t["group"].str[2:].astype(float)
    t = t.sort_values("T")
    ax.plot(t["recall"], t["precision"], color=MUTED, linewidth=1.2, linestyle=(0, (3, 2)), zorder=2)
    ax.scatter(t["recall"], t["precision"], s=10, color=MUTED, zorder=3)
    hot = t.iloc[-1]
    ax.annotate("more randomness\n(temperature up to 2.0)", (hot["recall"], hot["precision"]), xytext=(8, 2),
                textcoords="offset points", fontsize=7.5, color=MUTED, ha="left")

    def arrow(a, b, color):
        ax.annotate("", xy=(g.loc[b, "recall"], g.loc[b, "precision"]), xytext=(g.loc[a, "recall"], g.loc[a, "precision"]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, shrinkA=6, shrinkB=6, mutation_scale=10), zorder=4)

    arrow(BASE, FOCUS, ORANGE)
    for c in CREATIVE:
        arrow(FOCUS, c, AQUA)

    ax.scatter(g.loc[BASE, "recall"], g.loc[BASE, "precision"], s=60, color=BLUE, edgecolor="white", linewidth=1.5, zorder=5)
    ax.annotate("Qwen2.5 7B\nbefore chat training", (g.loc[BASE, "recall"], g.loc[BASE, "precision"]), xytext=(0, 12),
                textcoords="offset points", fontsize=8, color=INK, ha="center")
    ax.scatter(g.loc[FOCUS, "recall"], g.loc[FOCUS, "precision"], s=60, color=ORANGE, edgecolor="white", linewidth=1.5, zorder=5)
    ax.annotate("after chat training", (g.loc[FOCUS, "recall"], g.loc[FOCUS, "precision"]), xytext=(0.15, 0.655),
                textcoords="data", fontsize=8, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=2, shrinkB=5))
    offsets = {"ordinary_life": (0, -13, "center"), "playful": (6, -3, "left")}
    for c, label in CREATIVE.items():
        ax.scatter(g.loc[c, "recall"], g.loc[c, "precision"], s=40, color=AQUA, edgecolor="white", linewidth=1.2, zorder=5)
        dx, dy, ha = offsets.get(c, (6, -3, "left"))
        ax.annotate(label, (g.loc[c, "recall"], g.loc[c, "precision"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, color=INK, ha=ha)
    ax.scatter(g.loc["people", "recall"], g.loc["people", "precision"], s=150, marker="*", color=INK, zorder=6)
    ax.annotate("half of the players,\nagainst the other half", (g.loc["people", "recall"], g.loc["people", "precision"]), xytext=(10, -14),
                textcoords="offset points", fontsize=8, color=INK)

    ax.text(0.02, 0.97, "chat training", transform=ax.transAxes, fontsize=8, color=ORANGE, va="top")
    ax.text(0.02, 0.91, "creative personas, same chat model", transform=ax.transAxes, fontsize=8, color=AQUA, va="top")
    ax.set_xlabel("Share of the human map reached (recall)")
    ax.set_ylabel("Answers on the human map (precision)")
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    figs = ROOT / "paper" / "figs"
    fig.savefig(figs / "fig0_money.pdf"); fig.savefig(figs / "fig0_money.svg")


if __name__ == "__main__":
    if "--draw-only" in sys.argv:
        draw(pd.read_csv(ROOT / "results" / "paper" / "money_plot.csv"))
    else:
        main()
