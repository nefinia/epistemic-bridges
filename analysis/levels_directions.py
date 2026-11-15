"""Concrete and abstract pairs, and the direction of the arrow, for people and models.

Pairs: c1 bee/mirror and c2 salt/clock are concrete-concrete (cc); c3 river/secret and
c4 spider/melody are concrete-abstract (ca); c5 forgetting/festival and c6 distance/habit
are abstract-abstract (aa). Every pair was shown in both directions, to people (one
direction each) and to models (10 answers each way).

Direction is measured on the answer alone (first 40 words, without the two ideas, so word
order cannot create a difference):
- direction shift: distance between the centre of a->b answers and the centre of b->a
  answers, compared with the same distance after shuffling the direction labels
  (permutation test, 2000 shuffles);
- anchoring: how much closer an answer sits to the idea it starts from than to the idea it
  ends at (cosine similarity to the start word minus similarity to the end word).

    python analysis/levels_directions.py --humans data/human_bridges.json --tools /path/to/tools
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import dots  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEVEL = {"cc": "concrete-concrete", "ca": "concrete-abstract", "aa": "abstract-abstract"}
HUBS = {"c1": r"reflect", "c2": r"hour ?glass|sand", "c3": r"whisper|murmur", "c4": r"itsy|web|weav",
        "c5": r"drink|alcohol|forget (?:all )?(?:your|our) (?:worries|troubles)", "c6": r"walk|step"}
RNG = np.random.default_rng(0)


def vendi(E):
    K = np.clip(E @ E.T, 0, 1) / len(E)
    lam = np.linalg.eigvalsh(K)
    lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))


def shift(E, labels):
    a, b = E[labels == "ab"], E[labels == "ba"]
    ca, cb = a.mean(0), b.mean(0)
    return 1 - float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humans", type=Path, default=ROOT / "data" / "human_bridges.json")
    ap.add_argument("--ai", type=Path, default=ROOT / "data" / "ai")
    ap.add_argument("--tools", type=Path, required=True)
    args = ap.parse_args()

    h = pd.DataFrame(json.loads(args.humans.read_text()))
    h = h[(h["lang"] == "en") & (h["set"] == "core")].assign(model="people", kind="people")
    a = pd.DataFrame([json.loads(l) for f in sorted(args.ai.glob("*.jsonl")) for l in f.read_text().splitlines() if l.strip()])
    a = a[(a["task"] == "bridge") & (a["lang"] == "en") & (a["set"] == "core") & (a["condition"] == "baseline")
          & (a["run"].fillna("fer") == "fer")].assign(kind=lambda d: d["model_kind"])
    df = pd.concat([h, a], ignore_index=True)
    df["final"] = df["final"].fillna("").astype(str)
    df = df[df["final"].str.strip() != ""].reset_index(drop=True)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(str(args.tools / "all-MiniLM-L6-v2"))
    ans = m.encode(df["final"].apply(lambda s: " ".join(re.findall(r"\S+", s)[:40])).tolist(), normalize_embeddings=True, batch_size=128)
    ends = m.encode((df["from"] + " " + df["final"].apply(lambda s: " ".join(re.findall(r"\S+", s)[:40])) + " " + df["to"]).tolist(),
                    normalize_embeddings=True, batch_size=128)
    words = sorted(set(df["from"]) | set(df["to"]))
    W = dict(zip(words, m.encode(words, normalize_embeddings=True)))
    df["anchoring"] = [float(ans[i] @ W[f] - ans[i] @ W[t]) for i, (f, t) in enumerate(zip(df["from"], df["to"]))]

    out = ROOT / "results" / "paper"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Concrete and abstract pairs
    rows = []
    for pair, sub in df.groupby("pair"):
        hi = sub.index[sub["model"] == "people"].to_numpy()
        n = len(hi)
        row = {"pair": pair, "level": LEVEL[sub["level"].iloc[0]], "ideas": f"{sub.loc[hi[0], 'from']} / {sub.loc[hi[0], 'to']}",
               "people_n": n, "people_different": vendi(ans[hi]) / n,
               "people_hub": float(sub.loc[hi, "final"].str.lower().str.contains(HUBS[pair]).mean())}
        for kind in ("chat", "base"):
            rec, dif, hub = [], [], []
            for model, g in sub[sub["kind"] == kind].groupby("model"):
                gi = g.index.to_numpy()
                rs, vs = [], []
                for _ in range(30):
                    pick = RNG.choice(gi, size=min(n, len(gi)), replace=False)
                    rs.append(precision_recall(ends[hi], ends[pick], 3)[1]); vs.append(vendi(ans[pick]) / len(pick))
                rec.append(np.mean(rs)); dif.append(np.mean(vs))
                hub.append(float(g["final"].str.lower().str.contains(HUBS[pair]).mean()))
            row.update({f"{kind}_map_reached": np.mean(rec), f"{kind}_different": np.mean(dif), f"{kind}_hub": np.mean(hub)})
        rows.append(row)
    levels = pd.DataFrame(rows)
    levels.to_csv(out / "pairs_levels.csv", index=False)
    print(levels.round(2).to_string(index=False))

    # 2. Direction of the arrow
    rows = []
    for (pair, model), sub in df.groupby(["pair", "model"]):
        idx = sub.index.to_numpy()
        lab = sub["dir"].to_numpy()
        if (lab == "ab").sum() < 3 or (lab == "ba").sum() < 3:
            continue
        obs = shift(ans[idx], lab)
        null = np.array([shift(ans[idx], RNG.permutation(lab)) for _ in range(2000)])
        rows.append({"pair": pair, "model": model, "kind": sub["kind"].iloc[0], "n_ab": int((lab == "ab").sum()),
                     "n_ba": int((lab == "ba").sum()), "shift": obs, "null_mean": null.mean(),
                     "shift_over_chance": obs / null.mean(), "p": float((null >= obs).mean())})
    dirs = pd.DataFrame(rows)
    dirs.to_csv(out / "directions.csv", index=False)
    summ = dirs.groupby("kind").agg(pairs=("p", "size"), significant=("p", lambda p: int((p < 0.05).sum())),
                                    shift_over_chance=("shift_over_chance", "median"))
    print(summ)
    print(dirs[dirs["model"] == "people"][["pair", "n_ab", "n_ba", "shift", "null_mean", "shift_over_chance", "p"]].round(3).to_string(index=False))
    anch = df.groupby("kind")["anchoring"].agg(["mean", "sem", "size"])
    anch_pair = df.groupby(["kind", "pair"])["anchoring"].mean().unstack()
    anch.to_csv(out / "anchoring.csv")
    anch_pair.to_csv(out / "anchoring_by_pair.csv")
    print(anch.round(3)); print(anch_pair.round(3))
    figure(levels, anch_pair)


def figure(levels, anch_pair):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "svg.fonttype": "none"})
    order = ["c1", "c2", "c3", "c4", "c5", "c6"]
    names = ["bee\nmirror", "salt\nclock", "river\nsecret", "spider\nmelody", "forgetting\nfestival", "distance\nhabit"]
    lv = levels.set_index("pair").loc[order]
    x = np.arange(len(order))
    def diversity_panel(ax):
        ax.bar(x - 0.2, lv["people_different"], width=0.38, color=INK, label="People")
        ax.bar(x + 0.2, lv["chat_different"], width=0.38, color=ORANGE, label="Chat models (mean)")
        ax.set_ylabel("Really different answers per answer"); ax.set_title("Diversity by pair", fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right"); ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 0.8)
        finish(ax)

    def pull_panel(ax):
        for kind, color, name, off in (("people", INK, "People", -0.2), ("base", BLUE, "Base models", 0.0), ("chat", ORANGE, "Chat models", 0.2)):
            sr = anch_pair.loc[kind].reindex(order)
            dots(ax, x + off, sr.values, color=color, s=34, edgecolor="white", linewidth=1, label=name, zorder=3)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_ylabel("Closer to the starting idea (+)"); ax.set_title("Pull of the starting idea", fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right"); ax.grid(axis="x", visible=False)
        finish(ax)

    def finish(ax):
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
        for i0, lab in ((0.5, "concrete"), (2.5, "concrete–abstract"), (4.5, "abstract")):
            ax.text(i0, -0.2, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color=MUTED)
        ax.axvline(1.5, color=GRID, linewidth=1); ax.axvline(3.5, color=GRID, linewidth=1)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    diversity_panel(axes[0]); pull_panel(axes[1])
    fig.tight_layout()
    figs = ROOT / "paper" / "figs"
    fig.savefig(figs / "fig4_pairs_direction.pdf"); fig.savefig(figs / "fig4_pairs_direction.svg")
    (figs / "web").mkdir(exist_ok=True)
    for name, panel in (("fig4a", diversity_panel), ("fig4b", pull_panel)):
        f1, a1 = plt.subplots(figsize=(4.4, 3.4)); panel(a1); f1.tight_layout(); f1.savefig(figs / "web" / f"{name}.svg"); plt.close(f1)


if __name__ == "__main__":
    if "--figure-only" in sys.argv:
        res = ROOT / "results" / "paper"
        figure(pd.read_csv(res / "pairs_levels.csv"), pd.read_csv(res / "anchoring_by_pair.csv", index_col=0))
    else:
        main()
