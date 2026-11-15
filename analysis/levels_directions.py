"""Concrete and abstract pairs, and the direction of the arrow, for people and models.

Pairs: c1 bee/mirror and c2 salt/clock are concrete-concrete (cc); c3 river/secret and
c4 spider/melody are concrete-abstract (ca); c5 forgetting/festival and c6 distance/habit
are abstract-abstract (aa). Every pair was shown in both directions, to people (one
direction each) and to models (10 answers each way).

Everything is measured twice: on the full answer as written, and with the voice removed
(content words only, as in voice_content.py). Direction is measured on the answer alone,
without the two ideas, so word order cannot create a difference:
- direction shift: distance between the centre of a->b answers and the centre of b->a
  answers, compared with the same distance after shuffling the direction labels
  (permutation test, 2000 shuffles);
- anchoring (pull of the first word): how much closer an answer sits to the word the arrow starts
  from than to the word it ends at (cosine similarity to the start word minus similarity to the end word).

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


def content_words():
    """The voice-removed version of an answer, as in voice_content.py: scaffolding stripped, then content words."""
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from style_and_warmup import strip_scaffolding
    from voice_content import TASK_WORDS
    stop, lem = set(stopwords.words("english")), WordNetLemmatizer()

    def f(text, a, b):
        ideas = {lem.lemmatize(w.lower(), "n") for w in re.findall(r"[A-Za-z]+", a + " " + b)}
        out = []
        for w, tag in nltk.pos_tag(re.findall(r"[A-Za-z][A-Za-z'\-]*", strip_scaffolding(text))):
            pos = "n" if tag.startswith("NN") else "v" if tag.startswith("VB") else "a" if tag.startswith("JJ") else None
            if pos is None:
                continue
            l = lem.lemmatize(w.lower(), pos)
            if l not in stop and l not in TASK_WORDS and l not in ideas and len(l) >= 3:
                out.append(l)
        return out
    return f


def measure(df, ans, ends, W, tag, out):
    df = df.copy()
    df["anchoring"] = [float(ans[i] @ W[f] - ans[i] @ W[t]) for i, (f, t) in enumerate(zip(df["from"], df["to"]))]
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
    levels.to_csv(out / f"pairs_levels{tag}.csv", index=False)
    print(tag or "as written"); print(levels.round(2).to_string(index=False))
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
    dirs.to_csv(out / f"directions{tag}.csv", index=False)
    summ = dirs.groupby("kind").agg(pairs=("p", "size"), significant=("p", lambda p: int((p < 0.05).sum())),
                                    shift_over_chance=("shift_over_chance", "median"))
    print(summ)
    anch = df.groupby("kind")["anchoring"].agg(["mean", "sem", "size"])
    anch_pair = df.groupby(["kind", "pair"])["anchoring"].mean().unstack()
    anch.to_csv(out / f"anchoring{tag}.csv")
    anch_pair.to_csv(out / f"anchoring_by_pair{tag}.csv")
    print(anch.round(3)); print(anch_pair.round(3))
    # Closeness to the left word of the pair (as listed in pairs.json, "a" in the direction labels) minus the
    # right word, for answers in each direction: equal heights mean the direction does not matter.
    df["toward_left"] = np.where(df["dir"] == "ab", df["anchoring"], -df["anchoring"])
    by_dir = df.groupby(["kind", "pair", "dir"])["toward_left"].agg(["mean", "sem", "size"]).reset_index()
    by_dir.to_csv(out / f"toward_left_by_direction{tag}.csv", index=False)
    return levels, anch_pair, by_dir


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
    words = sorted(set(df["from"]) | set(df["to"]))
    W = dict(zip(words, m.encode(words, normalize_embeddings=True)))
    out = ROOT / "results" / "paper"
    out.mkdir(parents=True, exist_ok=True)

    # As written: the full answer alone, and the answer read between its two ideas
    ans = m.encode(df["final"].tolist(), normalize_embeddings=True, batch_size=128)
    ends = m.encode((df["from"] + " " + df["final"] + " " + df["to"]).tolist(), normalize_embeddings=True, batch_size=128)
    written = measure(df, ans, ends, W, "", out)

    # Voice removed: content words, model answers kept to as many as a player's answer to the same pair
    cw = content_words()
    conc = [cw(t, a_, b_) for t, a_, b_ in zip(df["final"], df["from"], df["to"])]
    people = (df["model"] == "people").to_numpy()
    hconc = {p: [len(c) for c, pp, ip in zip(conc, df["pair"], people) if ip and pp == p and len(c)] for p in df["pair"].unique()}
    rng = np.random.default_rng(0)
    text = [" ".join(c if ip or not len(c) else c[: int(rng.choice(hconc[p]))]) for c, ip, p in zip(conc, people, df["pair"])]
    keep = np.array([len(t) > 0 for t in text])
    d2 = df[keep].reset_index(drop=True)
    t2 = [t for t, k in zip(text, keep) if k]
    E = m.encode(t2, normalize_embeddings=True, batch_size=256)
    ideas = measure(d2, E, E, W, "_ideas", out)
    figure(written, ideas)


def figure(written, ideas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "svg.fonttype": "none",
                         "hatch.linewidth": 0.8})
    order = ["c1", "c2", "c3", "c4", "c5", "c6"]
    names = ["bee\nmirror", "salt\nclock", "river\nsecret", "spider\nmelody", "forgetting\nfestival", "distance\nhabit"]
    short = ["bee / mirror", "salt / clock", "river / secret", "spider / melody", "forgetting / festival", "distance / habit"]
    lw, li = written[0].set_index("pair").loc[order], ideas[0].set_index("pair").loc[order]
    x = np.arange(len(order))

    def diversity_panel(ax, lv, title):
        ax.bar(x - 0.2, lv["people_different"], width=0.38, color=INK, label="People")
        ax.bar(x + 0.2, lv["chat_different"], width=0.38, color=ORANGE, label="Chat models (mean)")
        ax.set_title(title, fontsize=9, color=INK); ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5, rotation=35, ha="right", rotation_mode="anchor")
        ax.axvline(1.5, color=GRID, linewidth=1); ax.axvline(3.5, color=GRID, linewidth=1)

    def direction_panel(ax, kind, color, title, legend=False):
        d = ideas[2][ideas[2]["kind"] == kind].set_index(["pair", "dir"])["mean"]
        ab, ba = d.xs("ab", level="dir").reindex(order).values, d.xs("ba", level="dir").reindex(order).values
        ax.vlines(x, np.minimum(ab, ba), np.maximum(ab, ba), color=color, linewidth=1.5, alpha=0.5)
        dots(ax, x, ab, color=color, marker="^", s=46, edgecolor="white", linewidth=0.8, zorder=3,
             label="arrow from the left word" if legend else None)
        dots(ax, x, ba, color="white", marker="v", s=40, edgecolor=color, linewidth=1.3, zorder=3,
             label="arrow from the right word" if legend else None)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(title, fontsize=9, color=INK); ax.grid(axis="x", visible=False)
        ax.set_ylim(-0.3, 0.3)
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5, rotation=35, ha="right", rotation_mode="anchor")
        ax.axvline(1.5, color=GRID, linewidth=1); ax.axvline(3.5, color=GRID, linewidth=1)

    def finish(ax):
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.5)
        for i0, lab in ((0.5, "concrete"), (2.5, "concrete–abstract"), (4.5, "abstract")):
            ax.text(i0, -0.3, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5, color=MUTED)
        ax.axvline(1.5, color=GRID, linewidth=1); ax.axvline(3.5, color=GRID, linewidth=1)

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(6.5, 6.4))
    diversity_panel(a1, lw, "Diversity by pair, as written"); diversity_panel(a2, li, "Diversity by pair, voice removed")
    a1.set_ylabel("Different answers\nper answer")
    a1.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    direction_panel(a3, "people", INK, "People: does the direction matter?")
    direction_panel(a4, "chat", ORANGE, "Chat models: does the direction matter?", legend=True)
    a3.set_ylabel("Closer to the left word (+)\nor the right word (\u2212)")
    h_, l_ = a4.get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    figs = ROOT / "paper" / "figs"
    fig.savefig(figs / "fig4_pairs_direction.pdf"); fig.savefig(figs / "fig4_pairs_direction.svg")
    (figs / "web").mkdir(exist_ok=True)
    f1, (b1, b2) = plt.subplots(2, 1, figsize=(4.6, 5.6))
    diversity_panel(b1, lw, "Diversity by pair, as written"); diversity_panel(b2, li, "Diversity by pair, voice removed")
    b1.set_ylabel("Different answers per answer"); b2.set_ylabel("Different answers per answer")
    b1.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    f1.tight_layout(); f1.savefig(figs / "web" / "fig4a.svg"); plt.close(f1)
    f1, (b3, b4) = plt.subplots(2, 1, figsize=(4.6, 6.0))
    direction_panel(b3, "people", INK, "People: does the direction matter?", legend=True)
    direction_panel(b4, "chat", ORANGE, "Chat models: does the direction matter?", legend=True)
    for b in (b3, b4):
        b.set_ylabel("Closer to the left word (+)"); b.legend(frameon=False, fontsize=7.5, loc="upper left")
    f1.tight_layout(); f1.savefig(figs / "web" / "fig4b.svg"); plt.close(f1)


if __name__ == "__main__":
    if "--figure-only" in sys.argv:
        res = ROOT / "results" / "paper"
        figure(pd.read_csv(res / "pairs_levels.csv"), pd.read_csv(res / "anchoring_by_pair.csv", index_col=0))
    else:
        main()
