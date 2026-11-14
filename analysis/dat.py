"""Scores the ten-word warm-up with the Divergent Association Task method (Olson et al. 2021):
the first seven valid words (lowercase, in the GloVe vocabulary), mean cosine distance over the
21 pairs, times 100. Players' scores are compared with the published pre-ChatGPT norm
(Study 1A: mean 78.38, SD 6.35) and with the models' warm-ups.

    python analysis/dat.py --players players_warmups.json --tools /path/to/tools
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
NORM_MEAN, NORM_SD = 78.38, 6.35


def load_glove(tools):
    vocab = (tools / "glove200k.vocab.txt").read_text(encoding="utf8").split("\n")
    V = np.load(tools / "glove200k.npy").astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return {w: i for i, w in enumerate(vocab)}, V


def score(words, idx, V):
    valid = []
    for w in words:
        w = re.sub(r"[^a-z\-]", "", str(w).strip().lower().replace(" ", "-"))
        if w and w in idx and w not in valid:
            valid.append(w)
        if len(valid) == 7:
            break
    if len(valid) < 7:
        return np.nan
    E = V[[idx[w] for w in valid]]
    S = E @ E.T
    return float((1 - S[np.triu_indices(7, 1)]).mean() * 100)


def model_words(text):
    lines = [re.sub(r"^[\s\d\.\)\-\*•]+", "", l).strip() for l in re.split(r"[\n,]", text)]
    return [re.sub(r"[^A-Za-z\- ]", "", l).strip().split(" ")[0] for l in lines if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=Path, required=True)
    ap.add_argument("--tools", type=Path, required=True)
    args = ap.parse_args()
    idx, V = load_glove(args.tools)
    pl = json.loads(args.players.read_text())
    people = pd.DataFrame([{"group": "people", "ai_use": u, "dat": score(w, idx, V)} for _, w, u in pl])
    rows = []
    for f in sorted((ROOT / "data" / "ai").glob("*.jsonl")):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if r["task"] == "warmup" and r["lang"] == "en" and r["condition"] == "baseline" and r.get("run", "fer") == "fer":
                rows.append({"group": r["model"], "kind": r["model_kind"], "year": r["year"], "dat": score(model_words(r["final"]), idx, V)})
    models = pd.DataFrame(rows)
    out = ROOT / "results" / "paper"
    people.to_csv(out / "dat_people.csv", index=False)
    summ = models.groupby(["group", "kind", "year"])["dat"].agg(["mean", "std", "count", lambda s: s.notna().sum()]).reset_index()
    summ.columns = ["model", "kind", "year", "mean", "sd", "n", "valid"]
    summ.to_csv(out / "dat_models.csv", index=False)
    p = people["dat"].dropna()
    print(f"people: n={len(p)} of {len(people)}, mean {p.mean():.1f}, sd {p.std():.1f}; norm {NORM_MEAN} (sd {NORM_SD})")
    print(people.groupby(people["ai_use"].fillna("not asked"))["dat"].agg(["mean", "count"]).round(1))
    print(summ.sort_values("year").round(1).to_string(index=False))
    figure(p, summ)


def figure(p, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "svg.fonttype": "none"})
    names = {"openai-community/gpt2-xl": "GPT-2 XL", "EleutherAI/gpt-j-6b": "GPT-J 6B", "lmsys/vicuna-7b-v1.5": "Vicuna 7B",
             "HuggingFaceH4/zephyr-7b-beta": "Zephyr 7B", "microsoft/Phi-3-mini-4k-instruct": "Phi-3 mini",
             "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5 0.5B", "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5 1.5B",
             "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5 3B", "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5 7B",
             "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5 14B", "Qwen/Qwen2.5-7B": "Qwen2.5 7B base",
             "allenai/OLMo-2-1124-7B-Instruct": "OLMo 2 7B", "Qwen/Qwen3-8B": "Qwen3 8B"}
    s = summ[summ["valid"] >= 3].copy()
    s["name"] = s["model"].map(names)
    s = s.sort_values("mean")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.axvspan(NORM_MEAN - NORM_SD, NORM_MEAN + NORM_SD, color=GRID, alpha=0.8, zorder=0)
    ax.axvline(NORM_MEAN, color=MUTED, linewidth=1.2, linestyle=(0, (3, 2)))
    y = np.arange(len(s) + 1)
    ax.scatter(p, np.full(len(p), len(s)) + np.random.default_rng(0).uniform(-0.25, 0.25, len(p)), s=14, color=INK, alpha=0.6, zorder=3)
    ax.scatter([p.mean()], [len(s)], s=70, marker="D", color=INK, edgecolor="white", zorder=4)
    for i, (_, r) in enumerate(s.iterrows()):
        color = BLUE if r["kind"] == "base" else ORANGE
        ax.errorbar(r["mean"], i, xerr=r["sd"], fmt="o", color=color, ecolor=color, elinewidth=1.2, capsize=0, markersize=5)
    ax.set_yticks(y); ax.set_yticklabels(list(s["name"]) + ["People, now"], fontsize=8)
    ax.set_xlabel("Warm-up score (Divergent Association Task)")
    ax.text(NORM_MEAN, len(s) + 0.75, "people before ChatGPT (published norm, ± 1 SD)", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_ylim(-0.7, len(s) + 1.6); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    figs = ROOT / "paper" / "figs"
    fig.savefig(figs / "fig5_warmup.pdf"); fig.savefig(figs / "fig5_warmup.svg")


if __name__ == "__main__":
    if "--figure-only" in sys.argv:
        res = ROOT / "results" / "paper"
        figure(pd.read_csv(res / "dat_people.csv")["dat"].dropna(), pd.read_csv(res / "dat_models.csv"))
    else:
        main()
