"""All numbers and figures for the report, in one run.

    python analysis/paper_results.py --humans data/human_bridges.json --tools /path/to/tools

Main setting: English answers to the six core pairs, each answer read between
its two ideas and cut to its first 40 words, embedded with all-MiniLM-L6-v2.
Robustness: two other embedding models and a 15-word cut without the ideas.
Writes results/paper/*.csv and paper/figs/*.pdf|svg (figures contain no player text).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage import precision_recall  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import dots  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)

HUBS = {   # the obvious bridge for each core pair, as seen in the first answers
    "c1": r"reflect", "c2": r"hour ?glass|sand", "c3": r"whisper|murmur",
    "c4": r"itsy|web|weav", "c5": r"drink|alcohol|forget (all )?(your|our) (worries|troubles)", "c6": r"walk|step",
}
SHORT = {
    "openai-community/gpt2-xl": "GPT-2 XL", "EleutherAI/gpt-j-6b": "GPT-J 6B", "lmsys/vicuna-7b-v1.5": "Vicuna 7B",
    "HuggingFaceH4/zephyr-7b-beta": "Zephyr 7B", "microsoft/Phi-3-mini-4k-instruct": "Phi-3 mini",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5 0.5B", "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5 1.5B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5 3B", "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5 7B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5 14B", "Qwen/Qwen2.5-7B": "Qwen2.5 7B base",
    "allenai/OLMo-2-1124-7B-Instruct": "OLMo 2 7B", "Qwen/Qwen3-8B": "Qwen3 8B", "Qwen/Qwen3-32B": "Qwen3 32B", "openai/gpt-oss-20b": "gpt-oss 20B",
}


def load(humans: Path, ai: Path) -> pd.DataFrame:
    h = pd.DataFrame(json.loads(humans.read_text()))
    h["model"], h["condition"], h["run"], h["temperature"], h["model_kind"] = "human", "human", "human", np.nan, "human"
    a = pd.DataFrame([json.loads(l) for f in sorted(ai.glob("*.jsonl")) for l in f.read_text().splitlines() if l.strip()])
    a = a[a["task"] == "bridge"].copy()
    a["run"] = a["run"].fillna("fer")
    a["temperature"] = a["temperature"].fillna(1.0)
    df = pd.concat([h, a], ignore_index=True)
    df = df[(df["set"] == "core") & (df["lang"] == "en")]
    df["final"] = df["final"].fillna("").astype(str).str.strip()
    df = df[df["final"] != ""].reset_index(drop=True)
    df["words"] = df["final"].apply(lambda t: len(re.findall(r"\S+", t)))
    return df


def texts(df: pd.DataFrame, trim: int, ends: bool) -> list[str]:
    t = df["final"].apply(lambda s: " ".join(re.findall(r"\S+", s)[:trim]))
    return (df["from"] + " " + t + " " + df["to"]).tolist() if ends else t.tolist()


def vendi(E: np.ndarray) -> float:
    K = np.clip(cosine_similarity(E), 0, 1) / len(E)
    lam = np.linalg.eigvalsh(K)
    lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))


def group_stats(df: pd.DataFrame, E: np.ndarray, mask: pd.Series, draws: int = 30, k: int = 3) -> dict:
    """Precision, recall and Vendi at the human sample size, averaged over pairs."""
    P, R, V, H = [], [], [], []
    for pair, sub in df.groupby("pair"):
        hi = sub.index[sub["model"] == "human"].to_numpy()
        gi = sub.index[mask.loc[sub.index]].to_numpy()
        if len(hi) < k + 2 or len(gi) < len(hi):
            continue
        ps, rs, vs = [], [], []
        for _ in range(draws):
            pick = RNG.choice(gi, size=len(hi), replace=False)
            p, r = precision_recall(E[hi], E[pick], k)
            ps.append(p); rs.append(r); vs.append(vendi(E[pick]))
        P.append(np.mean(ps)); R.append(np.mean(rs)); V.append(np.mean(vs) / len(hi))
        H.append(sub.loc[gi, "final"].str.lower().str.contains(HUBS[pair]).mean())
    return {"precision": np.mean(P), "recall": np.mean(R), "vendi_share": np.mean(V), "hub_rate": np.mean(H), "pairs": len(P)}


def human_stats(df: pd.DataFrame, E: np.ndarray) -> dict:
    V, H = [], []
    for pair, sub in df.groupby("pair"):
        hi = sub.index[sub["model"] == "human"].to_numpy()
        V.append(vendi(E[hi]) / len(hi))
        H.append(sub.loc[hi, "final"].str.lower().str.contains(HUBS[pair]).mean())
    return {"vendi_share": np.mean(V), "hub_rate": np.mean(H), "n_answers": int((df["model"] == "human").sum()),
            "n_players": int(df.loc[df["model"] == "human", "code"].nunique())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--humans", type=Path, default=ROOT / "data" / "human_bridges.json")
    ap.add_argument("--ai", type=Path, default=ROOT / "data" / "ai")
    ap.add_argument("--tools", type=Path, required=True, help="folder with the embedding models (release 'tools')")
    args = ap.parse_args()
    out = ROOT / "results" / "paper"
    figs = ROOT / "paper" / "figs"
    out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    from sentence_transformers import SentenceTransformer

    df = load(args.humans, args.ai)
    print(f"{(df.model=='human').sum()} human answers, {(df.model!='human').sum()} AI answers (English, core pairs)")
    enc = {}

    def E_for(name, trim, ends):
        key = (name, trim, ends)
        if key not in enc:
            m = SentenceTransformer(str(args.tools / name))
            enc[key] = m.encode(texts(df, trim, ends), normalize_embeddings=True, batch_size=128)
        return enc[key]

    main_E = E_for("all-MiniLM-L6-v2", 40, True)
    hum = human_stats(df, main_E)
    pd.DataFrame([hum]).to_csv(out / "humans.csv", index=False)

    # 1. Models with no persona (first run, temperature 1)
    rows = []
    base_mask = (df["run"] == "fer") & (df["condition"] == "baseline") & (df["temperature"] == 1.0)
    for model, sub in df[base_mask].groupby("model"):
        r = group_stats(df, main_E, base_mask & (df["model"] == model))
        rows.append({"model": model, "name": SHORT.get(model, model), "year": sub["year"].iloc[0], "size_b": sub["size_b"].iloc[0],
                     "kind": sub["model_kind"].iloc[0], "median_words": sub["words"].median(), **r})
    models = pd.DataFrame(rows).sort_values(["year", "size_b"])
    models.to_csv(out / "models_baseline.csv", index=False)
    print(models.round(3).to_string(index=False))

    # 2. Robustness: recall with other rulers
    rob = []
    for name, trim, ends in [("all-MiniLM-L6-v2", 40, True), ("all-MiniLM-L6-v2", 15, False),
                             ("bge-small-en-v1.5", 40, True), ("paraphrase-multilingual-MiniLM-L12-v2", 40, True)]:
        E = E_for(name, trim, ends)
        for model in models["model"]:
            r = group_stats(df, E, base_mask & (df["model"] == model), draws=15)
            rob.append({"embedder": name, "trim": trim, "ends": ends, "model": SHORT.get(model, model), **r})
    pd.DataFrame(rob).to_csv(out / "robustness.csv", index=False)

    # 3. Personas: FER (first run) and creative (second run), against each run's own no-persona answers
    cond = []
    for run in ("fer", "creative"):
        m_run = (df["run"] == run) & (df["temperature"] == 1.0)
        for (model, c), sub in df[m_run & (df["model_kind"] == "chat")].groupby(["model", "condition"]):
            r = group_stats(df, main_E, m_run & (df["model"] == model) & (df["condition"] == c), draws=15)
            cond.append({"run": run, "model": SHORT.get(model, model), "condition": c, **r})
    cond = pd.DataFrame(cond)
    cond.to_csv(out / "conditions.csv", index=False)

    # 4. Temperature
    temps = []
    m_t = df["run"] == "temps"
    for (model, t), sub in df[m_t].groupby(["model", "temperature"]):
        r = group_stats(df, main_E, m_t & (df["model"] == model) & (df["temperature"] == t), draws=15)
        temps.append({"model": SHORT.get(model, model), "temperature": t, **r})
    temps = pd.DataFrame(temps)
    temps.to_csv(out / "temperature.csv", index=False)

    # 5. Blind spots: human answers far from every AI answer (all runs, temperature 1)
    gaps = []
    ai_mask = (df["model"] != "human") & (df["temperature"] == 1.0)
    for pair, sub in df.groupby("pair"):
        hi = sub.index[sub["model"] == "human"].to_numpy()
        ai = sub.index[ai_mask.loc[sub.index]].to_numpy()
        d = cosine_distances(main_E[hi], main_E[ai]).min(axis=1)
        for i, dist in zip(hi, d):
            gaps.append({"pair": pair, "from": df.loc[i, "from"], "to": df.loc[i, "to"], "bridge": df.loc[i, "final"],
                         "words": df.loc[i, "words"], "distance_to_nearest_ai": dist})
    pd.DataFrame(gaps).sort_values("distance_to_nearest_ai", ascending=False).to_csv(out / "blind_spots.csv", index=False)

    figures(models, cond, temps, hum, figs)


def figures(models, cond, temps, hum, figs: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "svg.fonttype": "none"})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figs / f"{name}.pdf"); fig.savefig(figs / f"{name}.svg")
        plt.close(fig)

    # Figure 1: sense vs coverage, one point per model
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for kind, color, label in (("base", BLUE, "Base model (continues text)"), ("chat", ORANGE, "Chat model")):
        s = models[models["kind"] == kind]
        dots(ax, s["recall"], s["precision"], s=36, color=color, edgecolor="white", linewidth=1.5, label=label, zorder=3)
    offsets = {"Qwen2.5 3B": (-62, -3), "Qwen2.5 1.5B": (6, 4), "Phi-3 mini": (6, -2), "Zephyr 7B": (-58, -6),
               "Qwen2.5 0.5B": (6, -8), "OLMo 2 7B": (-58, 2), "Vicuna 7B": (6, 3)}
    for _, r in models.iterrows():
        ax.annotate(f"{r['name']} ({int(r['year'])})", (r["recall"], r["precision"]), xytext=offsets.get(r["name"], (5, 2)),
                    textcoords="offset points", fontsize=7, color=MUTED)
    ax.set_xlabel("Share of the human map reached (recall)")
    ax.set_ylabel("Answers on the human map (precision)")
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, "fig1_precision_recall")

    # Figure 2: coverage across one family's sizes, and base vs chat
    q = models[models["name"].str.startswith("Qwen2.5") & (models["kind"] == "chat")].sort_values("size_b")
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(q["size_b"], q["recall"], color=ORANGE, linewidth=2, marker="o", markersize=6, markeredgecolor="white",
            label="Qwen2.5 chat")
    b = models[models["name"] == "Qwen2.5 7B base"]
    dots(ax, b["size_b"], b["recall"], color=BLUE, s=40, edgecolor="white", linewidth=1.5, zorder=3, label="Qwen2.5 7B base")
    q3 = models[models["name"] == "Qwen3 8B"]
    dots(ax, q3["size_b"], q3["recall"], color=AQUA, s=40, marker="D", edgecolor="white", linewidth=1.5, zorder=3,
               label="Qwen3 8B (2025)")
    ax.set_xscale("log"); ax.set_xticks([0.5, 1.5, 3, 7, 14]); ax.set_xticklabels(["0.5", "1.5", "3", "7", "14"])
    ax.set_xlabel("Parameters (billions)"); ax.set_ylabel("Share of the human map reached")
    ax.set_ylim(0, max(0.6, models["recall"].max() * 0.7)); ax.legend(frameon=False, fontsize=8)
    save(fig, "fig2_scale")

    # Figure 3: change in coverage from each condition, against the same model with no persona
    rows = []
    for run, sub in cond.groupby("run"):
        base = sub[sub["condition"] == "baseline"].set_index("model")["recall"]
        for c, s in sub[sub["condition"] != "baseline"].groupby("condition"):
            d = (s.set_index("model")["recall"] - base).dropna()
            rows.append({"run": run, "condition": c, "delta": d.mean(), "lo": d.min(), "hi": d.max()})
    deltas = pd.DataFrame(rows).sort_values("delta")
    deltas.to_csv(figs.parent.parent / "results" / "paper" / "condition_deltas.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 0.28 * len(deltas) + 0.9))
    y = np.arange(len(deltas))
    colors = [BLUE if r == "fer" else ORANGE for r in deltas["run"]]
    ax.hlines(y, deltas["lo"], deltas["hi"], color=GRID, linewidth=3)
    dots(ax, deltas["delta"], y, color=colors, s=30, zorder=3, edgecolor="white", linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels(deltas["condition"].str.replace("_", " "))
    ax.set_xlabel("Change in share of the map reached, against no persona\n(dot: mean; bar: range across models)")
    dots(ax, [], [], color=BLUE, label="Investigative (FER)"); dots(ax, [], [], color=ORANGE, label="Creative")
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.grid(axis="y", visible=False)
    save(fig, "fig3_conditions")

    # Figure 4: the temperature dial, two panels with their own axes
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), sharex=True)
    for (name, s), color in zip(temps.groupby("model"), (ORANGE, AQUA)):
        s = s.sort_values("temperature")
        axes[0].plot(s["temperature"], s["recall"], color=color, linewidth=2, marker="o", markersize=5, label=name)
        axes[1].plot(s["temperature"], s["precision"], color=color, linewidth=2, marker="o", markersize=5, label=name)
    axes[0].set_title("Coverage (recall)", fontsize=9, color=INK); axes[1].set_title("Answers on the human map (precision)", fontsize=9, color=INK)
    for a in axes:
        a.set_xlabel("Temperature"); a.set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "fig4_temperature")


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        res = ROOT / "results" / "paper"
        figures(pd.read_csv(res / "models_baseline.csv"), pd.read_csv(res / "conditions.csv"),
                pd.read_csv(res / "temperature.csv"), pd.read_csv(res / "humans.csv").iloc[0].to_dict(), ROOT / "paper" / "figs")
    else:
        main()
