"""Epistemic diversity of bridges, for humans and for AI.

Reads human answers (data/human_bridges.json, from tools/export_humans.sh) and
AI answers (data/ai_bridges.jsonl, from generate/generate.py), groups them by
pair, direction and source, and reports established diversity measures.
Nothing here asks a model to judge creativity: embedding models are used only
as fixed measuring tools, and every semantic measure has a model-free
companion. See docs/methods.md for what each number means and where it comes from.

    python analysis/diversity.py                       # sentence-transformers embeddings
    python analysis/diversity.py --embedder tfidf      # no model download, for a quick look
    python analysis/diversity.py --embedder all-MiniLM-L6-v2 paraphrase-multilingual-MiniLM-L12-v2

Writes results/diversity_by_group.csv and results/answers_with_rarity.csv.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DISTANCE = 0.4      # same threshold as False Epistemic Redundancy (analysis/tail_recovery.py)
MIN_WORDS = 3               # shorter answers are kept but flagged as fragments


# ---------- loading ----------

def load_humans(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.DataFrame(json.loads(path.read_text()))
    df["source"], df["condition"] = "human", "human"
    df["stones"] = df["stones"].apply(lambda s: json.loads(s) if isinstance(s, str) else (s or []))
    return df


def load_ai(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])


def words(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).lower())


# ---------- embeddings ----------

def embed(texts: list[str], name: str) -> np.ndarray:
    if name == "tfidf":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(texts).toarray()
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name).encode(texts, normalize_embeddings=True)


# ---------- measures ----------

def mean_pairwise_distance(E: np.ndarray) -> float:
    """Mean cosine distance between all answers (the DAT-style spread)."""
    n = len(E)
    if n < 2:
        return float("nan")
    S = cosine_similarity(E)
    return float(1 - S[~np.eye(n, dtype=bool)].mean())


def vendi_score(E: np.ndarray) -> float:
    """Effective number of distinct answers (Friedman and Dieng 2023):
    exp of the entropy of the eigenvalues of the normalised similarity matrix."""
    n = len(E)
    if n < 2:
        return float(n)
    K = np.clip(cosine_similarity(E), 0, 1) / n
    lam = np.linalg.eigvalsh(K)
    lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))


def clusters(E: np.ndarray) -> np.ndarray:
    if len(E) < 2:
        return np.zeros(len(E), dtype=int)
    return AgglomerativeClustering(n_clusters=None, distance_threshold=CLUSTER_DISTANCE,
                                   metric="cosine", linkage="average").fit_predict(E)


def hill_numbers(labels: np.ndarray) -> dict:
    """Effective number of idea families (Jost 2006): q=0 counts families,
    q=1 is exp(Shannon), q=2 is inverse Simpson."""
    counts = np.array(list(Counter(labels).values()), dtype=float)
    p = counts / counts.sum()
    return {"families_q0": float(len(p)),
            "families_q1": float(np.exp(-(p * np.log(p)).sum())),
            "families_q2": float(1 / (p ** 2).sum())}


def distinct_n(texts: list[str], n: int) -> float:
    """Share of word n-grams that are unique across the group (model-free)."""
    grams = [tuple(w[i:i + n]) for t in texts for w in [words(t)] for i in range(len(w) - n + 1)]
    return float(len(set(grams)) / len(grams)) if grams else float("nan")


def stone_overlap(stone_lists: list[list[str]]) -> float:
    """Mean Jaccard overlap of stepping stones between answers that have any."""
    sets = [set(s.strip().lower() for s in st if s.strip()) for st in stone_lists if st]
    if len(sets) < 2:
        return float("nan")
    vals = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:] if a | b]
    return float(np.mean(vals)) if vals else float("nan")


# ---------- main ----------

def run(embedders: list[str], humans: Path, ai: Path, with_ends: bool) -> None:
    df = pd.concat([load_humans(humans), load_ai(ai)], ignore_index=True)
    if df.empty:
        print("No data yet: run tools/export_humans.sh and/or generate/generate.py first.")
        return
    df["final"] = df["final"].fillna("").astype(str).str.strip()
    df = df[df["final"] != ""].reset_index(drop=True)
    df["n_words"] = df["final"].apply(lambda t: len(words(t)))
    df["fragment"] = df["n_words"] < MIN_WORDS
    # Some players fill the gap between the two ideas ("[answer removed]" -> a bee).
    # With --with-ends each answer is measured as "from + answer + to", so those read as full sentences.
    df["text"] = (df["from"] + " " + df["final"] + " " + df["to"]) if with_ends else df["final"]
    print(f"{len(df)} answers: " + ", ".join(f"{k} {v}" for k, v in df["condition"].value_counts().items()))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rows = []

    for emb_name in embedders:
        E = embed(df["text"].tolist(), emb_name)
        tag = emb_name.split("/")[-1]
        # Rarity: within each pair, how far an answer sits from the human crowd (1 - mean similarity).
        for pair, idx in df.groupby("pair").groups.items():
            idx = np.array(list(idx))
            human_idx = idx[df.loc[idx, "source"].eq("human").to_numpy()]
            if len(human_idx) >= 2:
                S = cosine_similarity(E[idx], E[human_idx])
                for k, i in enumerate(idx):
                    others = [j for j, h in enumerate(human_idx) if h != i]
                    if others:
                        df.loc[i, f"rarity_{tag}"] = 1 - S[k, others].mean()
        # Group measures: per pair x direction x condition, and pooled over directions.
        for keys, sub in list(df.groupby(["pair", "dir", "condition"])) + \
                         [((p, "both", c), s) for (p, c), s in df.groupby(["pair", "condition"])]:
            pair, direction, condition = keys
            E_sub = E[sub.index.to_numpy()]
            rows.append({"embedder": tag, "pair": pair, "dir": direction, "condition": condition,
                         "n": len(sub), "fragments": int(sub["fragment"].sum()),
                         "mean_distance": mean_pairwise_distance(E_sub),
                         "vendi": vendi_score(E_sub),
                         **hill_numbers(clusters(E_sub)),
                         "distinct_1": distinct_n(sub["final"].tolist(), 1),
                         "distinct_2": distinct_n(sub["final"].tolist(), 2),
                         "stone_overlap": stone_overlap(sub["stones"].tolist()),
                         "answers_with_stones": int(sub["stones"].apply(bool).sum())})
        # Direction: distance between the centre of a->b answers and the centre of b->a answers.
        for (pair, condition), sub in df.groupby(["pair", "condition"]):
            ab, ba = sub[sub["dir"] == "ab"].index, sub[sub["dir"] == "ba"].index
            if len(ab) and len(ba):
                c1, c2 = E[ab].mean(0, keepdims=True), E[ba].mean(0, keepdims=True)
                rows.append({"embedder": tag, "pair": pair, "dir": "ab_vs_ba", "condition": condition,
                             "n": len(sub), "direction_shift": float(1 - cosine_similarity(c1, c2)[0, 0])})

    res = pd.DataFrame(rows)
    res.to_csv(out / "diversity_by_group.csv", index=False)
    keep = [c for c in ["source", "condition", "pair", "dir", "lang", "from", "to", "final", "stones", "fragment"] if c in df] + \
           [c for c in df.columns if c.startswith("rarity_")]
    df[keep].to_csv(out / "answers_with_rarity.csv", index=False)

    pooled = res[res["dir"] == "both"].groupby(["embedder", "condition"])[
        ["n", "mean_distance", "vendi", "families_q1", "distinct_1", "stone_overlap"]].mean(numeric_only=True)
    print("\nAverage over pairs (directions pooled):")
    print(pooled.round(3).to_string())
    print(f"\nSaved {out / 'diversity_by_group.csv'} and {out / 'answers_with_rarity.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", nargs="+", default=["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
                    help="sentence-transformers model names, or 'tfidf' for a model-free check")
    ap.add_argument("--humans", type=Path, default=ROOT / "data" / "human_bridges.json")
    ap.add_argument("--ai", type=Path, default=ROOT / "data" / "ai_bridges.jsonl")
    ap.add_argument("--with-ends", action="store_true", help="measure each answer together with its two ideas")
    a = ap.parse_args()
    run(a.embedder, a.humans, a.ai, a.with_ends)
