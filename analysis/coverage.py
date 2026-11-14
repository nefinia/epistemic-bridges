"""Sense, coverage and blind spots: how AI bridges sit in the space of human bridges.

Uses the human answers to define the space people reach, with no ratings and no
model acting as a judge (precision and recall for generative models,
Kynkäänniemi et al. 2019):

- precision: share of a group's answers that fall inside the human space
  (inside the k-nearest-neighbour sphere of some human answer to the same pair)
- recall: share of human answers that fall inside the group's space
- blind spots: human answers that are far from every AI answer

Groups are compared at the size of the human sample for each pair (random
subsamples, averaged), since bigger groups cover more just by being bigger.

    python analysis/coverage.py --humans data/human_bridges.json --ai data/ai \
        --embedder all-MiniLM-L6-v2 --trim 40
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

ROOT = Path(__file__).resolve().parents[1]


def load(humans: Path, ai: Path, trim: int, conditions: list[str] | None, lang: str | None,
         with_ends: bool = False, tag: str | None = None) -> pd.DataFrame:
    h = pd.DataFrame(json.loads(humans.read_text()))
    h["group"], h["model"], h["condition"] = "human", "human", "human"
    files = sorted(ai.glob("*.jsonl")) if ai.is_dir() else [ai]
    a = pd.DataFrame([json.loads(l) for f in files for l in f.read_text().splitlines() if l.strip()])
    a = a[a["task"] == "bridge"]
    if tag is not None and "run" in a:
        a = a[a["run"].fillna("fer") == tag]
    if "temperature" in a:
        a = a[a["temperature"].fillna(1.0) == 1.0]
    if conditions:
        a = a[a["condition"].isin(conditions)]
    a["group"] = a["model"].str.split("/").str[-1] + ":" + a["condition"]
    df = pd.concat([h, a], ignore_index=True)
    df = df[df["set"] == "core"]
    if lang:
        df = df[df["lang"] == lang]
    df["final"] = df["final"].fillna("").astype(str).str.strip()
    df = df[df["final"] != ""]
    if trim:   # same length budget for everyone: the first `trim` words
        df["text"] = df["final"].apply(lambda t: " ".join(re.findall(r"\S+", t)[:trim]))
    else:
        df["text"] = df["final"]
    if with_ends:   # read each answer between its two ideas, so gap-filling answers become full sentences
        df["text"] = df["from"] + " " + df["text"] + " " + df["to"]
    return df.reset_index(drop=True)


def embed(texts: list[str], name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name).encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=True)


def radii(D: np.ndarray, k: int) -> np.ndarray:
    """Distance from each point to its k-th nearest neighbour (itself excluded)."""
    k = min(k, D.shape[0] - 1)
    return np.sort(D, axis=1)[:, k]


def precision_recall(Eh: np.ndarray, Eg: np.ndarray, k: int) -> tuple[float, float]:
    rh = radii(cosine_distances(Eh), k)
    rg = radii(cosine_distances(Eg), k)
    Dgh = cosine_distances(Eg, Eh)
    precision = float((Dgh <= rh[None, :]).any(axis=1).mean())
    recall = float((Dgh.T <= rg[None, :]).any(axis=1).mean())
    return precision, recall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--humans", type=Path, default=ROOT / "data" / "human_bridges.json")
    ap.add_argument("--ai", type=Path, default=ROOT / "data" / "ai")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--trim", type=int, default=40, help="words kept from each answer (0 = all)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--with-ends", action="store_true", help="measure each answer together with its two ideas")
    ap.add_argument("--run", default=None, help="only AI rows from this run tag (fer, creative, temps)")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    df = load(args.humans, args.ai, args.trim, args.conditions, args.lang, args.with_ends, args.run)
    E = embed(df["text"].tolist(), args.embedder)
    rng = np.random.default_rng(0)
    rows, gaps = [], []
    for pair, sub in df.groupby("pair"):
        hi = sub.index[sub["group"] == "human"].to_numpy()
        if len(hi) < args.k + 2:
            continue
        Eh = E[hi]
        for group, gsub in sub[sub["group"] != "human"].groupby("group"):
            gi = gsub.index.to_numpy()
            if len(gi) < len(hi):
                continue
            ps, rs = [], []
            for _ in range(args.draws):
                pick = rng.choice(gi, size=len(hi), replace=False)
                p, r = precision_recall(Eh, E[pick], args.k)
                ps.append(p); rs.append(r)
            rows.append({"pair": pair, "group": group, "model": gsub["model"].iloc[0], "condition": gsub["condition"].iloc[0],
                         "year": gsub["year"].iloc[0], "size_b": gsub["size_b"].iloc[0], "n_human": len(hi),
                         "precision": np.mean(ps), "recall": np.mean(rs)})
        # Blind spots: how close the nearest AI answer (any model, any condition) comes to each human answer.
        ai_idx = sub.index[sub["group"] != "human"].to_numpy()
        if len(ai_idx):
            nearest = cosine_distances(Eh, E[ai_idx]).min(axis=1)
            for i, d in zip(hi, nearest):
                gaps.append({"pair": pair, "dir": df.loc[i, "dir"], "from": df.loc[i, "from"], "to": df.loc[i, "to"],
                             "human_bridge": df.loc[i, "final"], "distance_to_nearest_ai": float(d)})

    args.out.mkdir(exist_ok=True)
    tag = f"{args.embedder.split('/')[-1]}_trim{args.trim}_{args.lang or 'all'}{'_ends' if args.with_ends else ''}_{args.run or 'allruns'}"
    res = pd.DataFrame(rows)
    res.to_csv(args.out / f"coverage_{tag}.csv", index=False)
    gp = pd.DataFrame(gaps).sort_values("distance_to_nearest_ai", ascending=False)
    gp.to_csv(args.out / f"blind_spots_{tag}.csv", index=False)
    summary = res.groupby(["model", "condition", "year", "size_b"])[["precision", "recall"]].mean().reset_index()
    print(summary.sort_values(["year", "size_b"]).round(3).to_string(index=False))
    print("\nHuman bridges farthest from any AI answer:")
    print(gp.head(10)[["from", "to", "human_bridge", "distance_to_nearest_ai"]].to_string(index=False))


if __name__ == "__main__":
    main()
