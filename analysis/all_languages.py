"""The main comparison with every language together (English, Spanish, French).

Players answered in their own language, and the models answered the same pairs in each
language. Here all human answers form the map, measured with a multilingual embedding
model, and each model is placed on it with the same mix of languages as the players
(for each pair, as many answers in each language as the players gave). Core pairs,
no persona, answers read in full between their two ideas.

    python analysis/all_languages.py --humans data/human_bridges.json --tools /path/to/tools
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--humans", type=Path, default=ROOT / "data" / "human_bridges.json")
    ap.add_argument("--ai", type=Path, default=ROOT / "data" / "ai")
    ap.add_argument("--tools", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    h = pd.DataFrame(json.loads(args.humans.read_text()))
    h = h[h["set"] == "core"].assign(model="people")
    a = pd.DataFrame([json.loads(l) for f in sorted(args.ai.glob("*.jsonl")) for l in f.read_text().splitlines() if l.strip()])
    a = a[(a["task"] == "bridge") & (a["set"] == "core") & (a["condition"] == "baseline") & (a["run"].fillna("fer") == "fer")]
    df = pd.concat([h, a], ignore_index=True)
    df["final"] = df["final"].fillna("").astype(str)
    df = df[df["final"].str.strip() != ""].reset_index(drop=True)
    text = df["from"] + " " + df["final"] + " " + df["to"]
    from sentence_transformers import SentenceTransformer
    E = SentenceTransformer(str(args.tools / "paraphrase-multilingual-MiniLM-L12-v2")).encode(
        text.tolist(), normalize_embeddings=True, batch_size=128)

    rng = np.random.default_rng(0)
    rows = []
    for pair, sub in df.groupby("pair"):
        hi = sub.index[sub["model"] == "people"].to_numpy()
        mix = df.loc[hi, "lang"].value_counts()
        for model, g in sub[sub["model"] != "people"].groupby("model"):
            pools = {lang: g.index[g["lang"] == lang].to_numpy() for lang in mix.index}
            if any(len(pools[lang]) < n for lang, n in mix.items()):
                continue
            ps, rs = [], []
            for _ in range(args.draws):
                pick = np.concatenate([rng.choice(pools[lang], size=n, replace=False) for lang, n in mix.items()])
                p, r = precision_recall(E[hi], E[pick], args.k)
                ps.append(p); rs.append(r)
            rows.append({"pair": pair, "model": model, "kind": g["model_kind"].iloc[0], "n_human": len(hi),
                         "precision": np.mean(ps), "recall": np.mean(rs)})
    res = pd.DataFrame(rows).groupby(["model", "kind"])[["precision", "recall"]].mean().reset_index()
    out = ROOT / "results" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "all_languages.csv", index=False)
    print(df[df["model"] == "people"]["lang"].value_counts().to_dict())
    print(res.sort_values("recall").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
