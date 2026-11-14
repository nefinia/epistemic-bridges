"""Two checks on what the main measure picks up.

1. Style or ideas. Chat answers share a scaffolding ("Here's one way to connect...",
   numbered steps, bold titles, a closing summary). Here that scaffolding is removed before
   measuring: markdown and list markers go, and sentences that only talk about the task
   (announcing, framing or summing up the connection) are dropped. Then each answer is cut
   to 40 words and measured as in the main analysis. The same cleaning is applied to the
   players' answers, so both sides are treated alike. If chat models still reach little of
   the human map, the narrowing is in the ideas.

2. Warm-up and bridges, for people. For each English player: their warm-up score (DAT),
   how far their bridges sit from the other players' bridges for the same pair, and how far
   they sit from the nearest AI answer. Spearman correlations, with the small sample in mind.

    python analysis/style_and_warmup.py --humans data/human_bridges.json --warmups data/warmups_en.json --tools /path/to/tools
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
from dat import load_glove, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
META = re.compile(
    r"here['’]?s|here is|let['’]?s|let us|this gap|one way|one possible|a possible|to (?:get|go|move|travel|bridge|connect|link|create|draw|find|make)\b"
    r"|seemingly|may seem|might seem|at first glance|unrelated|conceptual (?:leap|bridge|link)|creative(?:ly)? (?:connection|link|leap|bridge)"
    r"|we can|we could|i would|i['’]d|could be (?:bridged|connected|linked)|the (?:connection|link|bridge|journey|path) (?:between|from)"
    r"|in (?:summary|conclusion|essence|short)|in this way|this (?:way|journey|path|connection|sequence|chain)|^thus|^so,|^overall|^ultimately",
    re.I)


def strip_scaffolding(text: str) -> str:
    t = re.sub(r"[*_#`>]+", " ", str(text))
    t = re.sub(r"(?m)^\s*(?:\d+[.)]|[-•]|step \d+:?)\s*", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    sents = re.split(r"(?<=[.!?:])\s+", t)
    keep = [s for s in sents if not META.search(s)]
    return " ".join(keep) if keep else t


def cut(s: str, n: int = 40) -> str:
    return " ".join(re.findall(r"\S+", str(s))[:n])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--humans", type=Path, default=ROOT / "data" / "human_bridges.json")
    ap.add_argument("--warmups", type=Path, default=ROOT / "data" / "warmups_en.json")
    ap.add_argument("--ai", type=Path, default=ROOT / "data" / "ai")
    ap.add_argument("--tools", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=30)
    args = ap.parse_args()

    h = pd.DataFrame(json.loads(args.humans.read_text()))
    h = h[(h["lang"] == "en") & (h["set"] == "core")].assign(model="people", model_kind="people")
    a = pd.DataFrame([json.loads(l) for f in sorted(args.ai.glob("*.jsonl")) for l in f.read_text().splitlines() if l.strip()])
    a = a[(a["task"] == "bridge") & (a["lang"] == "en") & (a["set"] == "core") & (a["condition"] == "baseline")
          & (a["run"].fillna("fer") == "fer")]
    df = pd.concat([h, a], ignore_index=True)
    df["final"] = df["final"].fillna("").astype(str)
    df = df[df["final"].str.strip() != ""].reset_index(drop=True)
    df["stripped"] = df["final"].apply(strip_scaffolding)  # the same cleaning for people and models

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(str(args.tools / "all-MiniLM-L6-v2"))
    enc = lambda col: st.encode((df["from"] + " " + df[col].apply(cut) + " " + df["to"]).tolist(),
                                normalize_embeddings=True, batch_size=128)
    E = {"as written": enc("final"), "scaffolding removed": enc("stripped")}

    # 1. Style or ideas
    rng = np.random.default_rng(0)
    rows = []
    for version, X in E.items():
        for pair, sub in df.groupby("pair"):
            hi = sub.index[sub["model"] == "people"].to_numpy()
            for model, g in sub[sub["model"] != "people"].groupby("model"):
                gi = g.index.to_numpy()
                ps, rs = zip(*[precision_recall(X[hi], X[rng.choice(gi, size=len(hi), replace=False)], 3)
                               for _ in range(args.draws)])
                rows.append({"version": version, "model": model, "kind": g["model_kind"].iloc[0],
                             "precision": np.mean(ps), "recall": np.mean(rs)})
    res = pd.DataFrame(rows).groupby(["model", "kind", "version"])[["precision", "recall"]].mean().unstack("version")
    res.columns = [f"{m} ({v})" for m, v in res.columns]
    ai = df["model"] != "people"
    share = df["stripped"].str.split().str.len() / df["final"].str.split().str.len()
    kept = share.groupby(df["model_kind"]).median()
    changed = (df["stripped"].str.split().str.join(" ") != df["final"].str.split().str.join(" ")).groupby(df["model_kind"]).mean()
    print("share of answers changed by the cleaning:", changed.round(2).to_dict())
    out = ROOT / "results" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "style_check.csv")
    print("median share of words kept after removing scaffolding:", kept.round(2).to_dict())
    print(res.round(2).to_string())

    # 2. Warm-up and bridges, for people
    idx, V = load_glove(args.tools)
    warm = {code: score(words, idx, V) for code, words, _ in json.loads(args.warmups.read_text())}
    X = E["as written"]
    per = []
    for i in df.index[df["model"] == "people"]:
        pair = df.loc[i, "pair"]
        others = df.index[(df["model"] == "people") & (df["pair"] == pair) & (df["code"] != df.loc[i, "code"])]
        machines = df.index[(df["model"] != "people") & (df["pair"] == pair)]
        per.append({"code": df.loc[i, "code"], "from_people": float(1 - (X[others] @ X[i]).mean()),
                    "from_ai": float(1 - (X[machines] @ X[i]).max())})
    p = pd.DataFrame(per).groupby("code").mean()
    p["warmup"] = p.index.map(warm)
    p = p.dropna()
    from scipy.stats import spearmanr
    lines = []
    for col, name in (("from_people", "distance from other players' bridges"), ("from_ai", "distance from the nearest AI answer")):
        r, pv = spearmanr(p["warmup"], p[col])
        lines.append({"bridges": name, "players": len(p), "spearman_r": r, "p": pv})
    corr = pd.DataFrame(lines)
    corr.to_csv(out / "warmup_vs_bridges.csv", index=False)
    print(corr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
