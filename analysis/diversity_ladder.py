"""How much of the change in diversity comes from each step of removing the voice.

Diversity within a pair (different answers per answer, Vendi score over the number of answers)
is measured for people and for chat models with no persona, on four versions of every answer,
each one step further than the last:

1. as written, the full answer;
2. the two ideas of the pair removed (no parameter: the same two words for everyone);
3. the scaffolding also removed (the phrase list in style_and_warmup.py);
3b. the same, with each model answer cut to the length of a player's answer to the same pair;
4. content words only, as in voice_content.py.

Each version is read as chat models' diversity over people's, measured the same way. If chat
models move toward people already at step 2, the change does not depend on the phrase list.

    python analysis/diversity_ladder.py --humans data/human_bridges.json --tools /path/to/tools
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
from levels_directions import content_words, vendi  # noqa: E402
from style_and_warmup import cut, strip_scaffolding  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def without_ideas(text: str, a: str, b: str) -> str:
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", a + " " + b)} - {"a", "an", "the"}
    pat = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in sorted(words)) + r")(?:e?s)?\b", re.I)
    return re.sub(r"\s+", " ", pat.sub(" ", text)).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--humans", type=Path, default=ROOT / "data" / "human_bridges.json")
    ap.add_argument("--ai", type=Path, default=ROOT / "data" / "ai")
    ap.add_argument("--tools", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=30)
    args = ap.parse_args()

    h = pd.DataFrame(json.loads(args.humans.read_text()))
    h = h[(h["lang"] == "en") & (h["set"] == "core")].assign(model="people", kind="people")
    a = pd.DataFrame([json.loads(l) for f in sorted(args.ai.glob("*.jsonl")) for l in f.read_text().splitlines() if l.strip()])
    a = a[(a["task"] == "bridge") & (a["lang"] == "en") & (a["set"] == "core") & (a["condition"] == "baseline")
          & (a["run"].fillna("fer") == "fer") & (a["model_kind"] == "chat")].assign(kind="chat")
    df = pd.concat([h, a], ignore_index=True)
    df["final"] = df["final"].fillna("").astype(str)
    df = df[df["final"].str.strip() != ""].reset_index(drop=True)
    people = (df["model"] == "people").to_numpy()

    cw = content_words()
    rng = np.random.default_rng(0)
    conc = [cw(t, x, y) for t, x, y in zip(df["final"], df["from"], df["to"])]
    hconc = {p: [len(c) for c, pp, ip in zip(conc, df["pair"], people) if ip and pp == p and len(c)] for p in df["pair"].unique()}
    hwords = {p: [len(t.split()) for t, pp, ip in zip(df["final"], df["pair"], people) if ip and pp == p] for p in df["pair"].unique()}
    stripped = [without_ideas(strip_scaffolding(t), x, y) for t, x, y in zip(df["final"], df["from"], df["to"])]
    versions = {
        "1 as written": df["final"].tolist(),
        "2 ideas removed": [without_ideas(t, x, y) for t, x, y in zip(df["final"], df["from"], df["to"])],
        "3 scaffolding removed": stripped,
        "3b also cut to a player's length": [t if ip else cut(t, int(rng.choice(hwords[p])))
                                             for t, ip, p in zip(stripped, people, df["pair"])],
        "4 content words": [" ".join(c if ip or not len(c) else c[: int(rng.choice(hconc[p]))]) for c, ip, p in zip(conc, people, df["pair"])],
    }

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(str(args.tools / "all-MiniLM-L6-v2"))
    rows = []
    for name, texts in versions.items():
        ok = np.array([len(t.strip()) > 0 for t in texts])
        E = np.zeros((len(texts), 384), dtype=np.float32)
        E[ok] = st.encode([t for t, k in zip(texts, ok) if k], normalize_embeddings=True, batch_size=256)
        for pair, sub in df[ok].groupby("pair"):
            hi = sub.index[sub["model"] == "people"].to_numpy()
            n = len(hi)
            same = E[hi] @ E[hi].T
            ppl = vendi(E[hi]) / n
            chat = []
            for model, g in sub[sub["model"] != "people"].groupby("model"):
                gi = g.index.to_numpy()
                chat.append(np.mean([vendi(E[pick]) / len(pick)
                                     for pick in (rng.choice(gi, size=min(n, len(gi)), replace=False) for _ in range(args.draws))]))
            rows.append({"version": name, "pair": pair, "people_different": ppl, "chat_different": float(np.mean(chat)),
                         "chat_over_people": float(np.mean(chat)) / ppl,
                         "people_similarity": float(same[np.triu_indices(n, 1)].mean())})
    res = pd.DataFrame(rows)
    out = ROOT / "results" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "diversity_ladder.csv", index=False)
    summ = res.groupby("version").agg(people=("people_different", "mean"), chat=("chat_different", "mean"),
                                      ratio_mean=("chat_over_people", "mean"), ratio_min=("chat_over_people", "min"),
                                      ratio_max=("chat_over_people", "max"), people_similarity=("people_similarity", "mean"))
    summ.to_csv(out / "diversity_ladder_summary.csv")
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main()
