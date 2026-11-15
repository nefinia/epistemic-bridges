"""Voice and ideas, measured apart.

Measured as written, chat answers mix two things: a shared assistant voice (announcing the
task, numbered steps, a closing summary) and the connection itself. This script measures
each on its own, for people, for every model with no persona, and for every persona, creative condition and
temperature (English, core pairs).

Voice. Each model answer is cut to a length drawn from the players' answers to the same
pair, and for each pair as many answers are drawn as players gave. The set is then
compressed, on the words and on the sequence of parts of speech (Shaib et al. 2024): the
compression ratio is the original size over the compressed size, so a set that repeats
the same wording or the same sentence patterns compresses more and scores higher.

Ideas. The scaffolding is removed from every answer in the same way (style_and_warmup.py),
and each answer is reduced to its content words (nouns, verbs and adjectives, lemmatised),
leaving out the two ideas of the pair and a few words that only name the task (connect,
link, bridge...). Model answers keep as many content words as a player's answer to the same
pair, drawn at random. The content words are embedded and compared on the human map as in
the main analysis, half against half: a random half of the players draws the map, and the
other half of the players, or as many model answers, is placed on it (k = 2, 200 draws).

    python analysis/voice_content.py --humans data/human_bridges.json --tools /path/to/tools
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage import precision_recall  # noqa: E402
from style_and_warmup import cut, strip_scaffolding  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TASK_WORDS = {"connect", "connection", "link", "bridge", "idea", "concept", "relate", "relationship", "path", "way",
              "step", "start", "go", "get", "lead", "think", "seem", "consider", "make", "use", "be", "have", "do"}
NAMES = {"openai-community/gpt2-xl": "GPT-2 XL", "EleutherAI/gpt-j-6b": "GPT-J 6B", "lmsys/vicuna-7b-v1.5": "Vicuna 7B",
         "HuggingFaceH4/zephyr-7b-beta": "Zephyr 7B", "microsoft/Phi-3-mini-4k-instruct": "Phi-3 mini",
         "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5 0.5B", "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5 1.5B",
         "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5 3B", "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5 7B",
         "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5 14B", "Qwen/Qwen2.5-7B": "Qwen2.5 7B base",
         "allenai/OLMo-2-1124-7B-Instruct": "OLMo 2 7B", "Qwen/Qwen3-8B": "Qwen3 8B", "Qwen/Qwen3-32B": "Qwen3 32B", "openai/gpt-oss-20b": "gpt-oss 20B"}


def ratio(text: str) -> float:
    b = text.encode("utf8")
    return len(b) / len(gzip.compress(b, mtime=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--humans", type=Path, default=ROOT / "data" / "human_bridges.json")
    ap.add_argument("--ai", type=Path, default=ROOT / "data" / "ai")
    ap.add_argument("--tools", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0, help="seed for the draws and the cut of content words")
    ap.add_argument("--drop-player", default="", help="leave this player out (for the leave-one-out check)")
    ap.add_argument("--plain", action="store_true", help="only models with no persona")
    args = ap.parse_args()

    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stop = set(stopwords.words("english"))
    lem = WordNetLemmatizer()

    h = pd.DataFrame(json.loads(args.humans.read_text()))
    h = h[(h["lang"] == "en") & (h["set"] == "core")].assign(model="people", model_kind="people")
    if args.drop_player:
        h = h[h["code"] != args.drop_player]
    a = pd.DataFrame([json.loads(l) for f in sorted(args.ai.glob("*.jsonl")) for l in f.read_text().splitlines() if l.strip()])
    run = a["run"].fillna("fer")
    a = a[(a["task"] == "bridge") & (a["lang"] == "en") & (a["set"] == "core")
          & run.isin(["fer", "creative", "temps", "combo", "lifestyle"])]
    if args.plain:
        a = a[(a["run"].fillna("fer") == "fer") & (a["condition"] == "baseline")]
    df = pd.concat([h, a], ignore_index=True)
    df["final"] = df["final"].fillna("").astype(str)
    df = df[df["final"].str.strip() != ""].reset_index(drop=True)
    people = df["model"] == "people"
    def group(m, c, r, temp):
        if m == "people" or (c == "baseline" and r == "fer"):
            return m
        if r == "temps":
            return f"{m} | T {temp:.1f}"
        return f"{m} | {r}:{c}"
    df["group"] = [group(m, c, r, tp) for m, c, r, tp in zip(df["model"], df["condition"].fillna("baseline"),
                                                         df["run"].fillna("fer"), df["temperature"].fillna(1.0))]

    def content(text: str, ideas: set[str]) -> list[str]:
        toks = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
        out = []
        for w, tag in nltk.pos_tag(toks):
            pos = "n" if tag.startswith("NN") else "v" if tag.startswith("VB") else "a" if tag.startswith("JJ") else None
            if pos is None:
                continue
            l = lem.lemmatize(w.lower(), pos)
            if l in stop or l in TASK_WORDS or l in ideas or len(l) < 3:
                continue
            out.append(l)
        return out

    ideas = {r: {lem.lemmatize(w.lower(), "n") for w in re.findall(r"[A-Za-z]+", f + " " + t)}
             for r, (f, t) in enumerate(zip(df["from"], df["to"]))}
    df["concepts"] = [content(strip_scaffolding(s), ideas[i]) for i, s in enumerate(df["final"])]
    df["n_words"] = df["final"].str.split().str.len()
    df["n_concepts"] = df["concepts"].str.len()

    rng = np.random.default_rng(args.seed)
    pairs = sorted(df.loc[people, "pair"].unique())
    hwords = {p: df.loc[people & (df["pair"] == p), "n_words"].to_numpy() for p in pairs}
    hconc = {p: df.loc[people & (df["pair"] == p) & (df["n_concepts"] > 0), "n_concepts"].to_numpy() for p in pairs}

    # Voice: compression of a set of answers at the players' length
    voice = []
    ptxt = df.loc[people, "final"].tolist()
    voice.append({"model": "people", "kind": "people", "cr_words": ratio("\n".join(ptxt)),
                  "cr_pos": ratio("\n".join(" ".join(t for _, t in nltk.pos_tag(s.split())) for s in ptxt))})
    for model, g in df[~people].groupby("group"):
        if any((g["pair"] == p).sum() < 10 for p in pairs):
            print(f"{model}: too few answers for some pair, left out")
            continue
        cw, cp = [], []
        for _ in range(30):
            texts = []
            for p in pairs:
                gi = g.index[g["pair"] == p].to_numpy()
                for i in rng.choice(gi, size=min(len(hwords[p]), len(gi)), replace=False):
                    texts.append(cut(df.loc[i, "final"], int(rng.choice(hwords[p]))))
            cw.append(ratio("\n".join(texts)))
            cp.append(ratio("\n".join(" ".join(t for _, t in nltk.pos_tag(s.split())) for s in texts)))
        voice.append({"model": model, "kind": g["model_kind"].iloc[0], "cr_words": np.mean(cw), "cr_pos": np.mean(cp)})
    voice = pd.DataFrame(voice)

    # Ideas: content words on the human map, half against half
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(str(args.tools / "all-MiniLM-L6-v2"))
    keep = df["n_concepts"] > 0
    # model answers keep as many content words as a player's answer to the same pair (drawn once per answer)
    df["concept_text"] = [" ".join(c if m == "people" or not len(c) else c[: int(rng.choice(hconc[p]))])
                          for c, m, p in zip(df["concepts"], df["model"], df["pair"])]
    E = st.encode(df["concept_text"].tolist(), normalize_embeddings=True, batch_size=256)
    rows = []
    for model, g in [("people", df[people & keep])] + list(df[~people & keep].groupby("group")):
        if model != "people" and model not in set(voice["model"]):
            continue
        P, R = [], []
        for p in pairs:
            hi = df.index[people & keep & (df["pair"] == p)].to_numpy()
            gi = g.index[g["pair"] == p].to_numpy()
            ps, rs = [], []
            for _ in range(args.draws):
                perm = rng.permutation(hi)
                ref, other = perm[: len(hi) // 2], perm[len(hi) // 2:]
                pick = other if model == "people" else rng.choice(gi, size=min(len(other), len(gi)), replace=False)
                pr, rc = precision_recall(E[ref], E[pick], 2)
                ps.append(pr); rs.append(rc)
            P.append(np.mean(ps)); R.append(np.mean(rs))
        rows.append({"model": model, "on_map_ideas": np.mean(P), "map_reached_ideas": np.mean(R)})
    ideas_res = pd.DataFrame(rows)

    res = voice.merge(ideas_res, on="model")
    res["share_with_content"] = res["model"].map(df.groupby("group")["n_concepts"].apply(lambda s: (s > 0).mean()))
    res["name"] = [NAMES.get(m.split(" | ")[0], m.split(" | ")[0]) + (" | " + m.split(" | ")[1] if " | " in m else "") for m in res["model"]]
    out = ROOT / "results" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    name = f"voice_content_drop_{args.drop_player}.csv" if args.drop_player else f"voice_content_seed{args.seed}.csv"
    res.to_csv(out / name, index=False)
    print(res[["name", "kind", "cr_words", "cr_pos", "on_map_ideas", "map_reached_ideas", "share_with_content"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
