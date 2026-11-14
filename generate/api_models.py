"""Frontier models through their APIs: Claude, GPT and Gemini, with no persona, in English.

Same text as the players and the open models (data/pairs.json, instructions v2), each core pair
in both directions, 10 answers each, plus the ten-word warm-up, at the provider's default
temperature of 1.0. Each provider runs only if its key is set:

    ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY

The script picks the newest available model from a short preference list per provider (or
the names given with --anthropic / --openai / --gemini), and writes data/ai/<model>.jsonl in
the same format as generate/ladder_modal.py.

    python generate/api_models.py --n 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WARMUP = "Write 10 words that are as different from each other as you can. Single words only, no names of people or places."
PREFER = {
    "anthropic": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-sonnet-4-5", "claude-opus-4-1"],
    "openai": ["gpt-5.5", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1", "gpt-4o"],
    "gemini": ["gemini-3-pro", "gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash"],
}


def prompts() -> list[tuple[dict, str]]:
    pairs = json.loads((ROOT / "data" / "pairs.json").read_text())
    welcome = pairs["instructions_versions"]["v2"]["welcome"]["en"]
    question = pairs["instructions"]["en"]["question"]
    out = []
    for p in pairs["pairs"]:
        if p["set"] != "core":
            continue
        for d in ("ab", "ba"):
            src, dst = (p["a"]["en"], p["b"]["en"]) if d == "ab" else (p["b"]["en"], p["a"]["en"])
            out.append(({"task": "bridge", "pair": p["id"], "level": p["level"], "set": "core", "dir": d, "lang": "en",
                         "from": src, "to": dst}, f"{welcome}\n\n{src} → {dst}\n\n{question}"))
    out.append(({"task": "warmup", "lang": "en"}, WARMUP))
    return out


def retry(fn, tries=5):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # rate limits and passing errors
            if i == tries - 1:
                raise
            print(f"  retry after error: {str(e)[:120]}")
            time.sleep(5 * (i + 1))


def anthropic_client(name):
    import anthropic
    c = anthropic.Anthropic()
    available = {m.id for m in c.models.list(limit=100)}
    model = name or next((m for m in PREFER["anthropic"] if m in available), sorted(available)[-1])
    def ask(text):
        r = c.messages.create(model=model, max_tokens=1024, temperature=1.0, messages=[{"role": "user", "content": text}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return model, ask


def openai_client(name):
    from openai import OpenAI
    c = OpenAI()
    available = {m.id for m in c.models.list()}
    model = name or next((m for m in PREFER["openai"] if m in available), None)
    def ask(text):
        r = c.responses.create(model=model, input=text, max_output_tokens=4000)
        return r.output_text
    return model, ask


def gemini_client(name):
    from google import genai
    c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    available = {m.name.split("/")[-1] for m in c.models.list()}
    model = name or next((m for m in PREFER["gemini"] if m in available), None)
    def ask(text):
        r = c.models.generate_content(model=model, contents=text, config={"temperature": 1.0})
        return r.text or ""
    return model, ask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--anthropic", default="")
    ap.add_argument("--openai", default="")
    ap.add_argument("--gemini", default="")
    args = ap.parse_args()
    providers = [("anthropic", "ANTHROPIC_API_KEY", anthropic_client, args.anthropic),
                 ("openai", "OPENAI_API_KEY", openai_client, args.openai),
                 ("gemini", "GEMINI_API_KEY", gemini_client, args.gemini)]
    out_dir = ROOT / "data" / "ai"
    out_dir.mkdir(parents=True, exist_ok=True)
    for provider, key, make, name in providers:
        if not os.environ.get(key):
            print(f"{provider}: no key, skipped")
            continue
        try:
            model, ask = make(name)
        except Exception as e:
            print(f"{provider}: could not start ({str(e)[:200]})")
            continue
        if not model:
            print(f"{provider}: none of the preferred models is available")
            continue
        print(f"{provider}: {model}")
        path = out_dir / f"{provider}__{model}.jsonl"
        count = 0
        with path.open("w") as f:
            for meta, text in prompts():
                for rep in range(1, args.n + 1):
                    try:
                        raw = retry(lambda: ask(text))
                    except Exception as e:
                        print(f"  failed {meta.get('pair', 'warmup')} {meta.get('dir', '')}: {str(e)[:120]}")
                        continue
                    final = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
                    f.write(json.dumps({"source": "ai", "model": f"{provider}/{model}", "year": 2026, "size_b": None,
                                        "model_kind": "chat", "series": "frontier", "prompt_mode": "api", "run": "fer",
                                        "instructions": "v2", "replicate": rep, **meta, "condition": "baseline",
                                        "temperature": 1.0, "persona_text": None, "final": final, "raw": raw},
                                       ensure_ascii=False) + "\n")
                    count += 1
        print(f"{provider}: {count} answers written to {path.name}")


if __name__ == "__main__":
    main()
