"""Generate AI bridges for the same pairs, directions and languages that players see.

Each call gets exactly the instructions shown in the game (data/pairs.json),
optionally preceded by one persona from personas/personas.py. Works with any
OpenAI-compatible endpoint: the Modal vLLM server (generate/serve_model.py),
or a hosted API.

Example:
    export BRIDGES_API_BASE=https://<your-modal-url>/v1
    export BRIDGES_API_KEY=not-needed
    python generate/generate.py --model Qwen/Qwen2.5-7B-Instruct --n 10 --langs en es \
        --out data/ai_bridges.jsonl

Output: one JSON object per line, with the same fields the analysis reads for
human answers (pair, dir, lang, from, to, final, stones) plus source and condition.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from personas.personas import CONDITIONS  # noqa: E402

PAIRS = json.loads((ROOT / "data" / "pairs.json").read_text())

FORMAT = {
    "en": 'Answer as JSON: {"bridge": "your answer", "stones": ["up to three ideas you passed through on the way, optional"]}',
    "fr": 'Réponds en JSON : {"bridge": "ta réponse", "stones": ["jusqu\'à trois idées par lesquelles tu es passé, facultatif"]}',
    "es": 'Responde en JSON: {"bridge": "tu respuesta", "stones": ["hasta tres ideas por las que pasaste, opcional"]}',
    "it": 'Rispondi in JSON: {"bridge": "la tua risposta", "stones": ["fino a tre idee che hai attraversato, facoltativo"]}',
    "de": 'Antworte als JSON: {"bridge": "deine Antwort", "stones": ["bis zu drei Ideen, über die du gegangen bist, optional"]}',
}


def build_prompt(pair: dict, direction: str, lang: str, version: str = "v2") -> tuple[str, str, str]:
    a, b = pair["a"][lang], pair["b"][lang]
    src, dst = (a, b) if direction == "ab" else (b, a)
    ins = PAIRS["instructions"][lang]
    welcome = PAIRS["instructions_versions"][version]["welcome"][lang]
    text = f"{welcome}\n\n{src} → {dst}\n\n{ins['question']}\n\n{FORMAT[lang]}"
    return text, src, dst


def parse(raw: str) -> tuple[str, list[str]]:
    raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            stones = [str(s) for s in obj.get("stones", []) if str(s).strip()][:3]
            return str(obj.get("bridge", "")).strip(), stones
        except json.JSONDecodeError:
            pass
    return raw, []


async def main(args: argparse.Namespace) -> None:
    client = AsyncOpenAI(base_url=os.environ["BRIDGES_API_BASE"], api_key=os.environ.get("BRIDGES_API_KEY", "not-needed"))
    sem = asyncio.Semaphore(args.concurrency)
    pairs = [p for p in PAIRS["pairs"] if args.set == "all" or p["set"] == args.set]
    conditions = args.conditions or list(CONDITIONS)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    async def one(pair, direction, lang, condition, rep):
        prompt, src, dst = build_prompt(pair, direction, lang, args.instructions)
        messages = []
        if CONDITIONS[condition]:
            messages.append({"role": "system", "content": CONDITIONS[condition]})
        messages.append({"role": "user", "content": prompt})
        async with sem:
            for attempt in range(3):
                try:
                    r = await client.chat.completions.create(model=args.model, messages=messages, temperature=args.temperature, max_tokens=300)
                    bridge, stones = parse(r.choices[0].message.content or "")
                    return {"source": "ai", "model": args.model, "condition": condition, "replicate": rep,
                            "pair": pair["id"], "level": pair["level"], "set": pair["set"], "dir": direction,
                            "lang": lang, "from": src, "to": dst, "final": bridge, "stones": stones,
                            "temperature": args.temperature, "instructions": args.instructions}
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        print(f"failed {pair['id']} {direction} {lang} {condition}: {e}", file=sys.stderr)
                        return None
                    await asyncio.sleep(2 * (attempt + 1))

    jobs = [one(p, d, l, c, r) for p in pairs for d in ("ab", "ba") for l in args.langs for c in conditions for r in range(1, args.n + 1)]
    print(f"{len(jobs)} generations")
    done = 0
    with out.open("a") as f:
        for coro in asyncio.as_completed(jobs):
            row = await coro
            done += 1
            if row:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if done % 50 == 0:
                print(f"{done}/{len(jobs)}")
    print(f"saved to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=10, help="replicates per pair, direction, language and condition")
    ap.add_argument("--langs", nargs="+", default=["en"], choices=list(FORMAT))
    ap.add_argument("--conditions", nargs="*", choices=list(CONDITIONS))
    ap.add_argument("--set", default="core", choices=["core", "open", "all"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--instructions", default="v2", choices=["v1", "v2"], help="which welcome text players saw (see data/pairs.json)")
    ap.add_argument("--out", default="data/ai_bridges.jsonl")
    asyncio.run(main(ap.parse_args()))
