"""Run many open models on Modal, from old to new, and save their bridges.

One command, from the repo root:

    modal run --detach generate/ladder_modal.py

It runs every model in MODELS on its own GPU, with the same instructions players
see (data/pairs.json), with no persona and with each persona from
personas/personas.py, plus the ten-word warm-up. With --detach it keeps running
in Modal's cloud after you close the laptop.

In the morning, copy the results into data/ai/:

    modal volume get bridges-results / data/ai/

Options:  --n 10  --langs en,es,fr  --only Qwen/Qwen2.5-7B-Instruct,gpt2-xl
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]

# Old to new, at roughly the same size (about 7B), plus one family at many sizes.
# "base" models only continue text, so they get the prompt as text to finish.
# All of these are open to download without accepting a license.
MODELS = [
    # before chat models
    {"id": "openai-community/gpt2-xl", "year": 2019, "kind": "base", "size": 1.5, "series": "timeline", "max_len": 1024},
    {"id": "EleutherAI/gpt-j-6b", "year": 2021, "kind": "base", "size": 6, "series": "timeline"},
    # first open chat models
    {"id": "tiiuae/falcon-7b-instruct", "year": 2023, "kind": "chat", "size": 7, "series": "timeline"},
    {"id": "lmsys/vicuna-7b-v1.5", "year": 2023, "kind": "chat", "size": 7, "series": "timeline"},
    {"id": "HuggingFaceH4/zephyr-7b-beta", "year": 2023, "kind": "chat", "size": 7, "series": "timeline"},
    # 2024
    {"id": "microsoft/Phi-3-mini-4k-instruct", "year": 2024, "kind": "chat", "size": 3.8, "series": "timeline"},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "year": 2024, "kind": "chat", "size": 7, "series": "timeline,size"},
    {"id": "allenai/OLMo-2-1124-7B-Instruct", "year": 2024, "kind": "chat", "size": 7, "series": "timeline"},
    # 2025
    {"id": "Qwen/Qwen3-8B", "year": 2025, "kind": "chat", "size": 8, "series": "timeline"},
    # one family at many sizes, and its base model
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "year": 2024, "kind": "chat", "size": 0.5, "series": "size"},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "year": 2024, "kind": "chat", "size": 1.5, "series": "size"},
    {"id": "Qwen/Qwen2.5-3B-Instruct", "year": 2024, "kind": "chat", "size": 3, "series": "size"},
    {"id": "Qwen/Qwen2.5-14B-Instruct", "year": 2024, "kind": "chat", "size": 14, "series": "size"},
    {"id": "Qwen/Qwen2.5-7B", "year": 2024, "kind": "base", "size": 7, "series": "base_vs_chat"},
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .uv_pip_install("vllm==0.21.0", "hf_transfer")   # same vLLM as False Epistemic Redundancy
    .add_local_file(str(ROOT / "data" / "pairs.json"), "/root/pairs.json")
    .add_local_file(str(ROOT / "personas" / "personas.py"), "/root/personas.py")
)

app = modal.App("epistemic-bridges-ladder")
results = modal.Volume.from_name("bridges-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("bridges-hf-cache", create_if_missing=True)

WARMUP = {
    "en": "Write 10 words that are as different from each other as you can. Single words only, no names of people or places.",
    "fr": "Écrivez 10 mots aussi différents les uns des autres que possible. Un seul mot à chaque fois, pas de noms de personnes ni de lieux.",
    "es": "Escribe 10 palabras lo más distintas posible entre sí. Solo palabras sueltas, sin nombres de personas ni de lugares.",
    "it": "Scrivi 10 parole il più possibile diverse tra loro. Solo parole singole, niente nomi di persone o luoghi.",
    "de": "Schreib 10 Wörter, die so verschieden voneinander sind wie möglich. Nur einzelne Wörter, keine Namen von Personen oder Orten.",
}


def clean(text: str, kind: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    if kind == "base":   # base models keep writing: keep the first paragraph
        text = text.split("\n\n")[0].strip()
    return text


@app.function(image=image, gpu="L40S", timeout=90 * 60,
              volumes={"/results": results, "/root/.cache/huggingface": hf_cache})
def run_model(spec: dict, n: int, langs: list[str]) -> str:
    import sys
    sys.path.insert(0, "/root")
    from vllm import LLM, SamplingParams
    from personas import CONDITIONS

    pairs = json.load(open("/root/pairs.json"))
    core = [p for p in pairs["pairs"] if p["set"] == "core"]
    kind = spec["kind"]
    conditions = CONDITIONS if kind == "chat" else {"baseline": None}

    jobs = []   # (meta, system, user)
    for lang in langs:
        welcome = pairs["instructions_versions"]["v2"]["welcome"][lang]
        question = pairs["instructions"][lang]["question"]
        for cond, system in conditions.items():
            for p in core:
                for d in ("ab", "ba"):
                    src, dst = (p["a"][lang], p["b"][lang]) if d == "ab" else (p["b"][lang], p["a"][lang])
                    user = f"{welcome}\n\n{src} → {dst}\n\n{question}"
                    jobs.append(({"task": "bridge", "condition": cond, "pair": p["id"], "level": p["level"],
                                  "set": p["set"], "dir": d, "lang": lang, "from": src, "to": dst}, system, user))
            jobs.append(({"task": "warmup", "condition": cond, "lang": lang}, system, WARMUP[lang]))

    llm = LLM(model=spec["id"], max_model_len=spec.get("max_len", 2048), gpu_memory_utilization=0.9,
              trust_remote_code=True, enforce_eager=True)
    sp = SamplingParams(n=n, temperature=1.0, top_p=1.0, max_tokens=220)

    def as_text(system, user):
        head = f"{system}\n\n" if system else ""
        return f"{head}{user}\nAnswer:" if kind == "base" else f"{head}USER: {user}\nASSISTANT:"

    outputs, mode = None, kind
    if kind == "chat":
        convs = [([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": u}] for _, s, u in jobs]
        merged = [[{"role": "user", "content": (f"{s}\n\n{u}" if s else u)}] for _, s, u in jobs]
        for attempt, c in (("chat", convs), ("chat_no_system", merged)):
            try:
                outputs = llm.chat(c, sp, chat_template_kwargs={"enable_thinking": False})
                mode = attempt
                break
            except Exception as e:  # no chat template, or no system role
                print(f"{spec['id']}: {attempt} failed ({e}), trying the next format")
    if outputs is None:
        mode = "completion" if kind == "base" else "completion_fallback"
        outputs = llm.generate([as_text(s, u) for _, s, u in jobs], sp)

    safe = spec["id"].replace("/", "__")
    path = Path(f"/results/{safe}.jsonl")
    with path.open("w") as f:
        for (meta, _, _), out in zip(jobs, outputs):
            for rep, o in enumerate(out.outputs, 1):
                f.write(json.dumps({"source": "ai", "model": spec["id"], "year": spec["year"], "size_b": spec["size"],
                                    "model_kind": kind, "series": spec["series"], "prompt_mode": mode,
                                    "instructions": "v2", "temperature": 1.0, "replicate": rep,
                                    **meta, "final": clean(o.text, kind), "raw": o.text}, ensure_ascii=False) + "\n")
    results.commit()
    return f"{spec['id']}: {len(jobs) * n} answers ({mode})"


@app.local_entrypoint()
def main(n: int = 10, langs: str = "en,es,fr", only: str = ""):
    wanted = [s.strip() for s in only.split(",") if s.strip()]
    specs = [m for m in MODELS if not wanted or m["id"] in wanted or m["id"].split("/")[-1] in wanted]
    print(f"{len(specs)} models, {n} answers per prompt, languages {langs}")
    for r in run_model.map(specs, kwargs={"n": n, "langs": langs.split(",")}, return_exceptions=True):
        print(r)
