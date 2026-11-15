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
          --conditions fer|creative|all|baseline,story,...  --temps 0.7,1.0,1.3  --tag creative

Creative run:      modal run --detach generate/ladder_modal.py --conditions creative --tag creative \
                       --only Qwen2.5-7B-Instruct,Qwen3-8B,OLMo-2-1124-7B-Instruct,zephyr-7b-beta
Temperature dial:  modal run --detach generate/ladder_modal.py --conditions baseline --tag temps \
                       --only Qwen2.5-7B-Instruct --temps 0.4,0.7,1.0,1.3,1.6,2.0
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
    # larger and newer open models, on a bigger GPU
    {"id": "Qwen/Qwen3-32B", "year": 2025, "kind": "chat", "size": 32, "series": "large", "gpu": "H100"},
    {"id": "openai/gpt-oss-20b", "year": 2025, "kind": "chat", "size": 21, "series": "large", "gpu": "H100", "max_len": 8192},
    {"id": "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "year": 2025, "kind": "chat", "size": 24, "series": "large",
     "gpu": "H100", "tokenizer_mode": "mistral"},
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .uv_pip_install("vllm==0.21.0", "hf_transfer")   # same vLLM as False Epistemic Redundancy
    .add_local_file(str(ROOT / "data" / "pairs.json"), "/root/pairs.json")
    .add_local_file(str(ROOT / "personas" / "personas.py"), "/root/personas.py")
    .add_local_file(str(ROOT / "personas" / "creative.py"), "/root/creative.py")
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
    # gpt-oss writes its reasoning first, then the final answer after this marker
    if "assistantfinal" in text:
        text = text.split("assistantfinal", 1)[1].strip()
    elif text.startswith("analysis"):   # ran out of room before the final answer: no answer
        text = ""
    if kind == "base":   # base models keep writing: keep the first paragraph
        text = text.split("\n\n")[0].strip()
    return text


@app.function(image=image, gpu="L40S", timeout=90 * 60,
              volumes={"/results": results, "/root/.cache/huggingface": hf_cache})
def run_model(spec: dict, n: int, langs: list[str], conditions: str = "fer", temps: list[float] | None = None,
              tag: str = "") -> str:
    return _run(spec, n, langs, conditions, temps, tag)


@app.function(image=image, gpu="H100", timeout=120 * 60,
              volumes={"/results": results, "/root/.cache/huggingface": hf_cache})
def run_model_big(spec: dict, n: int, langs: list[str], conditions: str = "fer", temps: list[float] | None = None,
                  tag: str = "") -> str:
    return _run(spec, n, langs, conditions, temps, tag)


def _run(spec: dict, n: int, langs: list[str], conditions: str = "fer", temps: list[float] | None = None,
         tag: str = "") -> str:
    import random
    import sys
    sys.path.insert(0, "/root")
    from vllm import LLM, SamplingParams
    from personas import CONDITIONS
    import creative

    pairs = json.load(open("/root/pairs.json"))
    core = [p for p in pairs["pairs"] if p["set"] == "core"]
    kind = spec["kind"]
    temps = temps or [1.0]

    # Which conditions: "fer" = no persona + the four FER personas, "creative" = no persona +
    # the creative conditions, "all" = both, or a comma-separated list of names.
    fixed = {"baseline": None}
    sampled = []
    if kind == "chat":
        groups = {"fer": dict(CONDITIONS), "creative": {**creative.FIXED}}
        if conditions in ("fer", "all"):
            fixed.update(groups["fer"])
        if conditions in ("creative", "all"):
            fixed.update(groups["creative"])
            sampled = list(creative.SAMPLED)
        if conditions not in ("fer", "creative", "all"):
            every = {**CONDITIONS, **creative.FIXED}
            names = [c.strip() for c in conditions.split(",") if c.strip()]
            fixed = {c: every[c] for c in names if c in every}
            sampled = [c for c in names if c in creative.SAMPLED or c.startswith(creative.COMBO_PREFIX) or c.startswith(creative.STYLE_PREFIX) or c.startswith(creative.LIFE_FER_PREFIX) or c.startswith(creative.WAY_PREFIX)]

    jobs = []   # (meta, system, user, n)
    for temp in temps:
        for lang in langs:
            welcome = pairs["instructions_versions"]["v2"]["welcome"][lang]
            question = pairs["instructions"][lang]["question"]
            tasks = []
            for p in core:
                for d in ("ab", "ba"):
                    src, dst = (p["a"][lang], p["b"][lang]) if d == "ab" else (p["b"][lang], p["a"][lang])
                    tasks.append(({"task": "bridge", "pair": p["id"], "level": p["level"], "set": p["set"], "dir": d,
                                   "lang": lang, "from": src, "to": dst}, f"{welcome}\n\n{src} → {dst}\n\n{question}"))
            tasks.append(({"task": "warmup", "lang": lang}, WARMUP[lang]))
            for meta, user in tasks:
                for cond, system in fixed.items():
                    jobs.append(({**meta, "condition": cond, "temperature": temp, "persona_text": system}, system, user, n))
                for cond in sampled:
                    for rep in range(n):
                        rng = random.Random(f"{cond}|{meta.get('pair', 'warmup')}|{meta.get('dir', '')}|{lang}|{rep}|{temp}")
                        system = creative.sample(cond, rng, CONDITIONS)
                        jobs.append(({**meta, "condition": cond, "temperature": temp, "persona_text": system}, system, user, 1))

    extra = {"tokenizer_mode": spec["tokenizer_mode"]} if spec.get("tokenizer_mode") else {}
    llm = LLM(model=spec["id"], max_model_len=spec.get("max_len", 4096 if spec.get("gpu") else 2048),
              gpu_memory_utilization=0.9, trust_remote_code=True, enforce_eager=True, **extra)
    # Jobs that share the same sampling settings are sent together (a single SamplingParams per call).
    room = 6000 if "gpt-oss" in spec["id"] else 0
    keys = [(k, m["temperature"], room + (320 if m["condition"] == "step_by_step" else 220)) for m, _, _, k in jobs]
    groups: dict = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)

    def run_grouped(call, items):
        out = [None] * len(items)
        for (k, t, mx), idx in groups.items():
            res = call([items[i] for i in idx], SamplingParams(n=k, temperature=t, top_p=1.0, max_tokens=mx))
            for i, r in zip(idx, res):
                out[i] = r
        return out

    def as_text(system, user):
        head = f"{system}\n\n" if system else ""
        return f"{head}{user}\nAnswer:" if kind == "base" else f"{head}USER: {user}\nASSISTANT:"

    outputs, mode = None, kind
    if kind == "chat":
        convs = [([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": u}] for _, s, u, _ in jobs]
        merged = [[{"role": "user", "content": (f"{s}\n\n{u}" if s else u)}] for _, s, u, _ in jobs]
        for attempt, c in (("chat", convs), ("chat_no_system", merged)):
            try:
                outputs = run_grouped(lambda items, sp: llm.chat(items, sp, chat_template_kwargs={"enable_thinking": False}), c)
                mode = attempt
                break
            except Exception as e:  # no chat template, or no system role
                print(f"{spec['id']}: {attempt} failed ({e}), trying the next format")
    if outputs is None:
        mode = "completion" if kind == "base" else "completion_fallback"
        outputs = run_grouped(lambda items, sp: llm.generate(items, sp), [as_text(s, u) for _, s, u, _ in jobs])

    safe = spec["id"].replace("/", "__") + (f"__{tag}" if tag else "")
    path = Path(f"/results/{safe}.jsonl")
    count = 0
    with path.open("w") as f:
        rep_counter: dict = {}
        for (meta, _, _, _), out in zip(jobs, outputs):
            key = (meta["condition"], meta.get("pair"), meta.get("dir"), meta["lang"], meta["task"], meta["temperature"])
            for o in out.outputs:
                rep_counter[key] = rep_counter.get(key, 0) + 1
                text = clean(o.text, kind)
                if meta["condition"] == "step_by_step":
                    m = re.search(r"Bridge:\s*(.+)", text, flags=re.S)
                    text = m.group(1).strip() if m else text
                f.write(json.dumps({"source": "ai", "model": spec["id"], "year": spec["year"], "size_b": spec["size"],
                                    "model_kind": kind, "series": spec["series"], "prompt_mode": mode, "run": tag or "fer",
                                    "instructions": "v2", "replicate": rep_counter[key],
                                    **meta, "final": text, "raw": o.text}, ensure_ascii=False) + "\n")
                count += 1
    results.commit()
    return f"{spec['id']}: {count} answers ({mode})"


@app.local_entrypoint()
def main(n: int = 10, langs: str = "en,es,fr", only: str = "", conditions: str = "fer", temps: str = "1.0",
         tag: str = ""):
    wanted = [s.strip() for s in only.split(",") if s.strip()]
    specs = [m for m in MODELS if not wanted or m["id"] in wanted or m["id"].split("/")[-1] in wanted]
    temp_list = [float(t) for t in temps.split(",")]
    print(f"{len(specs)} models, {n} answers per prompt, languages {langs}, conditions {conditions}, temperatures {temp_list}")
    kwargs = {"n": n, "langs": langs.split(","), "conditions": conditions, "temps": temp_list, "tag": tag}
    small = [s for s in specs if not s.get("gpu")]
    big = [s for s in specs if s.get("gpu")]
    calls = []
    if small:
        calls.append(run_model.map(small, kwargs=kwargs, return_exceptions=True))
    if big:
        calls.append(run_model_big.map(big, kwargs=kwargs, return_exceptions=True))
    for c in calls:
        for r in c:
            print(r)
