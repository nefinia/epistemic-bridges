"""Writes paper/appendix_examples.tex: sample AI bridges for one pair, across models and conditions.
Only AI answers are shown; player answers are never quoted."""
import json, random, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for f in sorted((ROOT / "data" / "ai").glob("*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
PICKS = [("GPT-2 XL (2019, base)", "openai-community/gpt2-xl", "baseline", "fer", 1.0),
         ("Qwen2.5 7B base", "Qwen/Qwen2.5-7B", "baseline", "fer", 1.0),
         ("Qwen2.5 7B chat", "Qwen/Qwen2.5-7B-Instruct", "baseline", "fer", 1.0),
         ("Qwen3 8B (2025)", "Qwen/Qwen3-8B", "baseline", "fer", 1.0),
         ("Qwen2.5 7B, causal mechanist", "Qwen/Qwen2.5-7B-Instruct", "causal_mechanist", "fer", 1.0),
         ("Qwen2.5 7B, ordinary life", "Qwen/Qwen2.5-7B-Instruct", "ordinary_life", "creative", 1.0),
         ("Qwen2.5 7B, memory", "Qwen/Qwen2.5-7B-Instruct", "memory", "creative", 1.0),
         ("Qwen2.5 7B, playful", "Qwen/Qwen2.5-7B-Instruct", "playful", "creative", 1.0),
         ("Qwen2.5 7B, wordplay", "Qwen/Qwen2.5-7B-Instruct", "wordplay", "creative", 1.0),
         ("Qwen2.5 7B, step by step", "Qwen/Qwen2.5-7B-Instruct", "step_by_step", "creative", 1.0),
         ("Qwen2.5 7B, temperature 2.0", "Qwen/Qwen2.5-7B-Instruct", "baseline", "temps", 2.0)]

def tex(s):
    s = s.encode("ascii", "ignore").decode()       # drops non-Latin characters (temperature 2.0 mixes scripts)
    s = s.replace("**", "")
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s

out = []
for label, model, cond, run, temp in PICKS:
    c = [r for r in rows if r["model"] == model and r["condition"] == cond and r.get("run", "fer") == run
         and r["task"] == "bridge" and r["lang"] == "en" and r["pair"] == "c2" and r["dir"] == "ab"
         and float(r.get("temperature", 1.0)) == temp]
    r = random.Random(7).choice(c)
    words = r["final"].split()
    text = tex(" ".join(words[:40])) + (" \\dots" if len(words) > 40 else "")
    extra = f" ({tex(r['persona_text'].split('.')[0])})" if cond == "ordinary_life" else ""
    out.append(f"\\item[\\textbf{{{label}}}{extra}.] {text}")
(ROOT / "paper" / "appendix_examples.tex").write_text("\\begin{description}[leftmargin=0pt, itemsep=0.5em]\n" + "\n".join(out) + "\n\\end{description}\n")
print("\n".join(out)[:1500])
