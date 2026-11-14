"""Writes paper/appendix_conditions.tex: the exact texts given to the models (game instructions, personas,
creative conditions, ordinary lives, constraints) and the robustness table."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from personas.personas import PERSONAS
from personas import creative

def tex(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("->", r"$\to$"), ("\n", " ")]:
        s = s.replace(a, b)
    return s

pairs = json.loads((ROOT / "data" / "pairs.json").read_text())
ins = pairs["instructions"]["en"]
L = []
L.append(r"\subsection*{The game text}")
L.append(r"Players and models received the same text, here in English (the game also runs in French, Spanish, Italian and German). For a model, the pair was written as ``salt $\to$ a clock''.")
L.append(r"\begin{quote}\small " + tex(pairs["instructions_versions"]["v2"]["welcome"]["en"]) + r"\\[0.3em] salt $\to$ a clock \\[0.3em] " + tex(ins["question"]) + r"\end{quote}")
L.append(r"The first version of the welcome text, seen by players before 14 November 00:30 Paris time, read: ``" + tex(pairs["instructions_versions"]["v1"]["welcome"]["en"]) + "''. Base models received the same text followed by ``Answer:'', to be continued.")
L.append(r"\subsection*{Generation settings}")
L.append(r"vLLM 0.21 on one NVIDIA L40S GPU per model (Modal), temperature 1.0 unless stated, top-$p$ 1.0, at most 220 new tokens (320 for step by step, whose answer was read from a final line starting with ``Bridge:''), 10 answers per pair, direction, language and condition. Thinking mode was switched off for Qwen3. Chat models received the persona as a system message; where a model's chat template had no system role, the persona was placed before the question. Vicuna 7B had no chat template and received a plain ``USER: \dots\ ASSISTANT:'' format.")
L.append(r"\subsection*{Reasoning personas (from false epistemic redundancy, by Anya Habana)}")
L.append(r"\begin{description}[leftmargin=0pt, itemsep=0.4em]")
for k, v in PERSONAS.items():
    L.append(rf"\item[\textbf{{{tex(k.replace('_', ' '))}}}.] \small {tex(v)}")
L.append(r"\end{description}")
L.append(r"\subsection*{Creative conditions}")
L.append(r"\begin{description}[leftmargin=0pt, itemsep=0.4em]")
for k, v in creative.FIXED.items():
    L.append(rf"\item[\textbf{{{tex(k.replace('_', ' '))}}}.] \small {tex(v)}")
L.append(r"\item[\textbf{ordinary life}.] \small ``You are [one sketch, drawn at random for each answer]. Answer as yourself, in your own words.''")
L.append(r"\item[\textbf{random constraint}.] \small One sentence drawn at random for each answer from the list below.")
L.append(r"\end{description}")
L.append(r"\noindent\textbf{The 30 ordinary lives.} \small " + "; ".join(tex(x) for x in creative.LIVES) + ".")
L.append("")
L.append(r"\noindent\textbf{The 12 constraints.} \small " + " ".join(tex(x) for x in creative.CONSTRAINTS))
rob = pd.read_csv(ROOT / "results" / "paper" / "robustness.csv")
rob["setting"] = rob["embedder"].map({"all-MiniLM-L6-v2": "MiniLM", "bge-small-en-v1.5": "BGE small",
                                      "paraphrase-multilingual-MiniLM-L12-v2": "Multilingual MiniLM"}) + rob["trim"].map(lambda t: f", {t} words") + rob["ends"].map(lambda e: ", with ideas" if e else "")
order = ["GPT-2 XL", "GPT-J 6B", "Zephyr 7B", "Vicuna 7B", "Phi-3 mini", "OLMo 2 7B", "Qwen2.5 0.5B", "Qwen2.5 1.5B",
         "Qwen2.5 3B", "Qwen2.5 7B base", "Qwen2.5 7B", "Qwen2.5 14B", "Qwen3 8B"]
cols = ["MiniLM, 40 words, with ideas", "MiniLM, 15 words", "BGE small, 40 words, with ideas", "Multilingual MiniLM, 40 words, with ideas"]
L.append(r"\section{Robustness}\label{app:robust}")
L.append(r"The main setting reads each answer between its two ideas, keeps its first 40 words and uses all-MiniLM-L6-v2. Table~\ref{tab:robust} repeats the share of the map reached with a shorter cut and two other embedding models. The order of the models stays the same in every setting.")
L.append(r"\begin{table}[h]\centering\small\begin{tabular}{lrrrr}\toprule")
L.append("Model & " + " & ".join(r"\shortstack{" + c.replace(", ", r"\\") + "}" for c in cols) + r" \\ \midrule")
for m in order:
    v = rob[rob["model"] == m].set_index("setting")["recall"]
    L.append(m + " & " + " & ".join(f"{v.get(c, float('nan')):.2f}" for c in cols) + r" \\")
L.append(r"\bottomrule\end{tabular}\caption{Share of the human map reached, with no persona, in four settings.}\label{tab:robust}\end{table}")
(ROOT / "paper" / "appendix_conditions.tex").write_text("\n".join(L) + "\n")
print("ok")
