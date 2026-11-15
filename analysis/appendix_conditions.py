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
L.append(r"Base models received the same text followed by ``Answer:'', to be continued.")
L.append(r"\subsection*{Generation settings}")
L.append(r"vLLM 0.21 on one NVIDIA L40S GPU per model (Modal), temperature 1.0 unless stated, top-$p$ 1.0, at most 220 new tokens (320 for step by step, whose answer was read from a final line starting with ``Bridge:''), 10 answers per pair, direction, language and condition. Thinking mode was switched off for Qwen3. Chat models received the persona as a system message; where a model's chat template had no system role, the persona was placed before the question. Vicuna 7B had no chat template and received a plain ``USER: \dots\ ASSISTANT:'' format.")
L.append(r"\subsection*{Reasoning personas (from false epistemic redundancy \citep{gallego2026fer})}")
L.append(r"Four personas, each a paragraph given as a system message and reused verbatim; the full texts are in \texttt{personas/personas.py} in the repository. "
         r"\emph{Causal mechanist}: follows physical cause and effect. \emph{Analogical thinker}: maps the relation onto a different domain. "
         r"\emph{Teleological analyst}: asks what purpose links the two. \emph{Dialectical thinker}: looks for the contradiction and its synthesis.")
L.append(r"\subsection*{Creative conditions}")
L.append(r"\begin{description}[leftmargin=0pt, itemsep=0.4em]")
for k, v in creative.FIXED.items():
    L.append(rf"\item[\textbf{{{tex(k.replace('_', ' '))}}}.] \small {tex(v)}")
L.append(r"\item[\textbf{ordinary life}.] \small ``You are [one sketch, drawn at random for each answer]. Answer as yourself, in your own words.''")
L.append(r"\item[\textbf{random constraint}.] \small One sentence drawn at random for each answer from the list below.")
L.append(r"\end{description}")
L.append(r"\subsection*{An ordinary life with a way of reasoning}")
L.append(r"The four reasoning personas, adapted to connecting ideas and each kept to its own stance, were given to an ordinary life drawn at random for each answer: ``You are [life]. [way of reasoning] Answer as yourself, in your own words.''")
L.append(r"\begin{description}[leftmargin=0pt, itemsep=0.4em]")
for k, v in creative.WAYS.items():
    L.append(rf"\item[\textbf{{{tex(k.split('_')[0])}}}.] \small {tex(v)}")
L.append(r"\end{description}")
L.append(r"\noindent\textbf{The 30 ordinary lives.} \small " + "; ".join(tex(x) for x in creative.LIVES) + ".")
L.append("")
L.append(r"\noindent\textbf{The 12 constraints.} \small " + " ".join(tex(x) for x in creative.CONSTRAINTS))
rob = pd.read_csv(ROOT / "results" / "paper" / "robustness.csv")
rob["setting"] = rob["embedder"].map({"all-MiniLM-L6-v2": "MiniLM", "bge-small-en-v1.5": "BGE small",
                                      "paraphrase-multilingual-MiniLM-L12-v2": "Multilingual MiniLM"}) + rob["trim"].map(lambda t: ", full answer" if pd.isna(t) else f", {int(t)} words") + rob["ends"].map(lambda e: ", with ideas" if e else "")
order = ["GPT-2 XL", "GPT-J 6B", "Zephyr 7B", "Vicuna 7B", "Phi-3 mini", "OLMo 2 7B", "Qwen2.5 0.5B", "Qwen2.5 1.5B",
         "Qwen2.5 3B", "Qwen2.5 7B base", "Qwen2.5 7B", "Qwen2.5 14B", "Qwen3 8B", "gpt-oss 20B", "Qwen3 32B"]
cols = ["MiniLM, full answer, with ideas", "BGE small, full answer, with ideas", "Multilingual MiniLM, full answer, with ideas"]
full = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if False and full and full.exists():
    names = {"openai-community/gpt2-xl": "GPT-2 XL", "EleutherAI/gpt-j-6b": "GPT-J 6B", "lmsys/vicuna-7b-v1.5": "Vicuna 7B",
             "HuggingFaceH4/zephyr-7b-beta": "Zephyr 7B", "microsoft/Phi-3-mini-4k-instruct": "Phi-3 mini",
             "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5 0.5B", "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5 1.5B",
             "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5 3B", "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5 7B",
             "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5 14B", "Qwen/Qwen2.5-7B": "Qwen2.5 7B base",
             "allenai/OLMo-2-1124-7B-Instruct": "OLMo 2 7B", "Qwen/Qwen3-8B": "Qwen3 8B", "Qwen/Qwen3-32B": "Qwen3 32B", "openai/gpt-oss-20b": "gpt-oss 20B"}
    f = pd.read_csv(full)
    f = f.groupby("model")["recall"].mean().reset_index()
    f["model"] = f["model"].map(names); f["setting"] = "MiniLM, full answer, with ideas"
    rob = pd.concat([rob, f], ignore_index=True)
R0 = len(L)
L.append(r"\section{Robustness}\label{app:robust}")
L.append(r"The answers as written are read in full, between the two words of their pair, with all-MiniLM-L6-v2. Table~\ref{tab:robust} repeats recall, the share of the human map reached, with two other embedding models. In every setting the base models stay far above the chat models, and the newest and largest chat models stay lowest.")
L.append(r"\begin{table}[h]\centering\small\begin{tabular}{lrrr}\toprule")
L.append("Model & " + " & ".join(r"\shortstack{" + c.replace(", ", r"\\") + "}" for c in cols) + r" \\ \midrule")
for m in order:
    v = rob[rob["model"] == m].set_index("setting")["recall"]
    L.append(m + " & " + " & ".join(f"{v.get(c, float('nan')):.2f}" for c in cols) + r" \\")
L.append(r"\bottomrule\end{tabular}\caption{Recall, the share of the human map reached, with no persona, with three embedding models, to check that the result does not depend on the embedding. Base models stay far above chat models with all three, and the newest and largest chat models stay lowest.}\label{tab:robust}\end{table}")
(ROOT / "paper" / "appendix_conditions.tex").write_text("\n".join(L[:R0]) + "\n")
(ROOT / "paper" / "appendix_robust.tex").write_text("\n".join(L[R0:]) + "\n")
print("ok")
