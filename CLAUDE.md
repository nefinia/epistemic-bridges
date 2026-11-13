# Epistemic Bridges

Research code for the Bridges game at mapideas.org/bridges. The game itself lives in nefinia/themap (`bridges/`); this repo holds the pairs, personas, AI generation and analysis.

- Player answers never go into this repo. `tools/export_humans.sh` writes them to `data/`, which git ignores.
- Never type or repeat credentials, keys or secrets for Sofia. She sets them herself.
- Evaluation uses established measures only (`docs/methods.md`). No model judges creativity.
- The personas in `personas/personas.py` are verbatim from False Epistemic Redundancy (Anya Habana). Keep them unchanged.
- Writing for the README and docs: quiet and plain, no em dashes, no "not X but Y" contrasts, no all-caps.
