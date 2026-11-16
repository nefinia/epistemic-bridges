# Epistemic Bridges

How do people connect two distant ideas, and how does AI do it?

Players see two ideas with an arrow between them and write how they would get from one to the other. The same pairs go to AI models, plain, with reasoning personas, with creative conditions and at different temperatures. The players' answers draw a map of where people go, and each model is placed on it.

The longer aim is to find a harmonic space where collaborating with AI increases epistemic diversity, without shrinking the space of ideas.

Play: [mapideas.org/bridges](https://mapideas.org/bridges)

Built for the Apart Research AI x Epistemics sprint, November 2026.

## Why

Bridges comes from the [Map of Ideas](https://mapideas.org), a map of how concepts and debates relate to one another. Tools built on what is already written, like Wikipedia or AI models, tend to converge on what is already known. The map tries to keep room for opposing views and contrasts, and to make them visible, so that new spaces can open between ideas.

Bridges is a first test of that. It starts from random pairs of simple concepts, so anyone can play without knowing a particular topic, and looks at where new connections appear and whether AI helps people reach them.

## Where it comes from

This follows [False Epistemic Redundancy](https://apartresearch.com/project/false-epistemic-redundancy-do-ai-ensembles-share-a-blind-spot-b50u) ([code](https://github.com/nefinia/epistemic-fingerprints)), which found that reasoning personas made AI answers look more diverse while they still missed the same cause. Bridges asks the same question about creating ideas instead of checking them.

## The game

- Six core pairs that everyone answers, from concrete to abstract (a bee → a mirror, salt → a clock, a river → a secret, a spider → a melody, forgetting → a festival, distance → a habit), then 24 more pairs for anyone who wants to continue.
- Each pair is shown in one direction to each player, balanced so that both directions get the same number of answers.
- Optional stepping stones: the ideas passed through on the way.
- A short warm-up (ten words as different as possible) as a baseline.
- An AI bridge only when the player asks for it.
- English, French, Spanish, Italian and German; typed or spoken.
- No names, no cookies, no IP addresses. Privacy: [mapideas.org/bridges/privacy](https://mapideas.org/bridges/privacy)

The game's code lives in [nefinia/mapideas](https://github.com/nefinia/mapideas/tree/main/bridges).

## The AI side

The same pairs, directions, instructions and languages go to AI models, first with no persona, then with the four reasoning personas from False Epistemic Redundancy:

- **Causal mechanist:** follows physical cause and effect.
- **Analogical thinker:** maps the relation onto a different domain.
- **Teleological analyst:** asks what purpose links the two.
- **Dialectical thinker:** looks for the contradiction and its synthesis.

The question is whether the personas widen the space of bridges, and in which way.

## What we found

The paper is [paper/main.pdf](paper/main.pdf), and a short version is at [sofiagallego.com/epistemic-bridges](https://sofiagallego.com/epistemic-bridges). In short, with 27 players, 145 bridges and 15 open models from 2019 to 2025:

- Measured as written, chat models seem to reach almost none of the ground people cover. Most of that gap is the voice: chat answers share one way of speaking, and their sentence patterns repeat more than people's.
- With the voice removed from every answer in the same way, the larger and newer chat models still reach less of the human map of ideas (recall 47% to 59%, against 81% for people) while landing on it most often. No model comes near people on both counts at once.
- Creative conditions change the voice more than the ideas. Only one widens the ideas on every model: answering as an ordinary person drawn at random.
- The direction of the arrow matters for chat models and hardly for people: chat answers talk more about the word they start from.

## How it is measured

AI helps run the code; no model judges any answer.

- **The human map:** the region of meaning that the players' answers cover, for each pair (all-MiniLM-L6-v2 embeddings, k-nearest-neighbour spheres).
- **Precision and recall:** the share of a group's answers that land on the human map, and the share of the human map they reach (Kynkäänniemi et al. 2019). People are measured half against half.
- **Voice and ideas:** the voice is how much a set of answers repeats its sentence patterns (compression of parts of speech, Shaib et al. 2024). For the ideas, the scaffolding and the two words of the pair are removed from every answer, human or AI, and answers are compared on their content words at a player's length.
- **Hub rate:** the share of answers that take the most common human bridge, such as the hourglass for salt and clock.
- **Direction:** whether answers going a → b differ from answers going b → a, and which word they talk about more.
- **Warm-up:** the Divergent Association Task score, against published scores from before ChatGPT.
- **Uncertainty:** 11 seeds, two batches of answers, and leaving out one player at a time.

The paper's Methods section and appendix describe the measures in full. [docs/methods.md](docs/methods.md) and [docs/context-and-next.md](docs/context-and-next.md) are the plan written at the start of the sprint; some of it, such as rarity scores and blind human ratings, was left for later.

## In this repo

- `data/pairs.json`: the 30 pairs in five languages, and the instructions players see.
- `personas/personas.py`: the four reasoning personas, reused verbatim from False Epistemic Redundancy, where Anya Habana wrote them.
- `generate/`: `ladder_modal.py` runs the open models on Modal (vLLM), with every persona, creative condition and temperature, and the warm-up; `api_models.py` does the same for models behind an API; `generate.py` and `serve_model.py` are the simpler versions reused from False Epistemic Redundancy. All AI answers are in `data/ai/`, one file per model.
- `analysis/`:
  - `paper_results.py`: precision, recall and hub rate as written, and the robustness checks.
  - `voice_content.py` and `uncertainty.py`: voice and ideas measured apart, with seeds and the leave-one-player-out check.
  - `levels_directions.py`, `diversity_ladder.py`: pairs, direction, and diversity as the voice is removed step by step.
  - `style_and_warmup.py`, `dat.py`, `all_languages.py`: the cleaning of the scaffolding, the warm-up, and the check with every language.
  - `fig_money.py`, `fig_voice_ideas.py`: the figures.
- `paper/`: the LaTeX source, figures and PDF.
- `tools/export_humans.sh`: exports player answers from the game's database into `data/`. Player answers stay out of git.

```
pip install -r requirements.txt
modal run --detach generate/ladder_modal.py
python analysis/paper_results.py --humans data/human_bridges.json --tools /path/to/tools
python analysis/voice_content.py --humans data/human_bridges.json --tools /path/to/tools
python analysis/levels_directions.py --humans data/human_bridges.json --tools /path/to/tools
python analysis/fig_money.py
```

`/path/to/tools` holds the embedding models and GloVe vectors; the `tools` task in `.github/workflows/run.yml` builds it.

## What existed before the sprint

- The Bridges game itself, built before the sprint started on 13 November 2026. Its code is in [nefinia/mapideas](https://github.com/nefinia/mapideas/tree/main/bridges). Players have played it since 25 October; 6 of the 145 bridges in the paper, from one player, came before the sprint.
- From False Epistemic Redundancy: the four personas (written by Anya Habana), the Modal server, and the clustering and distance settings used in the analysis.

Everything else in this repo was written during the sprint.

## Status

The game is live and still collecting answers. The paper uses the answers given up to 15 November 2026.

Sofia Gallego · CosmoVentures · Paris
