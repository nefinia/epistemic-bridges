# Epistemic Bridges

How do people connect two distant ideas, and how does AI do it?

Players see two ideas with an arrow between them and write how they would get from one to the other. The same pairs go to AI models, both plain and with different reasoning personas. Comparing the two sets of bridges shows how varied each one is, which stepping stones they pass through, and whether the direction of the arrow changes the path.

The longer aim is to find a harmonic space where collaborating with AI increases epistemic diversity, without shrinking the space of ideas.

Play: [mapideas.org/bridges](https://mapideas.org/bridges)

Built for the Apart Research AI x Epistemics sprint, November 2026.

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

The game's code lives in [nefinia/themap](https://github.com/nefinia/themap/tree/main/bridges).

## The AI side

The same pairs, directions, instructions and languages go to AI models, first with no persona, then with the four reasoning personas from False Epistemic Redundancy:

- **Causal mechanist:** follows physical cause and effect.
- **Analogical thinker:** maps the relation onto a different domain.
- **Teleological analyst:** asks what purpose links the two.
- **Dialectical thinker:** looks for the contradiction and its synthesis.

The question is whether the personas widen the space of bridges, and in which way.

## How diversity is measured

AI helps run the code; it never judges creativity.

- **Rarity:** how few others found a similar bridge.
- **Diversity of each group:** the effective number of distinct bridges among humans and among AI answers.
- **Distance in meaning:** with several embedding models, plus a word-overlap check that uses no model.
- **Stepping stones:** which concepts each group passes through, and whether AI relies on a few central ones.
- **Direction:** whether a → b and b → a lead to different paths.
- **Human judgement:** a sample rated blind, so that rare answers which make no sense do not count as creative.

The measures, settings and references are in [docs/methods.md](docs/methods.md).

## In this repo

- `data/pairs.json`: the 30 pairs in five languages, and the instructions players see (with each version of the welcome text).
- `personas/personas.py`: the four personas, reused verbatim from False Epistemic Redundancy, where Anya Habana wrote them.
- `generate/generate.py`: sends the same prompt the players see to any OpenAI-compatible model, with or without a persona. `generate/serve_model.py` serves an open model on Modal, reused from False Epistemic Redundancy.
- `analysis/diversity.py`: all the diversity measures. The clustering and pairwise distance follow False Epistemic Redundancy.
- `tools/export_humans.sh`: exports player answers from the game's database into `data/`. Player answers stay out of git.

```
pip install -r requirements.txt
python generate/generate.py --model Qwen/Qwen2.5-7B-Instruct --n 10 --langs en es fr
python analysis/diversity.py --with-ends
```

## Status

The game is live and collecting answers. Analysis and write-up in progress.

Sofia Gallego · CosmoVentures · Paris
