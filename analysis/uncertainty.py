"""How much the voice-and-ideas numbers move, from three sources.

1. The measurement itself: voice_content.py run with ten seeds (which half of the players draws
   the map, which model answers are drawn, where content words are cut). Mean and standard
   deviation across seeds; the mean is what the paper reports.
2. The models' own sampling: five models answered twice with no persona, in two separate runs.
   The difference between the two batches, averaged over seeds.
3. The players: voice_content.py run again leaving out one player at a time (--drop-player,
   --plain). Jackknife standard error over the 16 runs.

    python analysis/uncertainty.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_voice_ideas import deltas  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "paper"
COLS = ["cr_words", "cr_pos", "on_map_ideas", "map_reached_ideas"]


def split(res: pd.DataFrame) -> pd.DataFrame:
    res = res.copy()
    res["base"] = res["model"].str.split(" | ", regex=False).str[0]
    res["cond"] = res["model"].str.split(" | ", regex=False).str[1]
    return res


def main() -> None:
    seeds = sorted(RES.glob("voice_content_seed*.csv"))
    runs = [pd.read_csv(f).assign(seed=int(f.stem.split("seed")[1])) for f in seeds]
    allr = pd.concat(runs, ignore_index=True)
    info = allr.drop_duplicates("model").set_index("model")[["kind", "name", "share_with_content"]]
    mean = allr.groupby("model")[COLS].mean().join(info).reset_index()
    sd = allr.groupby("model")[COLS].std().add_suffix("_sd").reset_index()
    mean.merge(sd, on="model").to_csv(RES / "voice_content_with_sd.csv", index=False)
    mean.to_csv(RES / "voice_content.csv", index=False)
    print(f"{len(runs)} seeds")

    # deltas per seed, then mean and spread across seeds
    d = pd.concat([deltas(split(r)).assign(seed=r["seed"].iloc[0]) for r in runs], ignore_index=True)
    per_model = d.groupby(["run", "cond", "model"])[["d_ideas", "d_onmap", "d_voice"]].mean().reset_index()
    per_model.to_csv(RES / "voice_content_deltas.csv", index=False)
    cond_seed = d.groupby(["run", "cond", "seed"])[["d_ideas", "d_voice"]].mean().reset_index()
    cond = cond_seed.groupby(["run", "cond"]).agg(ideas=("d_ideas", "mean"), ideas_sd=("d_ideas", "std"),
                                                  voice=("d_voice", "mean"), voice_sd=("d_voice", "std")).reset_index()
    rng_models = per_model.groupby(["run", "cond"]).agg(lo=("d_ideas", "min"), hi=("d_ideas", "max"),
                                                        positive=("d_ideas", lambda s: int((s > 0).sum())),
                                                        models=("d_ideas", "size")).reset_index()
    cond = cond.merge(rng_models, on=["run", "cond"]).sort_values("ideas")
    cond.to_csv(RES / "conditions_uncertainty.csv", index=False)
    print(cond.round(3).to_string(index=False))

    # sampling of the models: two separate batches with no persona
    s = split(allr)
    a = s[s["cond"].isna()].set_index(["base", "seed"])[COLS]
    b = s[s["cond"] == "creative:baseline"].set_index(["base", "seed"])[COLS]
    both = a.join(b, lsuffix="_main", rsuffix="_second", how="inner")
    gen = pd.DataFrame({c: (both[f"{c}_main"] - both[f"{c}_second"]).abs() for c in COLS}).groupby(level=0).mean()
    gen.to_csv(RES / "sampling_uncertainty.csv")
    print("difference between two batches of answers (mean over seeds):")
    print(gen.round(3).to_string())

    # players: leave one out
    drops = sorted(RES.glob("voice_content_drop_*.csv"))
    if drops:
        j = pd.concat([pd.read_csv(f) for f in drops], ignore_index=True)
        n = len(drops)
        jk = j.groupby("model")[COLS].agg(lambda x: float(np.sqrt((n - 1) / n * ((x - x.mean()) ** 2).sum()))).add_suffix("_jk")
        jk.to_csv(RES / "players_uncertainty.csv")
        print(f"leave one player out ({n} runs), jackknife standard error:")
        print(jk.join(info["name"]).round(3).to_string())

    show = mean.merge(sd, on="model")
    show = show[~show["model"].str.contains("|", regex=False)]
    print(show[["name", "map_reached_ideas", "map_reached_ideas_sd", "on_map_ideas", "on_map_ideas_sd", "cr_pos", "cr_pos_sd"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
