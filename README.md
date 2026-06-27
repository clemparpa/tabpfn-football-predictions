# TabPFN Football Predictions

This repository is a template to participate in Prior Labs' [World Cup Game Outcome Prediction competition](https://ux.priorlabs.ai/worldcup). It has a basic script that outputs predictions with a standard prediction template. Use this template to generate predictions. The `legacy/predict.py` script should only be a source of inspiration, feel free to fork the repo and add your own ideas.

> **Note** : ce dépôt a évolué. Le pipeline maintenu vit dans le package `training/` ; la soumission se génère via `python -m cli.submit`. Les scripts pandas d'origine (`predict.py`, `baseline.py`) sont conservés dans `legacy/` comme références historiques (non maintenues). La section ci-dessous décrit le script `legacy/predict.py` d'origine.

The script predicts international football match outcomes using [TabPFN](https://github.com/PriorLabs/TabPFN) using the [client repository](https://github.com/PriorLabs/tabpfn-client). It achieves ~59% accuracy and ~0.86 log-loss on held-out data. There is a good margin of progression. We look forward to your submission!

The model is trained on engineered features: ELO ratings, recent form, head-to-head record, rest days, and tournament importance. Data comes from [martj42/international_results](https://github.com/martj42/international_results).

## Setup

```bash
git clone https://github.com/eliott-kalfon/tabpfn-football-predictions.git
cd tabpfn-football-predictions
pip install -r requirements.txt
```

## Run

```bash
python legacy/predict.py
```

This will:

1. Download the full international results dataset (~47 000 matches) on first run
2. Build features with a single chronological pass (no leakage)
3. Run a quick backtest on the previous calendar month and print accuracy + log-loss
4. Train on up to 10 000 recent matches and predict all upcoming fixtures
5. Save predictions to `predictions_YYYYMMDD.csv` and print them to the console

To refresh the dataset from source before predicting:

```bash
python legacy/predict.py --refresh
```

## Output

```
Latest game in dataset: 2026-06-14
Data freshness: 0 days 18:32:11

Backtest 2026-05 (87 matches): accuracy 59%, log-loss 0.861

142 fixture predictions -> predictions_20260616.csv

  2026-06-18           Argentina vs Australia             -> home_win   H  72% | D  17% | A  11%
  2026-06-18              France vs Morocco              -> home_win   H  61% | D  23% | A  16%
  ...
```

## Features

| Feature | Description |
|---|---|
| `elo_diff` | ELO gap (home + home advantage - away) |
| `home_elo`, `away_elo` | Current ELO ratings |
| `form5_diff` | Difference in average points per game over last 5 matches |
| `form10_diff` | Same over last 10 matches |
| `home_winrate`, `away_winrate` | Win rate over last 10 matches |
| `home_gf5`, `away_gf5` | Goals scored per game over last 5 matches |
| `home_ga5`, `away_ga5` | Goals conceded per game over last 5 matches |
| `gd10_diff` | Difference in average goal difference over last 10 matches |
| `home_streak`, `away_streak` | Current win streak |
| `home_rest`, `away_rest` | Days since last match (capped at 90) |
| `home_played`, `away_played` | Total matches played in history |
| `h2h_n` | Number of head-to-head meetings |
| `h2h_home_winrate` | Home team win rate in head-to-head |
| `h2h_draw_rate` | Draw rate in head-to-head |
| `h2h_gd` | Average goal difference in head-to-head (from home team's perspective) |
| `neutral` | 1 if played at a neutral venue |
| `importance` | Tournament importance score (60 = World Cup, 20 = friendly) |


### BASE Preds:

Prediction methodology
We used a dataset of historical national games to generate game-outcome predictions with TabPFN-3. On top of the raw data, we engineered features like Elo rating, form and recent goal differential that take short-term information into account.

Feature engineering
The feature set is built in a single chronological pass over the data, so every feature at kickoff time uses only matches that have already been played — no future leakage.

Elo ratings
elo_diff
home_elo
away_elo
The most important features. Each team starts at 1500 and is updated after every result using the standard formula, scaled by two multipliers: margin of victory (a 3-goal win counts more than a 1-goal win) and tournament importance — major-tournament matches move the needle more than friendlies. The raw ratings capture long-run strength; the difference between the two teams, with home advantage folded in as a 65-point bonus for non-neutral venues, is the clearest single signal the model has.
Form
form5_diff
form10_diff
home_form5
away_form5
Average points-per-game over the last 5 and 10 matches. Two windows, because a team can be tactically hot over 5 games while a 10-game window catches a longer drift. The differential between teams compresses both into one number without losing the individual values.
Goal stats
home_gf5
away_gf5
home_ga5
away_ga5
gd10_diff
These go beyond results. A team winning 1-0 every week and one winning 4-1 carry the same points tally but very different attacking profiles. Average goals scored and conceded over the last 5 games, combined with a 10-game goal differential, let the model distinguish them.
Streak and rest
home_streak
away_streak
home_rest
away_rest
These round out the short-term picture. Consecutive wins capture momentum that points-per-game smooths over. Days since the last game (capped at 90) capture fatigue from fixture congestion, which matters in tournament group stages.
Head-to-head history
h2h_n
h2h_home_winrate
h2h_draw_rate
h2h_gd
Captures matchup-specific dynamics that aggregate ratings miss. Some pairings have persistent tactical or psychological patterns that show up reliably over dozens of meetings.