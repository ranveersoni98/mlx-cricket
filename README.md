# MLX-Cricket

Predict cricket winners with Apple MLX. Trained on ~27k historical matches, works on live fixtures through CricAPI.

## Setup

```bash
uv sync
source .venv/bin/activate
cp .env.template .env   # drop your CricAPI key in there (free at cricapi.com)
```

## Usage

```bash
mlx-cric upcoming --limit 10     # predict live fixtures
mlx-cric predict --team1 India --team2 Australia --venue-country India --format ODI
mlx-cric train --epochs 30 --rebuild
mlx-cric evaluate
```

Tests get three-way output since draws are real:

```json
{"team1": "England", "team2": "Australia",
 "p_team1": 0.435, "p_draw": 0.23, "p_team2": 0.335,
 "winner": "England", "confidence": 0.435}
```

## How it works

Two stages. A winner model (MLX MLP + gradient-boosting sidecar, combined by a stacking classifier) predicts who wins *given* a decisive game. For Tests, a small logistic regression predicts P(draw) from tier, venue draw-rate, mismatch size, and year. Everything is pre-match only — no score leakage.

Features are all chronological: Elo (with proper IPL home cities), last-5 form, head-to-head, venue and toss numbers, rest days, season, player-pool strength (top-11 active batsmen/bowlers from 700k+ ball-by-ball innings), plus team/venue/format embeddings. The MLP also trains on victory margins as an auxiliary target. Splits are walk-forward (train ≤2021, val 2022–23, test 2024–26), so the numbers below are honest. The MLP is trained over 5 seeds and stacked with the booster.

## Accuracy (test set, 2024–26)

| Slice | Acc |
|---|---|
| Overall (MLP ×5 + HGB, stacked + blended) | 64.4% |
| International | 71.4% |
| Women's | 66.9% |
| T20 | 65.7% |
| ODI | 60.5% |
| Test (decisive) | 60.2% |
| Test (win/loss/draw) | 43.9% |
| Domestic/club | 57.2% |

Club cricket and draws are the weak spots thin data and rain, respectively. The honest numbers are in `artifacts/metrics.json` after training.

## Layout

```
src/mlx_cric/
  config.py       settings + split years
  data/           HF downloads, team/venue normalization
  features/       Elo + rolling form (chrono.py), vocabularies (store.py)
  models/         the MLX network
  training/       loop, baselines, calibration
  inference/      CricAPI client, predictor
```

## Data

Model trains on public datasets — they keep their own licenses:

- `Sarthak213/doosra-cricket` (ODC-BY) — 23k matches to Sep 2026, the bulk of training
- `bhuvaneshprasad/*` ODI/T20I/IPL sets — 1971–2014 internationals + IPL
- Cricsheet (CC-BY-4.0) via the doosra redistribution — credit cricsheet.org if you reuse the data
- CricAPI — live scores only, needs your own key

## License

Code is MIT — see [LICENSE](LICENSE).

<p align="center">
  Built with <3 and a lot of bad decisions! by <a href="https://ranveersoni.me">Ranveer Soni</a>
</p>
