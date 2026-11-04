# Match Compare

Pick a football team and one of its matches. See what changed against the team's other matches, at team and player level, with charts and a written summary whose every number is checked against the data.

**Live demo:** https://samitontheball.streamlit.app

## What it does
- **Compare one match to a baseline you choose:** the matches before it, the matches after it, or all other matches, optionally limited to the same competition.
- **Rank the biggest rises and falls** at team level (xG, shots, xG per shot, shots on target, shot distance, share of shots in the box, and the same conceded) and at player level (per 90 minutes: shots, shots on target, key passes, xG, xA, xGChain, xGBuildup).
- **Charts:** xG created vs conceded, shot maps (baseline vs the match), and change bars for team and players.
- **Findings written by rules, not by a model:** `matchreport/insights.py` holds fixed rules such as "shot volume held but xG per shot collapsed" or "key passes vanished while shots did not". The same rules run for every team, so the output is reproducible and testable.
- **Optional written summary:** a language model rewrites the findings as prose. A validator then checks every number in the prose against the data and flags anything invented. Without a model, the findings and charts stand on their own.

## Data
| Source | Coverage | Provides |
|---|---|---|
| Understat (`matchreport/sources/understat.py`) | Top 5 European leagues, current season | match xG, every shot with location and xG, player xG, xA, xGChain, xGBuildup, key passes |
| Highlightly (`matchreport/sources/highlightly.py`) | Current season, every major competition including the Champions League | per-player passes (attempted and completed), key passes, dribbles, duels, tackles, interceptions, fouls, offsides, match rating, xG and xA; team xG; lineups. Free plan: 100 requests a day. No shot locations |
| API-Football (`matchreport/sources/apifootball.py`) | All major leagues; the free plan serves seasons 2022 to 2024, the current season needs a paid plan | per-player passes, pass accuracy, tackles, interceptions, duels, dribbles (attempted, completed, dribbled past), fouls, match rating, and team xG where the provider has it. No shot locations and no xA or xGChain |
| Manual CSV (`matchreport/sources/manual.py`) | Anything, entered by hand with a source URL per row | team-level goals and xG |

Every source is an adapter that writes the same three tables (`matchreport/schema.py`), so adding a source means writing one adapter. Free data for current matches is match and shot level; passes, tackles, dribbles, and tracking are not available from any free current source.

## Highlightly (full player stats, current season, free)
```bash
export HIGHLIGHTLY_KEY=your_key            # free plan at highlightly.net; never commit or paste the key
python report.py fetch-hl --team Barcelona --season 2026 --lineups
python report.py --team Barcelona --vs "Real Madrid" --source highlightly --no-llm
```
About 2 requests per match (3 with `--lineups`), cached, and a daily-quota refusal stops the fetch cleanly; run the same command again after the quota resets at midnight UTC and it continues. Highlightly's terms allow storing the data and displaying it in your own site and articles, and forbid passing it on as data (exports, feeds, datasets), so fetched files stay out of the repository.

## API-Football (full player stats)
```bash
export APIFOOTBALL_KEY=your_key            # never commit or paste the key anywhere public
python report.py fetch-api --league "ESP-La Liga" --season 2024 --team Barcelona --lineups
python report.py --team Barcelona --vs "Real Madrid" --source apifootball --no-llm
```
- **Resumable.** Every response is cached under `data/cache/`, and the client reads the remaining daily quota from the response headers and stops before it runs out. On the free plan (100 requests a day) a season takes two days; run the same command again and it continues from the cache.
- **Not fetched by site visitors.** On the website this source is read from files fetched ahead of time, so visitors cannot use up the owner's request quota.
- **Seasons.** `--season 2024` means 2024/25. A free key gets a clear plan error for newer seasons.
- **Check the terms** of your plan before publishing data fetched from the API; this repository does not include any fetched data.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q

# Web app (fetches on demand, cached for 6 hours)
streamlit run app.py

# Command line
python report.py fetch --league "ESP-La Liga" --season 2026-2027 --team Barcelona
python report.py --team Barcelona --vs "Real Madrid" --baseline before --no-llm
```
Reports land in `reports/<team>_vs_<opponent>_<date>_<baseline>/` with `report.md`, CSV tables, charts, `findings.txt`, and (with a model) the exact prompt and `validation.json`.

## Written summary: choosing a model
Set these as environment variables locally, or as secrets on Streamlit Cloud. With none set, the app shows findings only.

| Backend | Settings | Cost |
|---|---|---|
| Local Ollama | `LLM_BACKEND=ollama`, `OLLAMA_MODEL=llama3.1:8b` | free, runs on your machine |
| Any OpenAI-compatible API | `LLM_BACKEND=openai`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` | Groq's free tier works (`https://api.groq.com/openai/v1`, model `openai/gpt-oss-20b`; providers retire models, so check their model list if you get a 404) |
| None | `LLM_BACKEND=none` | free |

## Deploy free on Streamlit Community Cloud
1. Push this repo to GitHub (public).
2. At share.streamlit.io, create an app from the repo with `app.py` as the entry point.
3. Optional: in the app's Secrets, add the model settings from the table above.

## How the summary is checked
1. `metrics.compare` builds the tables.
2. `insights.findings` applies the rules and produces findings with numbers.
3. The model receives only the findings and rewrites them.
4. `narrative.validate` extracts every number from the prose and checks it exists in the tables or findings. Unverified numbers are listed with the summary.

## Limitations
- The baseline is a plain average of other matches; it does not adjust for opponent strength.
- Shot locations come from Understat's model and coordinates, not tracking data.
- Player findings only consider players with 60+ minutes in the match and 200+ minutes in the baseline (less when the baseline has 3 or fewer matches), to avoid per-90 noise from short appearances.
- The validator checks numbers and position words, not every sentence.
- Players are matched within one source only. Understat and API-Football are not merged at player level, because their player names differ.
- Understat's terms apply to its data. The app fetches on demand with a short cache and does not redistribute data.

## Data attribution
Understat (understat.com). Manual rows cite their source URL.
