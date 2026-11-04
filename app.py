"""Streamlit app. Run locally: streamlit run app.py
Deployed free on Streamlit Community Cloud; secrets (optional) set the LLM backend.

Data sources
- Understat: fetched on demand for the current season (xG, shots, key passes, xGChain/xGBuildup).
- API-Football: fetched ahead of time with the command line (python report.py fetch-api ...), never by visitors,
  so a public site cannot burn the owner's daily request quota. Adds ratings, passes, tackles, dribbles, duels."""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# Secrets (on Streamlit Cloud) or environment variables select the LLM. Without any, findings only.
SECRETS_ERROR = None
try:
    for key in ("LLM_BACKEND", "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL", "OLLAMA_HOST", "OLLAMA_MODEL", "HIGHLIGHTLY_KEY"):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception as e:  # no secrets file is fine (environment variables are used); a broken one is reported
    if "No secrets found" not in str(e):
        SECRETS_ERROR = str(e)
os.environ.setdefault("LLM_BACKEND", "none")

from matchreport import charts, insights, metrics, narrative, report, store  # noqa: E402
from matchreport.sources import highlightly, understat  # noqa: E402

DATA = Path(os.environ.get("MR_DATA_DIR", "data"))
st.set_page_config(page_title="Match Compare", layout="wide")
st.title("Match Compare")
st.caption("Pick a team and a match. See what changed against the team's other matches, at team and player level. "
           "Every number comes from the data; the written summary is checked against it.")

local_sources = store.available_sources(DATA) if (DATA / "raw").exists() else []
SOURCE_LABELS = {"highlightly": "Highlightly (this season: ratings, passes, dribbles, duels, xG)",
                 "understat": "Understat (this season: xG, shots, shot maps)",
                 "apifootball": "API-Football (ratings, passes, tackles; seasons 2022 to 2024)"}

with st.sidebar:
    st.header("Data")
    # Understat scrapes a website on demand and can hang or be blocked from a cloud server, so it is offered only
    # when enabled (MR_ENABLE_UNDERSTAT=1) or when Understat data is already on disk (local use).
    understat_on = os.environ.get("MR_ENABLE_UNDERSTAT") == "1" or "understat" in local_sources
    options = ["highlightly"] + (["understat"] if understat_on else []) + (["apifootball"] if "apifootball" in local_sources else [])
    source = st.radio("Data source", options, format_func=SOURCE_LABELS.get)

    if source == "understat":
        league = st.selectbox("League", understat.LEAGUES, index=1)
        season = st.text_input("Season", "2026-2027")
        try:
            teams = st.cache_data(ttl=6 * 3600, show_spinner="Loading teams...")(understat.list_teams)(league, season)
        except Exception as e:
            teams = []
            st.error(f"Could not load the league schedule: {e}")
        team = st.selectbox("Team", teams) if teams else st.text_input("Team", "Barcelona")

        @st.cache_data(ttl=6 * 3600, show_spinner="Fetching matches from Understat (cached for 6 hours)...")
        def fetch(league, season, team):
            out = DATA / "raw" / f"understat_{league}_{season}_{team}".replace(" ", "_")
            return understat.fetch(league, season, team, out), str(out)
    elif source == "highlightly":
        hl_season = st.number_input("Season (start year)", 2020, 2030, 2026)
        hl_team = st.text_input("Team (as Highlightly spells it)", "Barcelona")
        owner_key = os.environ.get("HIGHLIGHTLY_KEY", "")
        user_key = "" if owner_key else st.text_input(
            "Your Highlightly API key", type="password",
            help="Free at highlightly.net (100 requests a day). Used only for this session; it is not saved.")
        hl_key = owner_key or user_key
        if not hl_key:
            st.caption("Enter a free Highlightly key to fetch a team. Teams other visitors fetched are listed below.")
        if hl_key and st.button("Fetch or update this team (about 2 requests per match, 100 a day)"):
            out = DATA / "raw" / f"highlightly_{hl_season}_{''.join(ch for ch in hl_team.lower() if ch.isalnum())}"
            with st.spinner("Fetching from Highlightly..."):
                try:
                    info = highlightly.fetch(hl_team, int(hl_season), out,
                                             highlightly.Client(key=hl_key, cache_dir=DATA / "cache" / "highlightly"),
                                             progress=lambda *_: None)
                    st.success(f"{info['matches_saved']} matches saved" + ("" if info["complete"] else f". {info['stopped']}"))
                except highlightly.ApiError as e:
                    st.error(str(e))
        try:
            m_hl, _, _ = store.load_all(DATA, sources=["highlightly"])
            counts = pd.concat([m_hl["home_team"], m_hl["away_team"]]).value_counts()
            teams = sorted(counts[counts >= 0.8 * counts.max()].index)
        except FileNotFoundError:
            teams = []
        team = st.selectbox("Team", teams) if teams else None
        if not teams:
            st.info("No Highlightly data yet." + ("" if hl_key else " Enter your free key above, then fetch a team."))
    else:
        m_api, _, _ = store.load_all(DATA, sources=["apifootball"])
        counts = pd.concat([m_api["home_team"], m_api["away_team"]]).value_counts()
        teams = sorted(counts[counts >= 0.8 * counts.max()].index)   # the teams that were fetched, not their opponents
        team = st.selectbox("Team", teams)
        st.caption("Fetched ahead of time with the command line, so visitors cannot use up the daily request quota. "
                   "The free plan covers seasons 2022 to 2024.")

    st.header("Comparison")
    baseline = st.radio("Compare the match with", ["before", "all", "after"],
                        format_func={"before": "the matches before it", "all": "all other matches",
                                     "after": "the matches after it"}.get)
    same_comp = st.checkbox("Same competition only", value=True)
    want_llm = st.checkbox("Written summary", value=os.environ["LLM_BACKEND"] != "none",
                           disabled=os.environ["LLM_BACKEND"] == "none",
                           help="Needs an LLM backend. Without one, the findings are shown on their own.")
    if SECRETS_ERROR:
        st.error(f"Could not read .streamlit/secrets.toml: {SECRETS_ERROR}. Put each setting on its own line.")
    elif os.environ["LLM_BACKEND"] == "none":
        st.caption("Summary off: no model configured. Add LLM_BACKEND and OPENAI_API_KEY as secrets "
                   "(Streamlit Cloud: the app's Settings, then Secrets; on your computer: .streamlit/secrets.toml), "
                   "then restart the app.")
    else:
        st.caption(f"Summary model: {narrative.model_name()}" + ("" if os.environ.get("OPENAI_API_KEY") or
                   os.environ["LLM_BACKEND"] != "openai" else ". OPENAI_API_KEY is missing."))

if not team:
    st.stop()
if source == "understat":
    try:
        fetch(league, season, team)
    except Exception as e:
        st.error(f"Could not fetch data for {team}: {e}")
        st.stop()

try:
    matches, shots, pm = store.load_all(DATA, sources=[source])
except FileNotFoundError as e:
    st.error(f"No {source} data yet. {e}")
    st.stop()
tm = metrics.team_matches(matches, team)
if tm.empty:
    st.warning("No matches found for this team."); st.stop()
labels = [f"{r.date}  vs {r.opponent} ({r.venue})  {r.result} {int(r.goals_for)}-{int(r.goals_against)}" for r in tm.itertuples()]
pick = st.selectbox("Match", list(range(len(labels))), format_func=lambda i: labels[i], index=len(labels) - 1)
row = tm.iloc[pick]

try:
    result = metrics.compare(matches, shots, pm, team, row["opponent"], row["date"], same_comp, baseline)
except ValueError as e:
    st.warning(str(e)); st.stop()
meta = result["meta"]

st.subheader(f"{team} vs {meta['opponent']}, {meta['date']}: {meta['result']} {meta['score']}")
st.write(f"Compared with {meta['baseline_label']} (season {meta['season']}/{str(meta['season'] + 1)[2:]}).")

if want_llm and os.environ["LLM_BACKEND"] != "none":
    key = (source, team, meta["opponent"], meta["date"], baseline, same_comp, narrative.model_name())
    cache = st.session_state.setdefault("summaries", {})
    if key not in cache:
        with st.spinner(f"Writing the summary with {narrative.model_name()}..."):
            try:
                cache[key] = narrative.write(result)
            except RuntimeError as e:
                st.warning(str(e))
    if key in cache:
        nar = cache[key]
        st.markdown("### Summary")
        st.write(nar["text"])
        chk = nar["check"]
        if chk["numbers_found"] == 0:
            st.info("This summary contains no numbers, so there was nothing to check.")
        elif chk["passed"]:
            st.success(f"Checked: all {chk['numbers_found']} numbers in this summary exist in the data below.")
        else:
            problems = []
            if chk["unverified"]:
                problems.append(f"numbers not in the data: {chk['unverified']}")
            if chk.get("unsupported_terms"):
                problems.append(f"claims about {chk['unsupported_terms']} that the findings do not support")
            st.error("Check failed: " + "; ".join(problems) + ". Treat those sentences with caution.")

found = insights.findings(result)
st.markdown("### Findings")
for s in found:
    st.markdown(f"- {s}")

figs = Path("reports") / "_app"
figs.mkdir(parents=True, exist_ok=True)
c1, c2 = st.columns(2)
if meta["has_xg"]:
    c1.image(str(charts.xg_bars(result, figs / "xg.png")))
else:
    c1.info("This data source has no team xG, so the xG chart is skipped.")
if not result["team_changes"].empty:
    c2.image(str(charts.team_change_bars(result, figs / "team.png")))

lineup_metrics = charts.available_lineup_metrics(result)
if result["lineup"].empty or not lineup_metrics:
    st.info("No starting-XI chart for this match: this source has no lineup or player data for it.")
else:
    metric = st.selectbox("Starting XI: metric", lineup_metrics, format_func=charts.LINEUP_METRICS.get)
    st.image(str(charts.lineup_chart(result, figs / f"lineup_{metric}.png", metric)))
    for line in charts.lineup_summary(result, metric):
        st.markdown(f"- {line}")
if meta["has_shots"]:
    with st.expander("Shot map (size = xG, star = goal)"):
        st.image(str(charts.shot_maps(result, figs / "shots.png")))
if not result["player_changes"].empty:
    st.image(str(charts.player_change_bars(result, figs / "players.png")))
elif meta["has_players"]:
    st.info(f"No player-level comparison for this match: no outfield player played at least 60 minutes in it and "
            f"{meta['min_base_minutes']} minutes across the {meta['n_base']} baseline matches. "
            f"Try a later match, or a baseline of 'all other matches'.")

with st.expander("Biggest team-level changes (table)"):
    st.dataframe(result["team_changes"].round(2))
with st.expander("Biggest player changes (table)"):
    st.dataframe(result["player_changes"].round(2))

if st.button("Save full report (markdown, CSVs, charts)"):
    out = report.build_from_result(result, Path("reports"), llm=want_llm)
    st.success(f"Saved to {out}")
    st.download_button("Download report.md", (out / "report.md").read_text(), file_name="report.md")

st.caption("Data: " + ", ".join(meta["sources"]) + ". Rules for the findings are in matchreport/insights.py; "
           "they are the same for every team.")
