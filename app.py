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
        owner_key = os.environ.get("HIGHLIGHTLY_KEY", "")
        if not owner_key:
            st.markdown("**Get your free API key** by opening a free account here: "
                        "[highlightly.net](https://highlightly.net). Then copy the API key from your dashboard and "
                        "paste it below. It is free and takes about a minute.")
        hl_season = st.number_input("Season (start year)", 2020, 2030, 2026,
                                    help="2026 means the 2026/27 season. Fetch each season you need separately: "
                                         "every season you fetch stays loaded, so you can compare across them.")
        hl_team = st.text_input("Team (as Highlightly spells it)", "Barcelona",
                                help="Examples: Barcelona, Real Madrid, Manchester United. If nothing is found, try the "
                                     "full club name, for example Paris Saint Germain or Celta de Vigo.")
        user_key = "" if owner_key else st.text_input(
            "Your Highlightly API key", type="password",
            help="Free at highlightly.net, about 100 requests a day. Used only to fetch your data during this visit; "
                 "this app does not store it.")
        hl_key = owner_key or user_key
        if not hl_key:
            st.caption("Paste your key above, then click Fetch. Fetching a team takes a minute or two.")
        if hl_key and st.button("Fetch or update this team (about 2 requests per match, 100 a day)"):
            out = DATA / "raw" / f"highlightly_{hl_season}_{''.join(ch for ch in hl_team.lower() if ch.isalnum())}"
            with st.spinner("Fetching from Highlightly (a minute or two)..."):
                try:
                    info = highlightly.fetch(hl_team, int(hl_season), out,
                                             highlightly.Client(key=hl_key, cache_dir=DATA / "cache" / "highlightly"),
                                             progress=lambda *_: None)
                    if info["matches_saved"] == 0 and info["complete"]:
                        st.warning(f"No finished matches found for '{hl_team}' in season {int(hl_season)}. "
                                   f"Check the spelling, or try the full club name.")
                    else:
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
    mode = st.radio("What to compare", ["match", "periods"],
                    format_func={"match": "One match against other matches",
                                 "periods": "A period against another period"}.get,
                    help="Periods: pick two sets of matches (any seasons, any dates, any results) and compare them.")
    baseline, same_comp = "before", True
    if mode == "match":
      baseline = st.radio("Compare the match with", ["before", "all", "after", "range"],
                          format_func={"before": "the matches before it", "all": "all other matches",
                                       "after": "the matches after it", "range": "a period I choose (any season)"}.get,
                          help="The first three stay within the match's season. 'A period I choose' can reach back into "
                               "earlier seasons, for example a manager's spell last season; fetch those seasons first.")
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
    st.caption("If a menu or chart shows 'Failed to fetch dynamically imported module', refresh the page. "
               "Data you already fetched is kept.")

if not team:
    if source == "highlightly":
        st.markdown("### Get started in three steps")
        st.markdown(
            "1. **Get a free API key.** [Create a free Highlightly account](https://highlightly.net) and copy your API key "
            "from its dashboard. It is free, and this app does not store it.\n"
            "2. **Fill in the sidebar.** Paste the key, enter a team as Highlightly spells it (for example `Barcelona` "
            "or `Real Madrid`) and the season's start year (`2026` for 2026/27).\n"
            "3. **Click Fetch,** wait a minute or two, then pick any match to compare with the team's other matches.")
        st.caption("The free plan allows about 100 requests a day, and each match uses about 2. "
                   "The first visit to this free app can take a minute to wake up.")
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
def match_label(r):
    return f"{r.date}  vs {r.opponent} ({r.venue})  {r.result} {int(r.goals_for)}-{int(r.goals_against)}"


if mode == "periods":
    comps = sorted(tm["competition"].unique())
    chosen = st.multiselect("Competitions", comps, default=[tm["competition"].mode().iat[0]],
                            help="Only matches in these competitions can be picked.")
    pool = tm[tm["competition"].isin(chosen)] if chosen else tm
    seasons = sorted(pool["season"].unique())
    st.caption("Seasons loaded for this team: " + ", ".join(f"{s}/{str(s + 1)[2:]}" for s in seasons)
               + ". To reach further back, fetch that season in the sidebar.")
    if len(seasons) >= 2:     # default: this season against last season
        a_def = pool[pool["season"] == seasons[-1]]; b_def = pool[pool["season"] == seasons[-2]]
        a_name_def = f"{seasons[-1]}/{str(seasons[-1] + 1)[2:]}"; b_name_def = f"{seasons[-2]}/{str(seasons[-2] + 1)[2:]}"
    else:                     # one season loaded: second half against first half
        half = len(pool) // 2
        a_def, b_def = pool.iloc[half:], pool.iloc[:half]
        a_name_def, b_name_def = "Later matches", "Earlier matches"
    first, last = pool["date"].min(), pool["date"].max()

    def period_picker(col, key, title, name_def, default):
        col.markdown(f"#### {title}")
        name = col.text_input("Name", value=name_def, key=f"{key}_name",
                              help="Used in the findings and charts, for example 'Carrick's spell'.")
        d1 = col.date_input("From", value=default["date"].min() if not default.empty else first,
                            min_value=first, max_value=last, key=f"{key}_from")
        d2 = col.date_input("To", value=default["date"].max() if not default.empty else last,
                            min_value=first, max_value=last, key=f"{key}_to")
        lo, hi = sorted([d1, d2])
        which = col.selectbox("Results", ["all", "W", "D", "L"], key=f"{key}_res",
                              format_func={"all": "All results", "W": "Wins only", "D": "Draws only", "L": "Losses only"}.get)
        inside = pool[(pool["date"] >= lo) & (pool["date"] <= hi)]
        if which != "all":
            inside = inside[inside["result"] == which]
        lab = {r.game_id: match_label(r) for r in inside.itertuples()}
        picked = col.multiselect("Matches", list(lab), default=list(lab), format_func=lab.get,
                                 key=f"{key}_games_{lo}_{hi}_{which}_{'-'.join(chosen)}",
                                 help="Every match in the dates is ticked. Remove any you want to leave out.")
        return name.strip() or title, picked

    ca, cb = st.columns(2)
    a_name, a_ids = period_picker(ca, "pa", "Period A (the one you're judging)", a_name_def, a_def)
    b_name, b_ids = period_picker(cb, "pb", "Period B (the yardstick)", b_name_def, b_def)
    if a_name == b_name:
        b_name = b_name + " (B)"
    try:
        result = metrics.compare_periods(matches, shots, pm, team, a_ids, b_ids, a_name, b_name)
    except ValueError as e:
        st.warning(str(e)); st.stop()
    meta = result["meta"]
    if set(a_ids) & set(b_ids):
        st.caption("Matches picked in both periods count only in Period A.")
    st.subheader(f"{team}: {a_name} vs {b_name}")
    st.write(f"{a_name}: {meta['n_focus']} matches, {meta['focus_record']} (won-drawn-lost). "
             f"{b_name}: {meta['n_base']} matches, {meta['base_record']}. Team figures are per match; player figures per 90.")
    summary_key = (source, team, tuple(meta["focus_ids"]), tuple(meta["base_ids"]), a_name, b_name, narrative.model_name())
else:
  labels = [match_label(r) for r in tm.itertuples()]
  pick = st.selectbox("Match", list(range(len(labels))), format_func=lambda i: labels[i], index=len(labels) - 1)
  row = tm.iloc[pick]

  period = None
  if baseline == "range":
      first, last = tm["date"].min(), tm["date"].max()
      earlier = tm[tm["date"] < row["date"]]["date"]
      c1, c2 = st.columns(2)
      p_start = c1.date_input("Period from", value=earlier.min() if not earlier.empty else first,
                              min_value=first, max_value=last, help="First day of the period to compare with.")
      p_end = c2.date_input("Period to", value=earlier.max() if not earlier.empty else last,
                            min_value=first, max_value=last, help="Last day of the period. The match itself is never included.")
      period = (p_start, p_end)
      seasons = sorted({metrics.season_of(d) for d in tm["date"]})
      st.caption("Seasons loaded for this team: " + ", ".join(f"{s}/{str(s + 1)[2:]}" for s in seasons)
                 + ". To reach further back, fetch that season in the sidebar.")
  try:
      result = metrics.compare(matches, shots, pm, team, row["opponent"], row["date"], same_comp, baseline, period=period)
  except ValueError as e:
      st.warning(str(e)); st.stop()
  meta = result["meta"]

  st.subheader(f"{team} vs {meta['opponent']}, {meta['date']}: {meta['result']} {meta['score']}")
  st.write(f"Compared with {meta['baseline_label']}." if meta.get("period") else
           f"Compared with {meta['baseline_label']} (season {meta['season']}/{str(meta['season'] + 1)[2:]}).")
  summary_key = (source, team, meta["opponent"], meta["date"], baseline, same_comp, str(period), narrative.model_name())

if want_llm and os.environ["LLM_BACKEND"] != "none":
    key = summary_key
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
if meta.get("mode") == "periods":     # no single starting XI: show every player for the chosen stat instead
    if lineup_metrics:
        st.subheader("Players: one stat, both periods")
        metric = st.selectbox("Stat", lineup_metrics, format_func=charts.LINEUP_METRICS.get,
                              help="Every player with enough minutes in both periods, grey = the second period, coloured = the first.")
        st.image(str(charts.period_player_chart(result, figs / f"period_{metric}.png", metric)))
        for line in charts.period_metric_summary(result, metric):
            st.markdown(f"- {line}")
        rows = charts.period_metric_rows(result, metric)
        if not rows.empty:
            unit = "" if metric == "rating" else " (per 90)"
            st.caption(f"Every player with {meta['min_match_minutes']}+ minutes in both periods{unit}, biggest drop first.")
            st.dataframe(rows, hide_index=True, use_container_width=True)
elif result["lineup"].empty or not lineup_metrics:
    st.info("No starting-XI chart for this match: this source has no lineup or player data for it.")
else:
    metric = st.selectbox("Starting XI: metric", lineup_metrics, format_func=charts.LINEUP_METRICS.get)
    st.image(str(charts.lineup_chart(result, figs / f"lineup_{metric}.png", metric)))
    for line in charts.lineup_summary(result, metric):
        st.markdown(f"- {line}")
    rows = charts.match_metric_rows(result, metric)
    if not rows.empty:
        unit = "" if metric == "rating" else " Values are per 90 minutes."
        st.caption(f"Every player who played, biggest drop first. Usual = his average in the earlier matches.{unit}")
        st.dataframe(rows, hide_index=True, use_container_width=True)
if meta["has_shots"]:
    with st.expander("Shot map (size = xG, star = goal)"):
        st.image(str(charts.shot_maps(result, figs / "shots.png")))
if not result["player_changes"].empty:
    st.image(str(charts.player_change_bars(result, figs / "players.png")))
elif meta["has_players"] and meta.get("mode") == "periods":
    st.info(f"No player-level comparison: no outfield player played at least {meta['min_match_minutes']} minutes in "
            f"each period, or none changed enough to matter.")
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
