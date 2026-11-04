"""Command line.
  Fetch:  python report.py fetch --league "ESP-La Liga" --season 2026-2027 --team Barcelona
  Highlightly:  python report.py fetch-hl --team Barcelona --season 2026 --lineups
  API-Football: python report.py fetch-api --league "ESP-La Liga" --season 2024 --team Barcelona
  Report: python report.py --team Barcelona --vs "Real Madrid" [--baseline before|after|all] [--no-llm] [--source understat|apifootball]"""
import argparse
import re
from pathlib import Path

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd")
f = sub.add_parser("fetch", help="download a team's season from Understat into data/raw")
f.add_argument("--league", default="ESP-La Liga"); f.add_argument("--season", default="2026-2027")
f.add_argument("--team", required=True)
g = sub.add_parser("fetch-api", help="download a team's season from API-Football (needs APIFOOTBALL_KEY); resumable")
g.add_argument("--league", default="ESP-La Liga", help="ESP-La Liga, ENG-Premier League, GER-Bundesliga, ITA-Serie A, FRA-Ligue 1, or a league id")
g.add_argument("--season", type=int, required=True, help="start year, e.g. 2024 for 2024/25 (free plans: 2022 to 2024)")
g.add_argument("--team", required=True)
g.add_argument("--lineups", action="store_true", help="also fetch lineups (grid positions for the lineup chart); +1 request per match")
g.add_argument("--data", default="data")
h = sub.add_parser("fetch-hl", help="download a team's season from Highlightly (needs HIGHLIGHTLY_KEY); free plan: 100 requests a day")
h.add_argument("--team", required=True, help="team name as Highlightly spells it, e.g. Barcelona")
h.add_argument("--season", type=int, required=True, help="start year, e.g. 2026 for 2026/27")
h.add_argument("--league", help="limit to one competition, e.g. 'La Liga'; default: every competition")
h.add_argument("--lineups", action="store_true", help="also fetch lineups (grid positions); +1 request per match")
h.add_argument("--data", default="data")
ap.add_argument("--team"); ap.add_argument("--vs", help="opponent name (partial is fine)")
ap.add_argument("--date", help="YYYY-MM-DD, if the teams met more than once")
ap.add_argument("--baseline", default="before", choices=["before", "after", "all"])
ap.add_argument("--all-competitions", action="store_true", help="baseline uses every match, not just the same competition")
ap.add_argument("--no-llm", action="store_true", help="skip the written summary")
ap.add_argument("--source", choices=["understat", "apifootball", "highlightly"], help="use only this data source (default: all on disk)")
ap.add_argument("--data", default="data"); ap.add_argument("--out", default="reports")
a = ap.parse_args()

if a.cmd == "fetch":
    from matchreport.sources import understat
    out = Path(a.data) / "raw" / f"understat_{a.league}_{a.season}_{a.team}".replace(" ", "_")
    print(understat.fetch(a.league, a.season, a.team, out), "->", out)
elif a.cmd == "fetch-hl":
    from matchreport.sources import highlightly
    client = highlightly.Client(cache_dir=Path(a.data) / "cache" / "highlightly")
    slug = re.sub(r"[^a-z0-9]", "", a.team.lower())
    out = Path(a.data) / "raw" / f"highlightly_{a.season}_{slug}"
    try:
        info = highlightly.fetch(a.team, a.season, out, client, league=a.league, with_lineups=a.lineups)
    except highlightly.ApiError as e:
        raise SystemExit(f"Highlightly error: {e}")
    print(info)
    if not info["complete"]:
        print("Stopped early:", info["stopped"])
elif a.cmd == "fetch-api":
    from matchreport.sources import apifootball
    client = apifootball.Client(cache_dir=Path(a.data) / "cache" / "apifootball")
    slug = apifootball.norm(a.team)
    out = Path(a.data) / "raw" / f"apifootball_{a.league}_{a.season}_{slug}".replace(" ", "_")
    try:
        info = apifootball.fetch(a.league, a.season, a.team, out, client, with_lineups=a.lineups)
    except apifootball.PlanError as e:
        raise SystemExit(f"Your plan does not include this: {e}")
    except apifootball.ApiError as e:
        raise SystemExit(f"API-Football error: {e}")
    print(info)
    if not info["complete"]:
        print("Stopped early:", info["stopped"])
    print(f"Pass accuracy was read as a {info['pass_accuracy_is']}. Check one player against the match page to confirm.")
else:
    if not (a.team and a.vs):
        ap.error("--team and --vs are required (or use the fetch subcommand)")
    from matchreport import report
    try:
        out = report.build(a.team, a.vs, Path(a.data), Path(a.out), a.date, llm=not a.no_llm,
                           same_competition=not a.all_competitions, baseline=a.baseline, source=a.source)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"Could not build the report: {e}")
    print("Report written to", out)
