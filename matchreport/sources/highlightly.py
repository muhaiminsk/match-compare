"""Highlightly adapter (free plan: 100 requests a day): per-player passes, dribbles, tackles, duels, ratings,
offsides, xG and xA for current matches, plus team statistics and lineups.

Endpoints used (base https://sports.highlightly.net/football, header x-rapidapi-key):
  /matches?homeTeamName=..&season=..   and   /matches?awayTeamName=..&season=..   (paged, limit 100)
  /box-score/{matchId}       per-player statistics, grouped by team
  /statistics/{matchId}      team statistics (expected goals among them)
  /lineups/{matchId}         formation and starting rows (optional)
Every response is cached on disk and a daily-quota refusal stops the fetch cleanly, keeping what is cached.
"""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from matchreport import schema

BASE = os.environ.get("HIGHLIGHTLY_BASE", "https://sports.highlightly.net/football")
ROLE = {"Goalkeeper": "G", "Defender": "D", "Midfielder": "M", "Forward": "F"}


class ApiError(RuntimeError):
    pass


class BudgetReached(ApiError):
    """The daily request quota is used up."""


def num(v):
    if v is None or v == "":
        return np.nan
    try:
        return float(str(v).replace("%", "").strip())
    except ValueError:
        return np.nan


def _rating(v):
    """Match rating, or NaN: the provider reports 0 when a player has no rating."""
    r = num(v)
    return np.nan if (np.isnan(r) or r <= 0) else r


class Client:
    def __init__(self, key=None, cache_dir=Path("data/cache/highlightly"), base=None, min_interval=None):
        self.key = key or os.environ.get("HIGHLIGHTLY_KEY", "")
        self.cache_dir = Path(cache_dir)
        self.base = base or BASE
        self.min_interval = float(os.environ.get("HIGHLIGHTLY_MIN_INTERVAL", 1.0)) if min_interval is None else min_interval
        self.used = 0
        self.limit_message = None
        self._last = 0.0

    def _path(self, endpoint, params):
        name = endpoint.strip("/").replace("/", "_") + "__" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        return self.cache_dir / (re.sub(r"[^A-Za-z0-9_.-]", "", name) + ".json")

    def cached(self, endpoint, params=None):
        path = self._path(endpoint, params or {})
        return json.loads(path.read_text()) if path.exists() else None

    def get(self, endpoint: str, params: dict = None, ttl_hours=None):
        params = params or {}
        path = self._path(endpoint, params)
        if path.exists() and (ttl_hours is None or (time.time() - path.stat().st_mtime) < ttl_hours * 3600):
            return json.loads(path.read_text())
        if not self.key:
            raise ApiError("HIGHLIGHTLY_KEY is not set. Export your Highlightly API key first.")
        if self.limit_message:
            raise BudgetReached(self.limit_message)
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(f"{self.base}{endpoint}", headers={"x-rapidapi-key": self.key}, params=params, timeout=30)
        except requests.RequestException as e:
            raise ApiError(f"Could not reach Highlightly ({type(e).__name__}). Check your connection and try again.")
        self._last = time.time()
        self.used += 1
        text = r.text or ""
        if r.status_code == 429 or "quota" in text.lower():
            self.limit_message = ("Highlightly daily quota reached. Everything fetched so far is cached; "
                                  "run again after the quota resets (midnight UTC).")
            raise BudgetReached(self.limit_message)
        if r.status_code in (401, 403):
            raise ApiError("Highlightly rejected the API key. Check that it is copied correctly and active.")
        if r.status_code >= 400:
            raise ApiError(f"Highlightly returned an error ({r.status_code}). Try again later.")
        try:
            data = r.json()
        except ValueError:
            raise ApiError("Highlightly sent a response that is not valid data. Try again later.")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return data


# ------------------------------------------------------------------ parsers (pure functions)
def parse_score(match: dict):
    """(home_goals, away_goals) from state.score.current like '1 - 2'; (None, None) if absent."""
    cur = ((match.get("state") or {}).get("score") or {}).get("current")
    if not cur:
        return None, None
    m = re.match(r"\s*(\d+)\s*-\s*(\d+)", str(cur))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def is_finished(match: dict) -> bool:
    state = match.get("state") or {}
    desc = str(state.get("description") or state.get("status") or "").lower()
    if "finish" in desc or desc in {"ft", "aet", "pen", "full time"}:
        return True
    hg, ag = parse_score(match)
    return hg is not None and (state.get("clock") or 0) >= 90


def parse_box_score(match_id, resp: list) -> list:
    rows = []
    for block in resp or []:
        team = block["team"]["name"]
        for p in block.get("players", []):
            s = p.get("statistics") or {}
            if isinstance(s, list):          # documented as a list, served as an object: accept both
                s = s[0] if s else {}
            minutes = num(p.get("minutesPlayed"))
            if np.isnan(minutes) or minutes <= 0:
                continue
            role = ROLE.get(p.get("position"))
            position = "Sub" if p.get("isSubstitute") else ("GK" if role == "G" else role)
            rows.append({
                "game_id": match_id, "team": team, "player": p["name"], "player_id": p["id"],
                "position": position, "role": role, "grid": None, "minutes": minutes,
                "shots": num(s.get("shotsTotal")), "shots_on_target": num(s.get("shotsOnTarget")),
                "key_passes": num(s.get("passesKey")), "xg": num(s.get("expectedGoals")), "xa": num(s.get("expectedAssists")),
                "passes": num(s.get("passesTotal")), "passes_accurate": num(s.get("passesSuccessful")),
                "tackles": num(s.get("tacklesTotal")), "interceptions": num(s.get("interceptionsTotal")),
                "duels": num(s.get("duelsTotal")), "duels_won": num(s.get("duelsWon")),
                "dribbles_attempted": num(s.get("dribblesTotal")), "dribbles_completed": num(s.get("dribblesSuccessful")),
                "dribbled_past": np.nan, "fouls_committed": num(s.get("fouledOthers")), "fouls_drawn": num(s.get("fouledByOthers")),
                "offsides": num(p.get("offsides")), "rating": _rating(p.get("matchRating")),
                "goals": num(s.get("goalsScored")), "assists": num(s.get("assists")), "source": "highlightly",
            })
    return rows


def parse_team_stats(resp: list) -> dict:
    """team_id -> {'xg': float}. Statistics come as [{'value': 1.19, 'displayName': 'Expected Goals'}, ...]."""
    out = {}
    for block in resp or []:
        stats = {str(s.get("displayName", "")).strip().lower(): s.get("value") for s in block.get("statistics", [])}
        out[block["team"]["id"]] = {"xg": num(stats.get("expected goals"))}
    return out


def parse_lineups(resp: dict) -> dict:
    """player_id -> 'row:col' grid for starters. The starting lineup may come as rows (list of lists, from the
    goalkeeper outward) or as a flat list; a flat list gives no grid."""
    grid = {}
    for side in ("homeTeam", "awayTeam"):
        block = (resp or {}).get(side) or {}
        xi = block.get("initialLineup") or block.get("startingXI") or []
        if xi and isinstance(xi[0], list):
            for r, row in enumerate(xi, start=1):
                for c, p in enumerate(row, start=1):
                    if isinstance(p, dict) and p.get("id") is not None:
                        grid[p["id"]] = f"{r}:{c}"
    return grid


# ------------------------------------------------------------------ fetch
def list_team_matches(client: Client, team: str, season: int, league: str = None) -> list:
    """Every match of the team in the season (all competitions unless league is given), via home + away queries."""
    seen, out = set(), []
    for side in ("homeTeamName", "awayTeamName"):
        offset = 0
        while True:
            params = {side: team, "season": season, "limit": 100, "offset": offset}
            if league:
                params["leagueName"] = league
            page = client.get("/matches", params, ttl_hours=6)
            for m in page.get("data", []):
                if m["id"] not in seen:
                    seen.add(m["id"])
                    out.append(m)
            pg = page.get("pagination") or {}
            offset += int(pg.get("limit") or 100)
            if offset >= int(pg.get("totalCount") or 0):
                break
    return sorted(out, key=lambda m: m.get("date") or "")


def fetch(team: str, season: int, out_dir: Path, client: Client = None, league: str = None,
          with_lineups: bool = False, progress=print) -> dict:
    client = client or Client()
    try:
        matches = list_team_matches(client, team, season, league)
    except BudgetReached:
        raise
    finished = [m for m in matches if is_finished(m)]
    progress(f"{team} {season}: {len(finished)} finished matches of {len(matches)} listed")

    match_rows, player_rows, stopped = [], [], None
    for m in finished:
        mid = m["id"]
        try:
            box = client.get(f"/box-score/{mid}")
            stats = client.get(f"/statistics/{mid}")
            grid = parse_lineups(client.get(f"/lineups/{mid}")) if with_lineups else {}
        except BudgetReached as e:
            stopped = str(e)
            break
        rows = parse_box_score(mid, box)
        for r in rows:
            r["grid"] = grid.get(r["player_id"])
        player_rows += rows
        xg = parse_team_stats(stats)
        home, away = m["homeTeam"], m["awayTeam"]
        hg, ag = parse_score(m)
        match_rows.append({
            "game_id": mid, "date": pd.to_datetime(m["date"], utc=True).tz_localize(None),
            "competition": (m.get("league") or {}).get("name"), "home_team": home["name"], "away_team": away["name"],
            "home_goals": hg, "away_goals": ag,
            "home_xg": xg.get(home["id"], {}).get("xg", np.nan), "away_xg": xg.get(away["id"], {}).get("xg", np.nan),
            "source": "highlightly",
        })
        progress(f"  {len(match_rows)}/{len(finished)} {home['name']} {hg}-{ag} {away['name']}  "
                 f"[{(m.get('league') or {}).get('name')}]  (requests used: {client.used})")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(match_rows, columns=schema.MATCHES).to_csv(out_dir / "matches.csv", index=False)
    pdf = schema.conform(pd.DataFrame(player_rows), schema.PLAYER_MATCH) if player_rows else pd.DataFrame(columns=schema.PLAYER_MATCH)
    pdf.to_csv(out_dir / "player_match.csv", index=False)
    return {"team": team, "matches_saved": len(match_rows), "matches_total": len(finished), "player_rows": len(player_rows),
            "requests_used": client.used, "complete": stopped is None, "stopped": stopped,
            "has_xg": bool(match_rows) and any(pd.notna(r["home_xg"]) for r in match_rows), "output": str(out_dir)}
