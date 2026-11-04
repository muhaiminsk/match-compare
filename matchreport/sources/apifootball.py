"""API-Football (api-sports.io, v3) adapter: per-player passes, tackles, dribbles, duels, ratings, and team xG.

Design points
- Every response is cached on disk, so a match is never requested twice.
- The client tracks the daily quota from the response headers and stops cleanly before it runs out. Run the
  fetch again the next day and it resumes from the cache, so a free key (100 requests a day) can still build a season.
- The free plan only serves some seasons; a plan error is raised with the provider's own message.
"""
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from matchreport import schema

BASE = "https://v3.football.api-sports.io"
FINISHED = {"FT", "AET", "PEN"}
LEAGUES = {"ESP-La Liga": 140, "ENG-Premier League": 39, "GER-Bundesliga": 78, "ITA-Serie A": 135, "FRA-Ligue 1": 61}


class ApiError(RuntimeError):
    pass


class PlanError(ApiError):
    """The plan does not include this season or endpoint."""


class BudgetReached(ApiError):
    """The daily request budget is (almost) used up."""


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def num(v):
    """Convert API values (None, '7.3', '88%', 12) to float."""
    if v is None or v == "":
        return np.nan
    try:
        return float(str(v).replace("%", "").strip())
    except ValueError:
        return np.nan


class Client:
    def __init__(self, key=None, cache_dir=Path("data/cache/apifootball"), base=BASE, min_interval=None, reserve=3):
        self.key = key or os.environ.get("APIFOOTBALL_KEY", "")
        self.cache_dir = Path(cache_dir)
        self.base = base
        # The free plan allows about 10 requests a minute; set APIFOOTBALL_MIN_INTERVAL=0.3 on a paid plan.
        self.min_interval = float(os.environ.get("APIFOOTBALL_MIN_INTERVAL", 6.5)) if min_interval is None else min_interval
        self.reserve = reserve
        self.remaining = None   # daily requests left, from the latest response headers
        self.used = 0
        self.limit_message = None   # the provider's own words once it refuses for the day
        self._last = 0.0

    def _path(self, endpoint, params):
        name = endpoint.strip("/").replace("/", "_") + "__" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        return self.cache_dir / (re.sub(r"[^A-Za-z0-9_.-]", "", name) + ".json")

    def cached(self, endpoint: str, params: dict):
        """The cached response regardless of age, or None."""
        path = self._path(endpoint, params)
        return json.loads(path.read_text()) if path.exists() else None

    def get(self, endpoint: str, params: dict, ttl_hours=None) -> dict:
        path = self._path(endpoint, params)
        if path.exists():
            fresh = ttl_hours is None or (time.time() - path.stat().st_mtime) < ttl_hours * 3600
            if fresh:
                return json.loads(path.read_text())
        if not self.key:
            raise ApiError("APIFOOTBALL_KEY is not set. Export your API-Football key first.")
        if self.remaining is not None and self.remaining <= self.reserve:
            raise BudgetReached(self.limit_message or f"Daily request budget reached ({self.remaining} left). "
                                f"Run again tomorrow; everything fetched so far is cached.")
        for attempt in range(2):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            r = requests.get(f"{self.base}{endpoint}", headers={"x-apisports-key": self.key}, params=params, timeout=30)
            self._last = time.time()
            self.used += 1
            left = r.headers.get("x-ratelimit-requests-remaining")
            if left is not None and str(left).isdigit():
                self.remaining = int(left)
            if r.status_code == 429 and attempt == 0:
                time.sleep(65)
                continue
            r.raise_for_status()
            data = r.json()
            errors = data.get("errors")
            if errors:
                text = " ".join(str(v) for v in (errors.values() if isinstance(errors, dict) else errors))
                if isinstance(errors, dict) and "plan" in errors:
                    raise PlanError(str(errors["plan"]))
                if isinstance(errors, dict) and "requests" in errors:   # daily limit: stop, keep what is cached
                    self.remaining = 0
                    self.limit_message = f"{errors['requests']} Everything fetched so far is cached; run again tomorrow."
                    raise BudgetReached(self.limit_message)
                if isinstance(errors, dict) and "rateLimit" in errors and attempt == 0:   # per-minute limit: wait once
                    time.sleep(65)
                    continue
                raise ApiError(text)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
            return data
        raise ApiError("Rate limited twice; try again in a few minutes.")


# ------------------------------------------------------------------ parsers (pure functions, easy to test)
def parse_players(fixture_id: int, resp: dict) -> list:
    """One dict per player who played, for both teams, from /fixtures/players."""
    rows = []
    for block in resp.get("response", []):
        team = block["team"]["name"]
        for entry in block.get("players", []):
            st = (entry.get("statistics") or [{}])[0]
            games = st.get("games") or {}
            minutes = num(games.get("minutes"))
            if not minutes or np.isnan(minutes) or minutes <= 0:
                continue
            role = games.get("position")  # G, D, M, F
            position = "Sub" if games.get("substitute") else ("GK" if role == "G" else role)
            shots, passes, tackles = st.get("shots") or {}, st.get("passes") or {}, st.get("tackles") or {}
            duels, dribbles, fouls = st.get("duels") or {}, st.get("dribbles") or {}, st.get("fouls") or {}
            rows.append({
                "game_id": fixture_id, "team": team, "player": entry["player"]["name"], "player_id": entry["player"]["id"],
                "position": position, "role": role, "grid": None, "minutes": minutes,
                "shots": num(shots.get("total")), "shots_on_target": num(shots.get("on")),
                "key_passes": num(passes.get("key")), "passes": num(passes.get("total")),
                "_pass_accuracy_raw": num(passes.get("accuracy")),
                "tackles": num(tackles.get("total")), "interceptions": num(tackles.get("interceptions")),
                "duels": num(duels.get("total")), "duels_won": num(duels.get("won")),
                "dribbles_attempted": num(dribbles.get("attempts")), "dribbles_completed": num(dribbles.get("success")),
                "dribbled_past": num(dribbles.get("past")),
                "fouls_committed": num(fouls.get("committed")), "fouls_drawn": num(fouls.get("drawn")),
                "offsides": num(st.get("offsides")),
                "rating": num(games.get("rating")), "source": "apifootball",
            })
    return rows


def resolve_pass_accuracy(rows: list) -> str:
    """Fill passes_accurate. In /fixtures/players the 'accuracy' field is documented loosely: it is normally the
    NUMBER of accurate passes, but could be a percentage. If any accuracy exceeds the passes attempted it cannot
    be a count, so the whole set is treated as a percentage. Returns which interpretation was used."""
    is_percent = any(r["_pass_accuracy_raw"] > r["passes"] for r in rows
                     if not np.isnan(r["_pass_accuracy_raw"]) and not np.isnan(r["passes"]))
    for r in rows:
        raw, total = r.pop("_pass_accuracy_raw"), r["passes"]
        if np.isnan(raw) or np.isnan(total):
            r["passes_accurate"] = np.nan
        else:
            r["passes_accurate"] = round(total * raw / 100) if is_percent else raw
    return "percent" if is_percent else "count"


def parse_team_stats(resp: dict) -> dict:
    """team_id -> {'xg': float} from /fixtures/statistics (xG is only present for some seasons)."""
    out = {}
    for block in resp.get("response", []):
        stats = {str(s.get("type", "")).lower().replace(" ", "_"): s.get("value") for s in block.get("statistics", [])}
        out[block["team"]["id"]] = {"xg": num(stats.get("expected_goals"))}
    return out


def parse_lineups(resp: dict) -> dict:
    """player_id -> grid ('row:col') for starters, from /fixtures/lineups."""
    grid = {}
    for block in resp.get("response", []):
        for e in block.get("startXI", []):
            p = e.get("player", {})
            if p.get("id") is not None and p.get("grid"):
                grid[p["id"]] = p["grid"]
    return grid


# ------------------------------------------------------------------ fetch
def find_team(teams_resp: dict, name: str) -> dict:
    wanted = norm(name)
    teams = [t["team"] for t in teams_resp.get("response", [])]
    exact = [t for t in teams if norm(t["name"]) == wanted or str(t["id"]) == str(name)]
    near = exact or [t for t in teams if wanted in norm(t["name"])]
    if len(near) != 1:
        raise ApiError(f"Could not pick one team for '{name}'. Options: {sorted(t['name'] for t in near or teams)}")
    return near[0]


def fetch(league: str, season: int, team: str, out_dir: Path, client: Client = None, with_lineups: bool = False,
          progress=print) -> dict:
    client = client or Client()
    league_id = LEAGUES.get(league) or int(league)
    team_row = find_team(client.get("/teams", {"league": league_id, "season": season}), team)
    fx_params = {"league": league_id, "season": season, "team": team_row["id"]}
    try:
        fx = client.get("/fixtures", fx_params, ttl_hours=6)["response"]
    except BudgetReached:
        stale = client.cached("/fixtures", fx_params)
        if stale is None:
            raise
        fx = stale["response"]   # out of requests: an older fixture list is still fine for rebuilding from the cache
        progress("Daily limit reached: using the cached fixture list.")
    finished = sorted((f for f in fx if f["fixture"]["status"]["short"] in FINISHED), key=lambda f: f["fixture"]["date"])
    progress(f"{team_row['name']} {season}: {len(finished)} finished matches of {len(fx)} fixtures")

    match_rows, player_rows, done, stopped = [], [], 0, None
    for f in finished:
        fid = f["fixture"]["id"]
        try:
            players = client.get("/fixtures/players", {"fixture": fid})
            stats = client.get("/fixtures/statistics", {"fixture": fid})
            grid = parse_lineups(client.get("/fixtures/lineups", {"fixture": fid})) if with_lineups else {}
        except BudgetReached as e:
            stopped = str(e)
            break
        rows = parse_players(fid, players)
        for r in rows:
            r["grid"] = grid.get(r["player_id"])
        player_rows += rows
        xg = parse_team_stats(stats)
        home, away = f["teams"]["home"], f["teams"]["away"]
        match_rows.append({
            "game_id": fid, "date": pd.to_datetime(f["fixture"]["date"], utc=True).tz_localize(None),
            "competition": f["league"]["name"], "home_team": home["name"], "away_team": away["name"],
            "home_goals": f["goals"]["home"], "away_goals": f["goals"]["away"],
            "home_xg": xg.get(home["id"], {}).get("xg", np.nan), "away_xg": xg.get(away["id"], {}).get("xg", np.nan),
            "source": "apifootball",
        })
        done += 1
        progress(f"  {done}/{len(finished)} {home['name']} {f['goals']['home']}-{f['goals']['away']} {away['name']}"
                 f"  (requests used: {client.used}, left today: {client.remaining})")

    accuracy_mode = resolve_pass_accuracy(player_rows) if player_rows else "n/a"
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = pd.DataFrame(match_rows, columns=schema.MATCHES)
    matches.to_csv(out_dir / "matches.csv", index=False)
    players_df = schema.conform(pd.DataFrame(player_rows), schema.PLAYER_MATCH) if player_rows else pd.DataFrame(columns=schema.PLAYER_MATCH)
    players_df.to_csv(out_dir / "player_match.csv", index=False)
    return {"team": team_row["name"], "matches_saved": done, "matches_total": len(finished), "player_rows": len(player_rows),
            "requests_used": client.used, "requests_left_today": client.remaining, "pass_accuracy_is": accuracy_mode,
            "complete": stopped is None, "stopped": stopped, "has_xg": bool(matches["home_xg"].notna().any()),
            "output": str(out_dir)}
