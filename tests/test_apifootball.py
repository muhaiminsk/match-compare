"""Adapter tests against a fake API-Football server that mimics the documented v3 response shapes,
the daily-quota header, and the free-plan season error."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

from matchreport import charts, insights, metrics, store
from matchreport.sources import apifootball as af


def pl(pid, name, pos, mins, sub, rating, shots=(2, 1), passes=(50, 44), key=1, tackles=(2, 1), duels=(10, 6),
       dribbles=(3, 2, 1), fouls=(1, 2)):
    return {"player": {"id": pid, "name": name, "photo": ""}, "statistics": [{
        "games": {"minutes": mins, "number": 9, "position": pos, "rating": rating, "captain": False, "substitute": sub},
        "offsides": 0, "shots": {"total": shots[0], "on": shots[1]}, "goals": {"total": 0, "assists": 0},
        "passes": {"total": passes[0], "key": key, "accuracy": str(passes[1])},
        "tackles": {"total": tackles[0], "blocks": 0, "interceptions": tackles[1]},
        "duels": {"total": duels[0], "won": duels[1]},
        "dribbles": {"attempts": dribbles[0], "success": dribbles[1], "past": dribbles[2]},
        "fouls": {"drawn": fouls[0], "committed": fouls[1]}}]}


FIXTURES = [  # (id, date, home, away, goals_h, goals_a, status)
    (1001, "2024-08-24T19:00:00+00:00", ("Barcelona", 529), ("Getafe", 546), 3, 0, "FT"),
    (1002, "2024-09-15T19:00:00+00:00", ("Real Betis", 543), ("Barcelona", 529), 2, 5, "FT"),
    (1003, "2024-10-26T19:00:00+00:00", ("Barcelona", 529), ("Real Madrid", 541), 1, 2, "FT"),
    (1004, "2024-11-30T19:00:00+00:00", ("Barcelona", 529), ("Sevilla", 536), 0, 0, "NS"),
]


def players_block(fid):
    clasico = fid == 1003
    good, bad = (7.6, 7.2, 7.4), (5.8, 5.9, 6.0)
    r = bad if clasico else good
    barca = [pl(1, "Inaki Pena", "G", 90, False, "6.8", passes=(30, 24), duels=(2, 1), shots=(0, 0)),
             pl(2, "Lamine Yamal", "F", 90, False, str(r[0]), passes=(30 if clasico else 55, 20 if clasico else 48),
                dribbles=(8, 2 if clasico else 6, 0), shots=(4, 1)),
             pl(3, "Pedri", "M", 90, False, str(r[1]), passes=(40 if clasico else 85, 34 if clasico else 80), tackles=(1, 0)),
             pl(6, "Raphinha", "F", 90, False, str(r[2]), shots=(3, 1)),
             {**pl(4, "Ferran Torres", "F", 20, True, "6.5")},
             pl(5, "Bench Guy", "M", None, True, None)]
    other = [pl(20, "Rival Forward", "F", 90, False, "7.0")]
    other[0]["statistics"][0]["offsides"] = 6 if clasico else 1      # the high line catches Madrid offside
    return [{"team": {"id": 529, "name": "Barcelona"}, "players": barca},
            {"team": {"id": 999, "name": "Opponent"}, "players": other}]


class State:
    def __init__(self):
        self.requests = 0
        self.remaining = 100
        self.paths = []
        self.daily_limit = False


def make_server(state: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, body, code=200):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            state.remaining -= 1
            self.send_header("x-ratelimit-requests-remaining", str(state.remaining))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            state.requests += 1
            state.paths.append(u.path)
            if state.daily_limit:
                return self.reply({"errors": {"requests": "You have reached the request limit for the day, Go to "
                                              "https://dashboard.api-football.com to upgrade your plan."}, "response": []})
            if u.path == "/teams":
                return self.reply({"errors": [], "response": [{"team": {"id": 529, "name": "Barcelona"}},
                                                              {"team": {"id": 541, "name": "Real Madrid"}}]})
            if u.path == "/fixtures":
                if int(q["season"]) > 2024:
                    return self.reply({"errors": {"plan": "Free plans do not have access to this season, try from 2022 to 2024."}, "response": []})
                return self.reply({"errors": [], "response": [{
                    "fixture": {"id": i, "date": d, "status": {"short": s}}, "league": {"name": "La Liga"},
                    "teams": {"home": {"id": h[1], "name": h[0]}, "away": {"id": a[1], "name": a[0]}},
                    "goals": {"home": gh, "away": ga}} for i, d, h, a, gh, ga, s in FIXTURES]})
            fid = int(q.get("fixture", 0))
            if u.path == "/fixtures/players":
                return self.reply({"errors": [], "response": players_block(fid)})
            if u.path == "/fixtures/statistics":
                fx = next(f for f in FIXTURES if f[0] == fid)
                xg = {529: "0.85" if fid == 1003 else "3.10"}
                other = "1.95" if fid == 1003 else "0.60"
                return self.reply({"errors": [], "response": [
                    {"team": {"id": fx[2][1]}, "statistics": [{"type": "expected_goals", "value": xg.get(fx[2][1], other)}]},
                    {"team": {"id": fx[3][1]}, "statistics": [{"type": "expected_goals", "value": xg.get(fx[3][1], other)}]}]})
            if u.path == "/fixtures/lineups":
                return self.reply({"errors": [], "response": [{"team": {"id": 529}, "startXI": [
                    {"player": {"id": 1, "name": "Inaki Pena", "grid": "1:1"}},
                    {"player": {"id": 2, "name": "Lamine Yamal", "grid": "4:3"}}]}]})
            self.reply({"errors": ["unknown endpoint"], "response": []}, 404)
    return H


@pytest.fixture
def server():
    state = State()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_server(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", state
    httpd.shutdown()


def client(url, tmp_path, **kw):
    return af.Client(key="test", cache_dir=tmp_path / "cache", base=url, min_interval=0, **kw)


def test_parse_players_filters_unused_and_maps_positions():
    rows = af.parse_players(1001, {"response": players_block(1001)})
    names = {r["player"]: r for r in rows}
    assert "Bench Guy" not in names                      # never played: minutes is None
    assert names["Inaki Pena"]["position"] == "GK" and names["Inaki Pena"]["role"] == "G"
    assert names["Ferran Torres"]["position"] == "Sub"   # came off the bench
    assert names["Lamine Yamal"]["rating"] == 7.6 and names["Lamine Yamal"]["dribbles_completed"] == 6


def test_pass_accuracy_is_read_as_count_or_percent():
    rows = [{"passes": 50.0, "_pass_accuracy_raw": 44.0}, {"passes": 30.0, "_pass_accuracy_raw": 24.0}]
    assert af.resolve_pass_accuracy(rows) == "count" and rows[0]["passes_accurate"] == 44.0
    rows = [{"passes": 50.0, "_pass_accuracy_raw": 88.0}, {"passes": 30.0, "_pass_accuracy_raw": 24.0}]
    assert af.resolve_pass_accuracy(rows) == "percent" and rows[0]["passes_accurate"] == 44


def test_client_caches_and_respects_budget(server, tmp_path):
    url, state = server
    state.remaining = 5
    c = client(url, tmp_path, reserve=3)
    c.get("/teams", {"league": 140, "season": 2024})
    c.get("/teams", {"league": 140, "season": 2024})            # served from cache
    assert state.requests == 1 and c.remaining == 4
    c.get("/teams", {"league": 39, "season": 2024})
    with pytest.raises(af.BudgetReached):
        c.get("/teams", {"league": 78, "season": 2024})
    assert state.requests == 2                                   # the third call never reached the server


def test_plan_error_is_reported_and_not_cached(server, tmp_path):
    url, state = server
    c = client(url, tmp_path)
    for _ in range(2):
        with pytest.raises(af.PlanError, match="2022 to 2024"):
            c.get("/fixtures", {"league": 140, "season": 2026, "team": 529})
    assert state.requests == 2


def test_fetch_end_to_end_then_compare(server, tmp_path):
    url, state = server
    out = tmp_path / "data" / "raw" / "apifootball_test"
    info = af.fetch("ESP-La Liga", 2024, "Barcelona", out, client(url, tmp_path), with_lineups=True, progress=lambda *_: None)
    assert info["matches_saved"] == 3 and info["complete"] and info["has_xg"] and info["pass_accuracy_is"] == "count"
    assert info["player_rows"] == 18                              # (5 Barcelona players who played + 1 rival) x 3 matches; the unused sub is dropped

    matches, shots, pm = store.load_all(tmp_path / "data", sources=["apifootball"])
    assert len(matches) == 3 and shots.empty
    r = metrics.compare(matches, shots, pm, "Barcelona", "Real Madrid")
    meta = r["meta"]
    assert meta["has_xg"] and not meta["has_shots"] and meta["has_players"] and meta["n_base"] == 2
    assert {"rating", "passes", "dribbles_completed"} <= set(r["players"]["metric"])
    tp = r["team_player"]
    assert tp.loc["passes", "match"] < tp.loc["passes", "base"]
    found = insights.findings(r)
    assert any("created 0.85 xG" in s for s in found)
    assert any("regulars averaged a rating of" in s for s in found)
    assert any(s.startswith("Lamine Yamal (forward):") and "rating fell from an average of 7.60 before to 5.80" in s for s in found)
    assert any("passes" in s for s in found)
    assert charts.available_lineup_metrics(r)[0] == "rating"
    assert charts.lineup_chart(r, tmp_path / "rating.png", "rating").stat().st_size > 5000


def test_fetch_resumes_after_budget_runs_out(server, tmp_path):
    url, state = server
    out = tmp_path / "data" / "raw" / "apifootball_test"
    state.remaining = 9
    first = af.fetch("ESP-La Liga", 2024, "Barcelona", out, client(url, tmp_path), progress=lambda *_: None)
    assert not first["complete"] and first["matches_saved"] == 2
    state.remaining, before = 100, state.requests
    second = af.fetch("ESP-La Liga", 2024, "Barcelona", out, client(url, tmp_path), progress=lambda *_: None)
    assert second["complete"] and second["matches_saved"] == 3
    assert state.requests - before <= 3                          # only the missing match (plus maybe the fixture list refresh)


def test_grid_positions_put_column_one_on_the_right():
    grids = ["1:1", "2:1", "2:2", "2:3", "2:4", "3:1", "3:2", "3:3", "4:1", "4:2", "4:3"]
    lineup = pd.DataFrame({"player": [f"P{i}" for i in range(11)], "position": ["GK"] + ["D"] * 4 + ["M"] * 3 + ["F"] * 3,
                           "minutes": 90, "grid": grids})
    pos = charts.lineup_positions(lineup).set_index("player")
    assert pos.loc["P0", "x"] == 6 and pos.loc["P1", "y"] < pos.loc["P4", "y"]   # right back below left back in y
    assert pos.loc["P1", "x"] < pos.loc["P5", "x"] < pos.loc["P8", "x"]           # defenders behind midfield behind forwards
    assert pos[["x", "y"]].min().min() >= 0 and pos[["x", "y"]].max().max() <= 100


def test_offsides_and_result_sentence(server, tmp_path):
    url, state = server
    out = tmp_path / "data" / "raw" / "apifootball_test"
    af.fetch("ESP-La Liga", 2024, "Barcelona", out, client(url, tmp_path), progress=lambda *_: None)
    matches, shots, pm = store.load_all(tmp_path / "data", sources=["apifootball"])
    r = metrics.compare(matches, shots, pm, "Barcelona", "Real Madrid")
    assert r["team_player"].loc["opp_offsides", "match"] == 6 and r["team_player"].loc["opp_offsides", "base"] == 1
    found = insights.findings(r)
    assert found[0].startswith("Barcelona lost 1-2: they scored 1 from 0.85 xG and conceded 2 from 1.95 xG")
    assert any("caught their opponent offside 6 times versus 1.0 per match" in s for s in found)


def test_daily_limit_message_stops_cleanly_and_still_writes_cached_matches(server, tmp_path):
    url, state = server
    out = tmp_path / "data" / "raw" / "apifootball_test"
    state.remaining = 9
    af.fetch("ESP-La Liga", 2024, "Barcelona", out, client(url, tmp_path), progress=lambda *_: None)   # caches 2 matches
    (out / "matches.csv").unlink()
    import os, time
    for f in (tmp_path / "cache").glob("fixtures__*.json"):        # the fixture list is now older than its 6-hour refresh
        old = time.time() - 7 * 3600
        os.utime(f, (old, old))
    state.daily_limit = True                                       # provider now refuses every request for the day
    info = af.fetch("ESP-La Liga", 2024, "Barcelona", out, client(url, tmp_path), progress=lambda *_: None)
    assert not info["complete"] and info["matches_saved"] == 2 and "request limit for the day" in info["stopped"]
    assert len(pd.read_csv(out / "matches.csv")) == 2              # rebuilt from the cache despite the refusal
