"""Adapter tests against a fake Highlightly server using the response shapes observed in a real call
(statistics served as an object, score as 'h - a', ratings as strings, 23 listed players per team)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from matchreport import charts, insights, metrics, store
from matchreport.sources import highlightly as hl

MATCHES = [  # id, date, home, away, score, league
    (1, "2026-08-23T19:30:00.000Z", ("Elche", 11), ("Barcelona", 450963), "0 - 5", "La Liga"),
    (2, "2026-09-16T19:30:00.000Z", ("Barcelona", 450963), ("Racing", 12), "7 - 2", "La Liga"),
    (3, "2026-10-20T19:00:00.000Z", ("Paris Saint Germain", 73119), ("Barcelona", 450963), "4 - 2", "UEFA Champions League"),
    (4, "2026-10-25T20:00:00.000Z", ("Barcelona", 450963), ("Real Madrid", 461175), "1 - 2", "La Liga"),
    (5, "2026-11-08T20:00:00.000Z", ("Barcelona", 450963), ("Celta", 13), None, "La Liga"),  # not played yet
]


def match_obj(mid, date, home, away, score, league):
    state = {"clock": 90, "description": "Finished", "score": {"current": score}} if score else {"description": "Not started"}
    return {"id": mid, "date": date, "state": state, "league": {"id": 1, "name": league, "season": 2026},
            "homeTeam": {"id": home[1], "name": home[0]}, "awayTeam": {"id": away[1], "name": away[0]}}


def player(pid, name, pos, mins, sub, rating, passes=(80, 90), drib=(1, 2), tackles=1, duels=(3, 6), shots=(1, 2), xg=0.1):
    return {"id": pid, "name": name, "position": pos, "minutesPlayed": mins, "isSubstitute": sub, "isCaptain": False,
            "matchRating": f"{rating:.2f}" if rating else None, "offsides": 1 if name == "Lamine Yamal" else 0,
            "statistics": {"goalsScored": 0, "assists": 0, "passesSuccessful": passes[0], "passesTotal": passes[1],
                           "passesFailed": passes[1] - passes[0], "passesAccuracy": "80.00 %", "passesKey": 2,
                           "dribblesSuccessful": drib[0], "dribblesTotal": drib[1], "dribblesFailed": drib[1] - drib[0],
                           "tacklesTotal": tackles, "interceptionsTotal": 1, "duelsWon": duels[0], "duelsTotal": duels[1],
                           "duelsLost": duels[1] - duels[0], "shotsOnTarget": shots[0], "shotsTotal": shots[1],
                           "shotsOffTarget": shots[1] - shots[0], "fouledOthers": 1, "fouledByOthers": 2,
                           "cardsYellow": 0, "cardsRed": 0, "cardsSecondYellow": 0, "goalsSaved": 0, "goalsConceded": 0,
                           "expectedGoals": xg, "expectedAssists": 0.1}}


def box(mid):
    clasico = mid == 4
    barca = [player(100, "Joan García", "Goalkeeper", 90, False, 6.8, passes=(16, 19), shots=(0, 0), xg=None),
             player(101, "Rodri", "Midfielder", 90, False, 6.6 if clasico else 7.4, passes=(84, 96) if clasico else (60, 68)),
             player(102, "Pedri", "Midfielder", 90, False, 7.6, passes=(78, 88) if clasico else (70, 78)),
             player(103, "Lamine Yamal", "Forward", 90, False, 6.25 if clasico else 7.9,
                    passes=(36, 47), drib=(4, 9) if clasico else (3, 5), shots=(1, 5), xg=0.51),
             player(104, "Gerard Martín", "Defender", 90, False, 6.2, passes=(72, 81), duels=(2, 3)),
             player(105, "Dani Olmo", "Midfielder", 0, True, None),            # unused sub: 0 minutes
             player(106, "Fermín López", "Midfielder", 20, True, 6.5)]
    opp = [player(200, "Opp Player", "Forward", 90, False, 7.0, passes=(20, 30), shots=(2, 3), xg=0.6)]
    return [{"team": {"id": 450963, "name": "Barcelona"}, "players": barca},
            {"team": {"id": 999, "name": "Opponent"}, "players": opp}]


class State:
    def __init__(self):
        self.requests, self.quota_left, self.paths = 0, 100, []


def make_server(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, code=200):
            data = json.dumps(body).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
            state.requests += 1; state.paths.append(u.path)
            if state.quota_left <= 0:
                return self.send({"message": "You have exceeded the DAILY quota for Requests on your current plan, BASIC."}, 429)
            state.quota_left -= 1
            if u.path == "/matches":
                side = "homeTeamName" if "homeTeamName" in q else "awayTeamName"
                idx = 2 if side == "homeTeamName" else 3
                rows = [match_obj(*m) for m in MATCHES if m[idx][0] == q[side] and int(q["season"]) == 2026
                        and (not q.get("leagueName") or m[5] == q["leagueName"])]
                return self.send({"data": rows, "pagination": {"totalCount": len(rows), "offset": 0, "limit": 100},
                                  "plan": {"tier": "BASIC", "message": "All data available with current plan."}})
            mid = int(u.path.rsplit("/", 1)[-1])
            if u.path.startswith("/box-score/"):
                return self.send(box(mid))
            if u.path.startswith("/statistics/"):
                m = next(x for x in MATCHES if x[0] == mid)
                xg = {450963: 1.19 if mid == 4 else 3.0}
                return self.send([{"team": {"id": m[2][1], "name": m[2][0]}, "statistics": [{"value": xg.get(m[2][1], 1.5), "displayName": "Expected Goals"}]},
                                  {"team": {"id": m[3][1], "name": m[3][0]}, "statistics": [{"value": xg.get(m[3][1], 1.5), "displayName": "Expected Goals"}]}])
            if u.path.startswith("/lineups/"):
                return self.send({"homeTeam": {"formation": "4-3-3", "initialLineup": [[{"id": 100, "name": "Joan García"}],
                                  [{"id": 104, "name": "Gerard Martín"}], [{"id": 101, "name": "Rodri"}, {"id": 102, "name": "Pedri"}],
                                  [{"id": 103, "name": "Lamine Yamal"}]], "substitutes": []}, "awayTeam": {"initialLineup": []}})
            self.send({"error": "unknown"}, 404)
    return H


@pytest.fixture
def server():
    state = State()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_server(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", state
    httpd.shutdown()


def client(url, tmp_path):
    return hl.Client(key="test", cache_dir=tmp_path / "cache", base=url, min_interval=0)


def test_parsers_match_observed_shapes():
    assert hl.parse_score({"state": {"score": {"current": "1 - 2"}}}) == (1, 2)
    assert hl.is_finished(match_obj(*MATCHES[3])) and not hl.is_finished(match_obj(*MATCHES[4]))
    rows = {r["player"]: r for r in hl.parse_box_score(4, box(4))}
    assert "Dani Olmo" not in rows                                  # 0 minutes
    assert rows["Joan García"]["position"] == "GK" and rows["Fermín López"]["position"] == "Sub"
    assert rows["Rodri"]["passes"] == 96 and rows["Rodri"]["passes_accurate"] == 84 and rows["Rodri"]["rating"] == 6.6
    assert rows["Lamine Yamal"]["dribbles_completed"] == 4 and rows["Lamine Yamal"]["offsides"] == 1
    assert pd.isna(rows["Joan García"]["xg"])                     # xG absent for the keeper
    # documented list form is accepted too
    b = box(4); b[0]["players"][1]["statistics"] = [b[0]["players"][1]["statistics"]]
    assert hl.parse_box_score(4, b)[1]["passes"] == 96


def test_fetch_all_competitions_then_compare(server, tmp_path):
    url, state = server
    out = tmp_path / "data" / "raw" / "highlightly_2026_barcelona"
    info = hl.fetch("Barcelona", 2026, out, client(url, tmp_path), with_lineups=True, progress=lambda *_: None)
    assert info["complete"] and info["matches_saved"] == 4 and info["has_xg"]
    matches, shots, pm = store.load_all(tmp_path / "data", sources=["highlightly"])
    assert set(matches["competition"]) == {"La Liga", "UEFA Champions League"}
    r = metrics.compare(matches, shots, pm, "Barcelona", "Real Madrid")           # same competition: 2 earlier La Liga games
    assert r["meta"]["n_base"] == 2 and r["meta"]["has_xg"] and not r["meta"]["has_shots"]
    found = insights.findings(r)
    assert found[0].startswith("Barcelona lost 1-2: they scored 1 from 1.19 xG")
    assert any(s.startswith("Rodri (midfielder):") and "passes attempted per 90 rose from an average of 68.0 before to 96.0" in s
               for s in found)
    assert any("regulars averaged a rating" in s for s in found)
    assert {"rating", "passes_accurate", "dribbles_completed", "duels_won"} <= set(r["players"]["metric"])
    assert charts.available_lineup_metrics(r)[0] == "rating"
    pos = charts.lineup_positions(r["lineup"]).set_index("player")
    assert pos.loc["Joan García", "x"] < pos.loc["Gerard Martín", "x"] < pos.loc["Rodri", "x"] < pos.loc["Lamine Yamal", "x"]
    assert charts.lineup_chart(r, tmp_path / "xi.png", "rating").stat().st_size > 5000
    r2 = metrics.compare(matches, shots, pm, "Barcelona", "Paris", same_competition=False)   # the PSG game vs all earlier
    assert r2["meta"]["n_base"] == 2 and r2["meta"]["competition"] == "UEFA Champions League"


def test_league_filter_and_quota_stop(server, tmp_path):
    url, state = server
    out = tmp_path / "data" / "raw" / "hl"
    info = hl.fetch("Barcelona", 2026, out, client(url, tmp_path), league="La Liga", progress=lambda *_: None)
    assert info["matches_saved"] == 3                               # the Champions League game is excluded
    state.quota_left = 3                                            # list calls are cached; 3 requests = 1.5 matches
    out2 = tmp_path / "data" / "raw" / "hl2"
    c = hl.Client(key="test", cache_dir=tmp_path / "cache2", base=url, min_interval=0)
    info = hl.fetch("Barcelona", 2026, out2, c, progress=lambda *_: None)
    assert not info["complete"] and "quota" in info["stopped"].lower() and info["matches_saved"] >= 0
    assert len(pd.read_csv(out2 / "matches.csv")) == info["matches_saved"]



def test_lineup_summary_skips_positions_and_explains(server, tmp_path):
    url, state = server
    out = tmp_path / "data" / "raw" / "hl"
    hl.fetch("Barcelona", 2026, out, client(url, tmp_path), progress=lambda *_: None)
    matches, shots, pm = store.load_all(tmp_path / "data", sources=["highlightly"])
    r = metrics.compare(matches, shots, pm, "Barcelona", "Real Madrid")
    tackles = charts.lineup_summary(r, "tackles")
    assert any("Not judged on tackles because of their position: Lamine Yamal" in s for s in tackles)
    assert not any("Lamine Yamal (from" in s for s in tackles)
    rating = charts.lineup_summary(r, "rating")
    assert any("Lamine Yamal (from an average of 7.90 before to 6.25)" in s for s in rating)



def test_fetching_both_teams_does_not_double_count_the_shared_match(server, tmp_path):
    url, state = server
    data = tmp_path / "data"
    hl.fetch("Barcelona", 2026, data / "raw" / "highlightly_2026_barcelona", client(url, tmp_path), progress=lambda *_: None)
    single = metrics.compare(*store.load_all(data, sources=["highlightly"]), "Barcelona", "Real Madrid")
    import shutil
    shutil.copytree(data / "raw" / "highlightly_2026_barcelona", data / "raw" / "highlightly_2026_realmadrid")  # same match saved twice
    double = metrics.compare(*store.load_all(data, sources=["highlightly"]), "Barcelona", "Real Madrid")
    assert double["team_player"].loc["passes", "match"] == single["team_player"].loc["passes", "match"]
    assert len(double["matches"]) == len(single["matches"])



def test_bad_key_and_unreachable_server_give_clear_errors(server, tmp_path):
    url, state = server
    c = hl.Client(key="test", cache_dir=tmp_path / "c", base="http://127.0.0.1:9", min_interval=0)   # nothing listens on port 9
    with pytest.raises(hl.ApiError, match="Could not reach Highlightly"):
        c.get("/matches", {"homeTeamName": "Barcelona", "season": 2026})
    import json as _j
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading as _t
    class Deny(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_response(403); self.end_headers(); self.wfile.write(b'{"message":"invalid key"}')
    d = ThreadingHTTPServer(("127.0.0.1", 0), Deny); _t.Thread(target=d.serve_forever, daemon=True).start()
    c2 = hl.Client(key="bad", cache_dir=tmp_path / "c2", base=f"http://127.0.0.1:{d.server_address[1]}", min_interval=0)
    with pytest.raises(hl.ApiError, match="rejected the API key"):
        c2.get("/matches", {"homeTeamName": "Barcelona", "season": 2026})
    d.shutdown()


def test_box_score_keeps_goals_and_assists():
    resp = [{"team": {"id": 1, "name": "Barcelona"}, "players": [
        {"id": 7, "name": "Raphinha", "position": "Forward", "isSubstitute": False, "minutesPlayed": 90, "matchRating": "7.4",
         "offsides": 0, "statistics": {"goalsScored": 2, "assists": 1, "shotsTotal": 4, "shotsOnTarget": 3}}]}]
    row = hl.parse_box_score(99, resp)[0]
    assert row["goals"] == 2 and row["assists"] == 1 and row["shots_on_target"] == 3
