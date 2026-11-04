"""Run with: python -m pytest -q"""
import pandas as pd
import pytest

from matchreport import insights, metrics, narrative


def make_data():
    matches = pd.DataFrame({
        "game_id": [1, 2, 3, 4], "date": pd.to_datetime(["2026-08-20", "2026-09-01", "2026-10-25", "2026-11-01"]),
        "competition": "L", "home_team": ["A", "Barcelona", "Barcelona", "C"],
        "away_team": ["Barcelona", "B", "Real Madrid", "Barcelona"],
        "home_goals": [0, 3, 1, 0], "away_goals": [4, 0, 2, 2], "home_xg": [1.0, 3.0, 1.6, 0.8], "away_xg": [4.0, 0.5, 2.0, 2.5],
        "source": "test"})
    shots = pd.DataFrame({
        "game_id": [1, 1, 2, 3, 3, 4], "team": ["Barcelona", "A", "Barcelona", "Barcelona", "Real Madrid", "Barcelona"],
        "player": ["P", "Q", "P", "P", "R", "P"], "player_id": [1, 2, 1, 1, 3, 1], "minute": [10, 20, 30, 40, 50, 60],
        "x": [0.9, 0.8, 0.95, 0.7, 0.9, 0.9], "y": [0.5] * 6, "xg": [0.5, 0.1, 0.6, 0.05, 0.4, 0.5],
        "result": ["Goal", "MissedShots", "SavedShot", "BlockedShot", "Goal", "Goal"], "situation": "OpenPlay", "source": "test"})
    pm = pd.DataFrame({
        "game_id": [1, 2, 3, 4], "team": "Barcelona", "player": "P", "player_id": 1, "position": "FW",
        "minutes": [90, 90, 90, 90], "shots": [1, 1, 1, 1], "shots_on_target": [1, 1, 0, 1], "key_passes": [2, 2, 1, 2],
        "xg": [0.5, 0.6, 0.05, 0.5], "xa": [0.3, 0.3, 0.1, 0.3], "xg_chain": [1.0, 1.0, 0.2, 1.0],
        "xg_buildup": [0.5, 0.5, 0.1, 0.5], "source": "test"})
    return matches, shots, pm


def test_baseline_before():
    r = metrics.compare(*make_data(), team="Barcelona", opponent="Real", baseline="before")
    t = r["team_table"]
    assert r["meta"]["n_base"] == 2
    assert t.loc["xg_for", "base_avg"] == 3.5 and t.loc["xg_for", "match"] == 1.6
    assert r["meta"]["result"] == "L" and r["meta"]["score"] == "1-2"
    assert r["shot_quality"].loc["box_share", "for_match"] == 0.0


def test_baseline_all_and_after():
    r_all = metrics.compare(*make_data(), team="Barcelona", opponent="Real", baseline="all")
    r_after = metrics.compare(*make_data(), team="Barcelona", opponent="Real", baseline="after")
    assert r_all["meta"]["n_base"] == 3 and r_after["meta"]["n_base"] == 1
    assert r_after["team_table"].loc["xg_for", "base_avg"] == 2.5


def test_rankings_sorted_and_filtered():
    r = metrics.compare(*make_data(), team="Barcelona", opponent="Real", baseline="before")
    tc = r["team_changes"]
    assert tc["change_pct"].is_monotonic_increasing
    assert "xg_per_shot_for" in tc.index
    pc = r["player_changes"]
    assert (pc["minutes_base"] >= r["meta"]["min_base_minutes"]).all() or pc.empty


def test_validator_flags_invented_numbers():
    r = metrics.compare(*make_data(), team="Barcelona", opponent="Real")
    found = insights.findings(r)
    assert narrative.validate("Barcelona created 1.60 xG against a baseline of 3.50, and lost 1-2.", r, found)["passed"]
    assert narrative.validate("Barcelona created 7.77 xG.", r, found)["unverified"] == ["7.77"]


def test_findings_use_real_numbers():
    r = metrics.compare(*make_data(), team="Barcelona", opponent="Real")
    found = insights.findings(r)
    assert any("created 1.60 xG" in s for s in found)
    assert any("out-created" in s for s in found)


def test_tiny_baseline_minutes_are_ignored():
    rows = []
    for name, mins_b in [("Regular", 810), ("Rookie", 30)]:
        values = {k: (1.0, 1.0) for k in metrics.PLAYER_METRICS}
        values.update({"key_passes": (3.0, 0.0), "xa": (0.6, 0.0), "shots": (2.0, 2.0)})
        for metric, (b, m) in values.items():
            rows.append({"player": name, "metric": metric, "base": b, "match": m,
                         "minutes_base": mins_b, "minutes_match": 90, "change_pct": (m / b - 1) * 100})
    result = {"players": pd.DataFrame(rows), "goalkeepers": set(), "meta": {}}
    found = " ".join(insights.player_findings(result))
    assert "Regular" in found and "Rookie" not in found


def test_bad_baseline_rejected():
    with pytest.raises(ValueError):
        metrics.compare(*make_data(), team="Barcelona", opponent="Real", baseline="yesterday")


def small_baseline_data(minutes_before=80):
    """Four matches: three baseline matches where players play 80 minutes each (240 total, under 270)."""
    matches = pd.DataFrame({
        "game_id": [1, 2, 3, 4], "date": pd.to_datetime(["2026-08-20", "2026-08-27", "2026-09-01", "2026-09-05"]),
        "competition": "L", "home_team": ["Atletico", "B", "Atletico", "Atletico"], "away_team": ["A", "Atletico", "C", "Athletic"],
        "home_goals": [2, 0, 1, 0], "away_goals": [0, 2, 0, 3], "home_xg": [1.5, 0.8, 1.2, 0.75], "away_xg": [0.5, 1.4, 0.6, 1.87],
        "source": "test"})
    shots = pd.DataFrame({"game_id": [1, 2, 3, 4], "team": "Atletico", "player": "P", "player_id": 1, "minute": 10,
                          "x": 0.9, "y": 0.5, "xg": [0.3, 0.3, 0.3, 0.05], "result": "MissedShots", "situation": "OpenPlay", "source": "test"})
    rows = []
    for gid, mins in zip([1, 2, 3, 4], [minutes_before] * 3 + [90]):
        for name, kp, xa in [("P1", 3 if gid < 4 else 0, 0.5 if gid < 4 else 0.0), ("P2", 2, 0.3), ("P3", 2, 0.3)]:
            rows.append(dict(game_id=gid, team="Atletico", player=name, player_id=name, position="FW", minutes=mins,
                             shots=2, shots_on_target=1, key_passes=kp, xg=0.5, xa=xa, xg_chain=1.0, xg_buildup=1.0, source="test"))
    return matches, shots, pd.DataFrame(rows)


def test_early_season_match_still_gets_player_findings():
    r = metrics.compare(*small_baseline_data(), team="Atletico", opponent="Athletic")
    assert r["meta"]["n_base"] == 3 and r["meta"]["min_base_minutes"] == 90
    assert not r["player_changes"].empty
    found = " ".join(insights.findings(r))
    assert "P1" in found and "only 3 earlier matches" in found


def test_no_player_findings_explains_why():
    r = metrics.compare(*small_baseline_data(minutes_before=20), team="Atletico", opponent="Athletic")
    found = insights.findings(r)
    assert r["player_changes"].empty
    assert any(s.startswith("No player-level findings") and "90" in s for s in found)


def test_validator_flags_unsupported_position_claims():
    r = metrics.compare(*small_baseline_data(minutes_before=20), team="Atletico", opponent="Athletic")
    found = insights.findings(r)
    bad = "Atletico's forwards struggled while the midfield and defence allowed more chances."
    chk = narrative.validate(bad, r, found)
    assert not chk["passed"] and {"forwards", "midfield", "defence"} <= set(chk["unsupported_terms"])
    assert narrative.validate("Atletico created 0.75 xG.", r, found)["passed"]


def lineup_result():
    matches, shots, pm = small_baseline_data()
    pm = pm.copy()
    pm["position"] = pm["player"].map({"P1": "FWR", "P2": "FWL", "P3": "FW"})
    return metrics.compare(matches, shots, pm, team="Atletico", opponent="Athletic")


def test_lineup_positions_left_right_and_bounds():
    from matchreport import charts
    lineup = pd.DataFrame({"player": ["A", "B", "C", "D"], "position": ["DR", "DL", "DC", "DC"], "minutes": 90})
    pos = charts.lineup_positions(lineup).set_index("player")
    assert pos.loc["A", "y"] < pos.loc["B", "y"]          # right side has the smaller y
    assert pos.loc["C", "y"] != pos.loc["D", "y"]         # two centre-backs are spread apart
    assert pos[["x", "y"]].min().min() >= 0 and pos[["x", "y"]].max().max() <= 100


def test_lineup_chart_renders_every_metric(tmp_path):
    from matchreport import charts
    r = lineup_result()
    assert len(r["lineup"]) == 3
    for metric in charts.LINEUP_METRICS:
        out = charts.lineup_chart(r, tmp_path / f"{metric}.png", metric)
        assert out.exists() and out.stat().st_size > 5000


def test_lineup_chart_handles_missing_baseline_minutes(tmp_path):
    from matchreport import charts
    matches, shots, pm = small_baseline_data(minutes_before=20)
    pm["position"] = "FW"
    r = metrics.compare(matches, shots, pm, team="Atletico", opponent="Athletic")
    assert charts.lineup_chart(r, tmp_path / "x.png").exists()


def test_result_sentence_compares_score_with_chances():
    r = metrics.compare(*small_baseline_data(), team="Atletico", opponent="Athletic")
    first = insights.findings(r)[0]
    assert first.startswith("Atletico lost 0-3: they scored 0 from 0.75 xG and conceded 3 from 1.87 xG")
    assert "fell well short of the chances" in first



# ---------------------------------------------------------------- relevance and significance (user feedback)
def profile_result(rows, roles, min_base=162):
    """rows: (player, metric, base, match); everyone played 90 in the match and 270 before."""
    df = pd.DataFrame([{"player": p, "metric": k, "base": b, "match": mt, "minutes_base": 270, "minutes_match": 90,
                        "change_pct": (mt / b - 1) * 100 if b else float("nan")} for p, k, b, mt in rows])
    return {"players": df, "roles": roles, "goalkeepers": set(), "meta": {"min_base_minutes": min_base}}


def test_small_absolute_changes_are_not_findings_even_at_minus_100_percent():
    r = profile_result([("Cancelo", "tackles", 1.0, 0.0), ("Cancelo", "rating", 7.50, 5.51)], {"Cancelo": "D"})
    text = " ".join(insights.player_findings(r))
    assert "tackles" not in text
    assert "Cancelo (defender): rating fell from an average of 7.50 before to 5.51." in text


def test_attackers_are_never_judged_on_defending_and_centre_backs_never_on_shooting():
    r = profile_result([("Yamal", "tackles", 2.5, 0.0), ("Yamal", "rating", 8.85, 7.87),
                        ("Yamal", "key_passes", 4.0, 1.0), ("Yamal", "dribbles_completed", 3.2, 0.5),
                        ("Cubarsi", "shots_on_target", 1.5, 0.0), ("Cubarsi", "shots", 2.0, 0.0),
                        ("Cubarsi", "passes", 80.0, 60.0)], {"Yamal": "F", "Cubarsi": "D"})
    found = insights.player_findings(r)
    yamal = next(s for s in found if s.startswith("Yamal"))
    assert "tackles" not in yamal and "key passes per 90 fell from an average of 4.0 before to 1.0" in yamal
    assert "completed dribbles per 90 fell" in yamal and "rating fell from an average of 8.85 before to 7.87" in yamal
    cub = next(s for s in found if s.startswith("Cubarsi"))
    assert "shot" not in cub and "passes attempted per 90 fell from an average of 80.0 before to 60.0" in cub


def test_midfield_passing_drops_are_reported():
    r = profile_result([("Pedri", "passes_accurate", 70.0, 55.0), ("Rodri", "passes_accurate", 65.0, 48.0)],
                       {"Pedri": "M", "Rodri": "M"})
    text = " ".join(insights.player_findings(r))
    assert "Pedri (midfielder): completed passes per 90 fell from an average of 70.0 before to 55.0" in text
    assert "Rodri (midfielder): completed passes per 90 fell from an average of 65.0 before to 48.0" in text


def test_player_rankings_drop_irrelevant_and_tiny_changes():
    from matchreport import relevance
    assert not relevance.significant("tackles", 1.0, 0.0)
    assert relevance.significant("passes_accurate", 65.0, 48.0)
    assert not relevance.relevant("F", "tackles") and not relevance.relevant("D", "shots_on_target")
    roles = relevance.player_roles(pd.DataFrame({"player": ["A", "A", "B"], "position": ["Sub", "AMR", "DC"],
                                                 "role": [None, None, None]}))
    assert roles == {"A": "F", "B": "D"}



# ---------------------------------------------------------------- feedback round: ratings, wording, lineup without n/a
def test_a_rating_of_zero_is_no_rating():
    pm = pd.DataFrame({"player": ["A", "A", "B"], "period": ["base", "match", "base"], "minutes": [90, 90, 90],
                       "rating": [0.0, 6.75, 7.0]})
    rows = metrics._rating_rows(pm)
    assert ("A", "rating") in rows.index and pd.isna(rows.loc[("A", "rating"), "base"])     # A's zero is not an average
    assert rows.loc[("B", "rating"), "base"] == 7.0


def test_one_time_is_not_one_times():
    tp = pd.DataFrame({"base": [3.3], "match": [1.0], "change_pct": [-70.0]}, index=["offsides"])
    out = insights.team_player_findings({"team_player": tp, "meta": {"team": "Barcelona"}})
    assert out == ["Barcelona were caught offside 1 time versus 3.3 on average."]


def test_lineup_always_shows_this_match_and_never_n_a(tmp_path):
    from matchreport import charts
    r = metrics.compare(*small_baseline_data(minutes_before=20), team="Atletico", opponent="Athletic")   # 60 earlier minutes
    vals = charts.lineup_values(r, "shots").set_index("player")
    assert vals["match"].notna().all() and vals["base"].isna().all()            # this match kept, no usual value
    summary = " ".join(charts.lineup_summary(r, "shots"))
    assert "No usual value yet" in summary and "P1" in summary
    assert charts.lineup_chart(r, tmp_path / "x.png", "shots").stat().st_size > 5000
    assert "n/a" not in open(charts.__file__).read().split("def lineup_chart")[1].split("def lineup_summary")[0]
