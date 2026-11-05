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



def test_a_chosen_period_can_reach_into_last_season():
    import datetime as dt
    dates = ["2026-01-17", "2026-01-25", "2026-02-07", "2026-03-01", "2026-09-20", "2026-10-04"]
    matches = pd.DataFrame({"game_id": range(1, 7), "date": pd.to_datetime(dates), "competition": "L",
                            "home_team": ["United"] * 6, "away_team": [f"Opp{i}" for i in range(1, 7)],
                            "home_goals": [2, 3, 1, 2, 0, 0], "away_goals": [0, 2, 0, 1, 2, 1],
                            "home_xg": [2.0, 1.5, 1.8, 1.9, 0.6, 0.8], "away_xg": [0.5, 1.0, 0.7, 0.9, 1.8, 1.4], "source": "highlightly"})
    shots = pd.DataFrame(columns=["game_id", "team", "player", "player_id", "minute", "x", "y", "xg", "result", "situation", "source"])
    pm = pd.DataFrame(columns=["game_id", "team", "player", "player_id", "position", "minutes", "shots", "source"])
    r = metrics.compare(matches, shots, pm, "United", "Opp6", baseline="range", period=("2026-01-17", "2026-03-01"))
    assert r["meta"]["n_base"] == 4 and r["meta"]["period"] == ["2026-01-17", "2026-03-01"]
    assert r["meta"]["baseline_label"] == "the 4 same-competition matches between 17 January 2026 and 1 March 2026"
    assert abs(r["team_table"].loc["xg_for", "base_avg"] - 1.8) < 1e-9
    with pytest.raises(ValueError, match="Choose a start and an end date"):
        metrics.compare(matches, shots, pm, "United", "Opp6", baseline="range")
    with pytest.raises(ValueError, match="between 1 June 2025 and 30 June 2025"):
        metrics.compare(matches, shots, pm, "United", "Opp6", baseline="range", period=("2025-06-01", "2025-06-30"))


# ---------------------------------------------------------------- period against period
def two_season_data():
    dates = ["2026-01-17", "2026-01-25", "2026-02-07", "2026-03-01", "2026-08-22", "2026-09-13", "2026-10-25", "2026-10-31"]
    gf, ga = [2, 3, 1, 2, 0, 0, 0, 0], [0, 2, 0, 1, 2, 1, 3, 2]
    xf, xa = [2.0, 1.5, 1.8, 1.9, 1.83, 1.0, 1.18, 1.17], [0.5, 1.0, 0.7, 0.9, 1.01, 1.10, 2.18, 2.67]
    matches = pd.DataFrame({"game_id": range(1, 9), "date": pd.to_datetime(dates), "competition": "Premier League",
                            "home_team": ["United"] * 8, "away_team": [f"Opp{i}" for i in range(1, 9)],
                            "home_goals": gf, "away_goals": ga, "home_xg": xf, "away_xg": xa, "source": "highlightly"})
    rows = []
    for gid in range(1, 9):
        last = gid <= 4
        for name, pos, role, kp, passes in (("Bruno", "MC", "M", 4.8 if last else 1.0, 60), ("Maguire", "DC", "D", 0.2, 40 if last else 75),
                                             ("Keeper", "GK", "G", 0, 30)):
            rows.append(dict(game_id=gid, team="United", player=name, player_id=name, position=pos, role=role, minutes=90,
                             key_passes=kp, passes=passes, passes_accurate=passes * 0.85, tackles=2, rating=7.2 if last else 6.2,
                             source="highlightly"))
    pm = pd.DataFrame(rows)
    shots = pd.DataFrame(columns=["game_id", "team", "player", "player_id", "minute", "x", "y", "xg", "result", "situation", "source"])
    return matches, shots, pm


def test_period_against_period():
    matches, shots, pm = two_season_data()
    r = metrics.compare_periods(matches, shots, pm, "United", [5, 6, 7, 8], [1, 2, 3, 4], "This season's defeats", "Carrick's spell")
    meta, t = r["meta"], r["team_table"]
    assert meta["n_focus"] == 4 and meta["n_base"] == 4 and meta["focus_record"] == "0-0-4" and meta["base_record"] == "4-0-0"
    assert abs(t.loc["xg_against", "match"] - (1.01 + 1.10 + 2.18 + 2.67) / 4) < 1e-9
    assert t.loc["points_per_game", "base_avg"] == 3 and t.loc["points_per_game", "match"] == 0
    found = insights.findings(r)
    assert found[0].startswith("This season's defeats: 4 matches") and "Carrick's spell: 4 matches" in found[0]
    text = " ".join(found)
    assert "in Carrick's spell to" in text and " before to " not in text and "this match" not in text
    assert any(s.startswith("Bruno (midfielder):") and "key passes per 90 fell from an average of 4.8 in Carrick's spell to 1.0" in s for s in found)
    assert any(s.startswith("Maguire (defender):") and "passes attempted per 90 rose" in s for s in found)
    assert r["lineup"].empty


def test_a_match_picked_in_both_periods_counts_only_in_the_first():
    matches, shots, pm = two_season_data()
    r = metrics.compare_periods(matches, shots, pm, "United", [4, 5], [1, 2, 3, 4], "A", "B")
    assert r["meta"]["n_focus"] == 2 and r["meta"]["n_base"] == 3
    with pytest.raises(ValueError, match="Pick at least one match"):
        metrics.compare_periods(matches, shots, pm, "United", [1], [1], "A", "B")


def test_period_report_and_charts(tmp_path):
    from matchreport import report
    matches, shots, pm = two_season_data()
    r = metrics.compare_periods(matches, shots, pm, "United", [5, 6, 7, 8], [1, 2, 3, 4], "Defeats", "Carrick")
    out = report.build_from_result(r, tmp_path, llm=False)
    md = (out / "report.md").read_text()
    assert md.startswith("# United: Defeats vs Carrick") and md.count("## Findings") == 1
    assert "2026-10-31 vs Opp8 (H) L 0-2" in md
    for f in ("xg_bars.png", "team_changes.png", "player_changes.png"):
        assert (out / f).stat().st_size > 5000



def test_written_summary_works_for_a_period_comparison(monkeypatch):
    """Regression: the number checker assumed one match with a score and a date, and crashed on a period comparison."""
    matches, shots, pm = two_season_data()
    r = metrics.compare_periods(matches, shots, pm, "United", [5, 6, 7, 8], [1, 2, 3, 4], "This season's defeats", "Carrick's spell")
    good = ("United created 1.29 xG per match in This season's defeats against 1.80 in Carrick's spell, "
            "across 4 matches in each period.")
    monkeypatch.setattr(narrative, "BACKEND", "stub-model")
    monkeypatch.setattr(narrative, "_call", lambda prompt: good)
    out = narrative.write(r)
    assert out["check"]["passed"], out["check"]
    monkeypatch.setattr(narrative, "_call", lambda prompt: "United created 9.99 xG per match.")
    assert narrative.write(r)["check"]["unverified"] == ["9.99"]
    assert "Carrick's spell" in out["prompt"] and "two periods" in out["prompt"]


def test_period_stat_picker_shows_every_qualifying_player(tmp_path):
    from matchreport import charts
    matches, shots, pm = two_season_data()
    pm["goals"] = [0.0] * len(pm)
    pm.loc[(pm["player"] == "Bruno") & (pm["game_id"] <= 4), "goals"] = 1.0       # scored every game before, none since
    r = metrics.compare_periods(matches, shots, pm, "United", [5, 6, 7, 8], [1, 2, 3, 4], "This season", "Carrick's first season")
    assert "goals" in charts.available_lineup_metrics(r)
    t = charts.period_metric_table(r, "goals").set_index("player")
    assert t.loc["Bruno", "base"] == 1.0 and t.loc["Bruno", "match"] == 0.0 and "Keeper" not in t.index
    lines = charts.period_metric_summary(r, "goals")
    assert any(s.startswith("Biggest drops: Bruno (1.00 in Carrick's first season to 0.00 in This season)") for s in lines)
    assert charts.period_player_chart(r, tmp_path / "g.png", "goals").stat().st_size > 5000
    assert any("goals per 90 fell" in s for s in insights.findings(r))



def test_period_stat_table_lists_every_player():
    from matchreport import charts
    matches, shots, pm = two_season_data()
    r = metrics.compare_periods(matches, shots, pm, "United", [5, 6, 7, 8], [1, 2, 3, 4], "2026/27", "2025/26")
    rows = charts.period_metric_rows(r, "passes")
    assert list(rows["Player"]) == ["Bruno", "Maguire"]          # both outfield players, biggest drop first
    assert list(rows.columns) == ["Player", "2025/26", "2026/27", "Change", "Change %", "Minutes, 2025/26", "Minutes, 2026/27"]
    m = rows.set_index("Player").loc["Maguire"]
    assert m["2025/26"] == 40.0 and m["2026/27"] == 75.0 and m["Change"] == 35.0 and m["Change %"] == 88



def test_match_stat_table_lists_every_player_who_played():
    from matchreport import charts
    r = metrics.compare(*small_baseline_data(minutes_before=60), team="Atletico", opponent="Athletic")
    rows = charts.match_metric_rows(r, "shots")
    assert set(rows["Player"]) == set(r["players"].loc[r["players"]["metric"] == "shots", "player"])
    assert list(rows.columns) == ["Player", "Started", "Usual", "This match", "Change", "Change %",
                                  "Minutes earlier", "Minutes this match", "Note"]
    assert rows["Change"].is_monotonic_increasing or rows["Change"].isna().any()


def test_share_ready_charts_render_for_matches_and_periods(tmp_path):
    from matchreport import story_charts
    matches, shots, pm = two_season_data()
    pr = metrics.compare_periods(matches, shots, pm, "United", [5, 6, 7, 8], [1, 2, 3, 4], "2026/27", "Carrick's first season")
    mr = metrics.compare(matches, shots, pm, "United", "Opp8", baseline="range", period=("2026-01-17", "2026-03-01"))
    for r, tag in ((pr, "p"), (mr, "m")):
        for fn in (story_charts.chances_and_goals, story_charts.defence, story_charts.finishing):
            assert fn(r, tmp_path / f"{tag}_{fn.__name__}.png").stat().st_size > 20000
        assert story_charts.player_stat(r, tmp_path / f"{tag}_players.png", "rating").stat().st_size > 20000
    assert story_charts.same_opponents(mr, tmp_path / "x.png") is None            # single match: not applicable
    assert story_charts.same_opponents(pr, tmp_path / "y.png") is None            # no opponent met in both periods here
