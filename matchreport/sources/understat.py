"""Understat adapter: xG, shots, and player match stats for the top 5 European leagues."""
from pathlib import Path

import pandas as pd

from matchreport import schema


LEAGUES = ["ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1", "GER-Bundesliga", "ITA-Serie A"]


def list_teams(league: str, season: str) -> list:
    """Teams with at least one played match this season."""
    import soccerdata as sd
    sched = sd.Understat(leagues=league, seasons=season).read_schedule().reset_index()
    played = sched[sched["is_result"] == True]
    return sorted(set(played["home_team"]) | set(played["away_team"]))


def fetch(league: str, season: str, team: str, out_dir: Path) -> dict:
    """Download one team's played matches for a season and write the three canonical CSVs."""
    import soccerdata as sd  # imported here so the rest of the package works offline

    us = sd.Understat(leagues=league, seasons=season)
    sched = us.read_schedule().reset_index()
    mine = sched["home_team"].str.contains(team) | sched["away_team"].str.contains(team)
    sched = sched[mine & (sched["is_result"] == True)].sort_values("date")
    ids = sched["game_id"].astype(int).tolist()

    matches = pd.DataFrame({
        "game_id": sched["game_id"].astype(int), "date": pd.to_datetime(sched["date"]).dt.date,
        "competition": league, "home_team": sched["home_team"], "away_team": sched["away_team"],
        "home_goals": sched["home_goals"], "away_goals": sched["away_goals"],
        "home_xg": sched["home_xg"].astype(float), "away_xg": sched["away_xg"].astype(float),
        "source": "understat",
    })

    raw_shots = us.read_shot_events(match_id=ids).reset_index()
    shots = pd.DataFrame({
        "game_id": raw_shots["game_id"].astype(int), "team": raw_shots["team"],
        "player": raw_shots["player"], "player_id": raw_shots["player_id"],
        "minute": raw_shots["minute"], "x": raw_shots["location_x"].astype(float),
        "y": raw_shots["location_y"].astype(float), "xg": raw_shots["xg"].astype(float),
        "result": raw_shots["result"], "situation": raw_shots["situation"], "source": "understat",
    })

    raw_pm = us.read_player_match_stats(match_id=ids).reset_index()
    on_target = (shots[shots["result"].isin(schema.ON_TARGET_RESULTS)]
                 .groupby(["game_id", "player_id"]).size().rename("shots_on_target").reset_index())
    pm = raw_pm.merge(on_target, on=["game_id", "player_id"], how="left")
    player_match = pd.DataFrame({
        "game_id": pm["game_id"].astype(int), "team": pm["team"], "player": pm["player"],
        "player_id": pm["player_id"], "position": pm["position"], "minutes": pm["minutes"],
        "shots": pm["shots"], "shots_on_target": pm["shots_on_target"].fillna(0).astype(int),
        "key_passes": pm["key_passes"], "xg": pm["xg"].astype(float), "xa": pm["xa"].astype(float),
        "xg_chain": pm["xg_chain"].astype(float), "xg_buildup": pm["xg_buildup"].astype(float),
        "source": "understat",
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out_dir / "matches.csv", index=False)
    shots.to_csv(out_dir / "shots.csv", index=False)
    player_match.to_csv(out_dir / "player_match.csv", index=False)
    return {"matches": len(matches), "shots": len(shots), "player_match": len(player_match)}
