"""The shared data model every source must produce.

Three tables. Pitch coordinates are 0-1, attacking left to right, with (1, 0.5) the centre of the goal
being attacked. A source fills the columns it has and leaves the rest empty.
"""
MATCHES = ["game_id", "date", "competition", "home_team", "away_team",
           "home_goals", "away_goals", "home_xg", "away_xg", "source"]

SHOTS = ["game_id", "team", "player", "player_id", "minute", "x", "y",
         "xg", "result", "situation", "source"]

# Numeric per-player-per-match columns (everything a source may provide)
PLAYER_NUMERIC = ["minutes", "shots", "shots_on_target", "key_passes", "xg", "xa", "xg_chain", "xg_buildup",
                  "passes", "passes_accurate", "tackles", "interceptions", "duels", "duels_won",
                  "dribbles_attempted", "dribbles_completed", "dribbled_past", "fouls_committed", "fouls_drawn",
                  "offsides", "rating"]
PLAYER_MATCH = ["game_id", "team", "player", "player_id", "position", "role", "grid"] + PLAYER_NUMERIC + ["source"]

ON_TARGET_RESULTS = {"Goal", "SavedShot"}


def conform(df, columns):
    """Return df with exactly these columns, adding missing ones as empty."""
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = None
    return out[columns]
