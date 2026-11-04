"""Manual adapter: team-level match stats entered by hand (e.g. from UEFA.com), with a source URL
per row. Used for competitions the automated sources do not cover."""
from pathlib import Path

import pandas as pd

from matchreport import schema

REQUIRED = ["date", "competition", "home_team", "away_team", "home_goals", "away_goals",
            "home_xg", "away_xg", "source_url"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df["game_id"] = ["manual-" + str(i) for i in range(len(df))]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["source"] = "manual:" + df["source_url"].astype(str)
    return schema.conform(df, schema.MATCHES)
