"""Load the canonical tables from every source directory under data/raw, plus manual CSVs."""
from pathlib import Path

import pandas as pd

from matchreport import schema
from matchreport.sources import manual


def _token(df: pd.DataFrame) -> pd.Series:
    return df["source"].astype(str).str.split(":").str[0]


def available_sources(data_dir: Path) -> list:
    """Source names found on disk, e.g. ['apifootball', 'understat']. Manual rows are not a separate choice."""
    found = set()
    for f in (data_dir / "raw").glob("*/matches.csv"):
        found |= set(_token(pd.read_csv(f, usecols=["source"])))
    return sorted(found)


def load_all(data_dir: Path, sources=None):
    """Load every source, or only those named in `sources` (manual rows are always kept)."""
    raw = data_dir / "raw"
    matches, shots, pm = [], [], []
    for d in sorted(raw.glob("*/")):
        if (d / "matches.csv").exists():
            matches.append(pd.read_csv(d / "matches.csv", parse_dates=["date"]))
        if (d / "shots.csv").exists():
            shots.append(pd.read_csv(d / "shots.csv"))
        if (d / "player_match.csv").exists():
            pm.append(pd.read_csv(d / "player_match.csv"))
    for f in sorted((data_dir / "manual").glob("*.csv")):
        m = manual.load(f)
        if m.empty:
            continue
        m["date"] = pd.to_datetime(m["date"])
        matches.append(m)
    if not matches:
        raise FileNotFoundError(f"No data under {data_dir}. Run the fetch step first.")
    cat = lambda parts, cols: (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols))
    matches, shots, pm = (cat(matches, schema.MATCHES), cat(shots, schema.SHOTS), cat(pm, schema.PLAYER_MATCH))
    # The same match can be saved more than once (fetching both teams of a match writes it into each team's folder);
    # keep one copy per source, or team totals for that match get counted twice.
    if not matches.empty:
        matches = matches.drop_duplicates(subset=["source", "game_id"], keep="last")
    if not pm.empty:
        pm = pm.drop_duplicates(subset=["source", "game_id", "team", "player"], keep="last")
    if not shots.empty:
        shots = shots.drop_duplicates()
    if sources is not None:
        keep = set(sources) | {"manual"}
        matches, shots, pm = (df[_token(df).isin(keep)].reset_index(drop=True) for df in (matches, shots, pm))
    for df, cols in ((matches, ["home_goals", "away_goals", "home_xg", "away_xg"]),
                     (shots, ["minute", "x", "y", "xg"]),
                     (pm, schema.PLAYER_NUMERIC)):
        for c in cols:
            if c not in df.columns:
                df[c] = float("nan")
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("role", "grid"):
        if c not in pm.columns:
            pm[c] = None
    if matches.empty:
        raise FileNotFoundError(f"No matches from sources {sorted(sources)} under {data_dir}.")
    return matches.sort_values("date").reset_index(drop=True), shots, pm
