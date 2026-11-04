"""Compare one match to a baseline built from the team's other matches in the same season."""
import numpy as np
import pandas as pd

from matchreport import relevance

BASE_PLAYER_METRICS = ["shots", "shots_on_target", "key_passes", "xg", "xa", "xg_chain", "xg_buildup"]
EXTRA_PLAYER_METRICS = ["passes", "passes_accurate", "tackles", "interceptions", "duels_won", "dribbles_completed",
                        "dribbles_attempted", "dribbled_past"]
PLAYER_METRICS = BASE_PLAYER_METRICS + EXTRA_PLAYER_METRICS
BASELINES = {"before": "matches before it", "after": "matches after it", "all": "other matches",
             "range": "matches in the chosen period"}

# Player findings and rankings only consider players who meet these
MIN_MATCH_MINUTES = 60
MIN_BASE_MINUTES = 90    # a player needs at least one full match of earlier minutes to get a usual value
MIN_BASE_PER90 = 0.4


def min_base_minutes(n_base: int = 0) -> int:
    """Earlier minutes a player needs before his per-90 numbers and ratings are compared with this match.
    One full match (90 minutes) however long the baseline is; with very few earlier matches the findings carry
    a caution that per-90 changes are noisy."""
    return MIN_BASE_MINUTES


def season_of(d) -> int:
    """Start year of the season a date belongs to (August 2026 and March 2027 are both season 2026)."""
    d = pd.to_datetime(d)
    return int(d.year if d.month >= 7 else d.year - 1)


def team_matches(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    """One row per match from the team's point of view (for / against)."""
    m = matches[matches["home_team"].str.contains(team, regex=False) | matches["away_team"].str.contains(team, regex=False)].copy()
    home = m["home_team"].str.contains(team, regex=False)
    out = pd.DataFrame({
        "game_id": m["game_id"], "date": pd.to_datetime(m["date"]).dt.date, "competition": m["competition"],
        "opponent": m["away_team"].where(home, m["home_team"]), "venue": np.where(home, "H", "A"),
        "goals_for": m["home_goals"].where(home, m["away_goals"]).astype(float),
        "goals_against": m["away_goals"].where(home, m["home_goals"]).astype(float),
        "xg_for": m["home_xg"].where(home, m["away_xg"]).astype(float),
        "xg_against": m["away_xg"].where(home, m["home_xg"]).astype(float),
        "source": m["source"],
    }).sort_values("date").reset_index(drop=True)
    out["season"] = [season_of(d) for d in out["date"]]
    out["result"] = np.where(out.goals_for > out.goals_against, "W",
                             np.where(out.goals_for < out.goals_against, "L", "D"))
    return out


def find_match(tm: pd.DataFrame, opponent: str, date=None) -> pd.Series:
    hits = tm[tm["opponent"].str.contains(opponent, case=False, regex=False)]
    if date is not None:
        hits = hits[hits["date"] == pd.to_datetime(date).date()]
    if hits.empty:
        raise ValueError(f"No match found against '{opponent}'. Opponents: {sorted(tm['opponent'].unique())}")
    return hits.iloc[-1]


def shot_quality(shots: pd.DataFrame) -> dict:
    """Count, xG per shot, mean distance (m), share inside the box, shots on target."""
    if shots.empty:
        return {"shots": 0, "xg_per_shot": np.nan, "mean_distance_m": np.nan, "box_share": np.nan, "on_target": 0}
    dist = np.sqrt(((1 - shots["x"]) * 105) ** 2 + ((shots["y"] - 0.5) * 68) ** 2)
    in_box = (shots["x"] >= 0.843) & shots["y"].between(0.204, 0.796)
    return {"shots": len(shots), "xg_per_shot": shots["xg"].mean(), "mean_distance_m": dist.mean(),
            "box_share": in_box.mean(), "on_target": int(shots["result"].isin(["Goal", "SavedShot"]).sum())}


def _pct(base, match):
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((match / base - 1) * 100).replace([np.inf, -np.inf], np.nan)


def _rating_rows(pm: pd.DataFrame) -> pd.DataFrame:
    """Rating is an average, not a count: baseline = minutes-weighted mean of the player's ratings."""
    if "rating" not in pm.columns or pm["rating"].dropna().empty:
        return pd.DataFrame(columns=["base", "match"])
    r = pm.dropna(subset=["rating"]).copy()
    r = r[r["rating"] > 0]            # the provider uses 0 for "no rating": not a real score
    if r.empty:
        return pd.DataFrame(columns=["base", "match"])
    r["w"] = r["minutes"] * r["rating"]
    g = r.groupby(["player", "period"]).agg(w=("w", "sum"), m=("minutes", "sum"))
    avg = (g["w"] / g["m"]).unstack("period").reindex(columns=["base", "match"])
    avg["metric"] = "rating"
    return avg.reset_index().set_index(["player", "metric"])


def _team_player_table(pm_team: pd.DataFrame, base_ids, target_id, pm_opp: pd.DataFrame = None) -> pd.DataFrame:
    """Team totals built from player rows (passes, tackles, dribbles...), baseline average vs this match.
    pm_opp holds the opponents' player rows in the same matches; their offsides measure the team's defensive line."""
    empty = pd.DataFrame(columns=["base", "match", "change_pct"])
    cols = [c for c in ["passes", "passes_accurate", "tackles", "interceptions", "duels", "duels_won",
                        "dribbles_completed", "dribbled_past", "offsides"] if c in pm_team.columns]
    if not cols or pm_team[cols].dropna(how="all").empty:
        return empty
    sums = pm_team.groupby("game_id")[cols].sum(min_count=1)
    if {"passes", "passes_accurate"} <= set(cols):
        sums["pass_accuracy_pct"] = sums["passes_accurate"] / sums["passes"] * 100
    if {"duels", "duels_won"} <= set(cols):
        sums["duel_win_pct"] = sums["duels_won"] / sums["duels"] * 100
    if pm_opp is not None and "offsides" in pm_opp.columns and pm_opp["offsides"].notna().any():
        sums["opp_offsides"] = pm_opp.groupby("game_id")["offsides"].sum(min_count=1)
    keep = [c for c in ["passes", "pass_accuracy_pct", "tackles", "interceptions", "duel_win_pct", "dribbles_completed",
                        "dribbled_past", "offsides", "opp_offsides"] if c in sums.columns]
    ids = list(target_id) if isinstance(target_id, (list, tuple, set, pd.Series, np.ndarray)) else [target_id]
    focus_rows = sums.loc[sums.index.isin(ids), keep]          # one match, or several averaged per match
    if focus_rows.empty:
        return empty
    base_rows = sums.loc[sums.index.isin(list(base_ids)), keep]
    out = pd.DataFrame({"base": base_rows.mean(), "match": focus_rows.astype(float).mean()}).dropna()
    out["change_pct"] = _pct(out["base"], out["match"])
    return out


def _day(d) -> str:
    return f"{d.day} {d:%B %Y}"


def compare(matches, shots, player_match, team, opponent, date=None, same_competition=True, baseline="before",
            same_season=True, period=None):
    """Everything a report needs. baseline is 'before', 'after' or 'all' (other matches in the same season), or
    'range': every match between period[0] and period[1], in any season (for example a manager's spell last season)."""
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {list(BASELINES)}")
    tm = team_matches(matches, team)
    target = find_match(tm, opponent, date)
    others = tm[tm["game_id"] != target["game_id"]]
    start = end = None
    if baseline == "range":
        if not period or len(period) != 2 or None in period:
            raise ValueError("Choose a start and an end date for the period to compare with.")
        start, end = sorted(pd.to_datetime(d).date() for d in period)
        base = others[(others["date"] >= start) & (others["date"] <= end)]
    else:
        if same_season:
            others = others[others["season"] == target["season"]]
        if baseline == "before":
            base = others[others["date"] < target["date"]]
        elif baseline == "after":
            base = others[others["date"] > target["date"]]
        else:
            base = others
    if same_competition:
        base = base[base["competition"] == target["competition"]]
    if base.empty:
        where = f" between {_day(start)} and {_day(end)}" if start else ""
        comp = f" in the {target['competition']}" if same_competition else ""
        raise ValueError(f"No {BASELINES[baseline]}{comp}{where} to use as a baseline. "
                         f"Check that those matches have been fetched.")

    # Team level
    cols = ["goals_for", "goals_against", "xg_for", "xg_against"]
    team_table = pd.DataFrame({"base_avg": base[cols].mean(), "match": target[cols].astype(float)})
    team_table["change_pct"] = _pct(team_table["base_avg"], team_table["match"])
    has_xg = bool(team_table.loc[["xg_for", "xg_against"], ["base_avg", "match"]].notna().all().all())

    # Shot quality, for and against (needs shot-level data, e.g. Understat)
    s = shots[shots["game_id"].isin(tm["game_id"])]
    is_team = s["team"].str.contains(team, regex=False)
    mine_base, mine_match = s[s["game_id"].isin(base["game_id"]) & is_team], s[(s["game_id"] == target["game_id"]) & is_team]
    opp_base, opp_match = s[s["game_id"].isin(base["game_id"]) & ~is_team], s[(s["game_id"] == target["game_id"]) & ~is_team]
    n_base = int(base["game_id"].isin(s["game_id"]).sum())

    def per_match(d):
        d = dict(d)
        for k in ("shots", "on_target"):
            d[k] = d[k] / n_base if n_base else np.nan
        return d
    quality = pd.DataFrame({"for_base_avg": per_match(shot_quality(mine_base)), "for_match": shot_quality(mine_match),
                            "against_base_avg": per_match(shot_quality(opp_base)), "against_match": shot_quality(opp_match)})

    # Players
    pm_all = player_match[player_match["game_id"].isin(tm["game_id"]) & player_match["team"].str.contains(team, regex=False)].copy()
    lineup_cols = ["player", "position", "minutes"] + [c for c in ("role", "grid") if c in pm_all.columns]
    lineup = pm_all[(pm_all["game_id"] == target["game_id"]) & (pm_all["position"].astype(str) != "Sub")][lineup_cols].reset_index(drop=True)
    pm = pm_all.copy()
    pm["period"] = np.where(pm["game_id"] == target["game_id"], "match",
                            np.where(pm["game_id"].isin(base["game_id"]), "base", "other"))
    pm = pm[pm["period"] != "other"]
    metric_cols = [m for m in PLAYER_METRICS if m in pm.columns and pm[m].notna().any()]
    if metric_cols:
        tot = pm.groupby(["player", "period"])[metric_cols + ["minutes"]].sum()
        per90 = tot[metric_cols].div(tot["minutes"] / 90, axis=0)
        long = per90.stack().unstack("period").reindex(columns=["base", "match"])
        long.index.names = ["player", "metric"]
    else:
        long = pd.DataFrame(columns=["base", "match"])
    ratings = _rating_rows(pm)
    if not ratings.empty:
        long = pd.concat([long, ratings])
    if not long.empty:
        mins = pm.groupby(["player", "period"])["minutes"].sum().unstack("period").reindex(columns=["base", "match"])
        long = long.join(mins.rename(columns={"base": "minutes_base", "match": "minutes_match"}))
        long = long[long["minutes_match"].fillna(0) > 0]
        long["change_pct"] = _pct(long["base"], long["match"])
        players = long.reset_index().sort_values("change_pct", na_position="last").reset_index(drop=True)
    else:
        players = pd.DataFrame(columns=["player", "metric", "base", "match", "minutes_base", "minutes_match", "change_pct"])
    role = pm_all["role"] if "role" in pm_all.columns else pd.Series("", index=pm_all.index)
    goalkeepers = set(pm_all.loc[(pm_all["position"].astype(str) == "GK") | (role == "G"), "player"])

    pm_opp = player_match[player_match["game_id"].isin(tm["game_id"]) & ~player_match["team"].str.contains(team, regex=False)]
    team_player = _team_player_table(pm_all, base["game_id"], target["game_id"], pm_opp)

    # Rankings: biggest falls and rises
    parts = [team_table.rename(columns={"base_avg": "base"})]
    if not mine_match.empty:
        extra = {}
        for k in ("shots", "xg_per_shot", "box_share", "on_target", "mean_distance_m"):
            extra[f"{k}_for"] = (quality.loc[k, "for_base_avg"], quality.loc[k, "for_match"])
            extra[f"{k}_against"] = (quality.loc[k, "against_base_avg"], quality.loc[k, "against_match"])
        ex = pd.DataFrame(extra, index=["base", "match"]).T
        ex["change_pct"] = _pct(ex["base"], ex["match"])
        parts.append(ex)
    if not team_player.empty:
        parts.append(team_player)
    team_changes = pd.concat(parts).dropna(subset=["change_pct"]).sort_values("change_pct")

    roles = relevance.player_roles(pm_all)
    min_base = min_base_minutes(len(base))
    eligible = players[(players["minutes_match"] >= MIN_MATCH_MINUTES) & (players["minutes_base"] >= min_base)
                       & ~players["player"].isin(goalkeepers) & players["change_pct"].notna()].copy()
    if not eligible.empty:   # only changes that matter for the player's position and are big in absolute terms
        keep = [relevance.relevant(roles.get(r.player), r.metric) and relevance.significant(r.metric, r.base, r.match)
                for r in eligible.itertuples()]
        eligible = eligible[keep]
    player_changes = eligible.sort_values("change_pct").reset_index(drop=True)

    if baseline == "range":
        label = (f"the {len(base)} {'same-competition ' if same_competition else ''}matches between "
                 f"{_day(start)} and {_day(end)}")
    else:
        label = f"the {len(base)} {'same-competition ' if same_competition else ''}{BASELINES[baseline]}"
    meta = {"team": team, "opponent": target["opponent"], "date": str(target["date"]), "competition": target["competition"],
            "season": int(target["season"]), "venue": target["venue"],
            "score": f"{int(target['goals_for'])}-{int(target['goals_against'])}",
            "result": target["result"], "baseline": baseline, "baseline_label": label, "n_base": len(base),
            "period": [str(start), str(end)] if start else None,
            "n_base_with_shots": n_base, "has_shots": not mine_match.empty, "has_xg": has_xg,
            "has_players": not players.empty, "min_base_minutes": min_base,
            "sources": sorted(set(tm["source"].astype(str).str.split(":").str[0]))}
    return {"meta": meta, "team_table": team_table, "shot_quality": quality, "players": players,
            "team_player": team_player, "team_changes": team_changes, "player_changes": player_changes,
            "goalkeepers": goalkeepers, "lineup": lineup, "roles": roles,
            "matches": tm, "shots_base": mine_base, "shots_match": mine_match,
            "opp_shots_base": opp_base, "opp_shots_match": opp_match}



# ---------------------------------------------------------------------------------------------
# Period against period: a set of matches (A) compared with another set (B), averaged per match.
# The result has the same shape as compare(), with the "match" side holding period A's averages,
# so the same findings rules, charts and summary checks apply.
# ---------------------------------------------------------------------------------------------
def _record(df: pd.DataFrame) -> tuple:
    w, d, l = int((df["result"] == "W").sum()), int((df["result"] == "D").sum()), int((df["result"] == "L").sum())
    return f"{w}-{d}-{l}", (3 * w + d) / len(df)


def compare_periods(matches, shots, player_match, team, focus_ids, base_ids, focus_label="Period A",
                    base_label="Period B") -> dict:
    """Compare two hand-picked sets of a team's matches. focus_ids is period A (the one being judged), base_ids
    period B (the yardstick). A match picked in both counts only in A."""
    tm = team_matches(matches, team)
    focus = tm[tm["game_id"].isin(list(focus_ids))]
    base = tm[tm["game_id"].isin(list(base_ids)) & ~tm["game_id"].isin(list(focus_ids))]
    if focus.empty or base.empty:
        raise ValueError("Pick at least one match for each period (a match picked in both counts only in the first).")

    cols = ["goals_for", "goals_against", "xg_for", "xg_against"]
    f_rec, f_ppg = _record(focus)
    b_rec, b_ppg = _record(base)
    team_table = pd.DataFrame({"base_avg": base[cols].mean(), "match": focus[cols].mean()})
    team_table.loc["points_per_game"] = [b_ppg, f_ppg]
    team_table["change_pct"] = _pct(team_table["base_avg"], team_table["match"])
    has_xg = bool(team_table.loc[["xg_for", "xg_against"], ["base_avg", "match"]].notna().all().all())

    pm_all = player_match[player_match["game_id"].isin(tm["game_id"]) & player_match["team"].str.contains(team, regex=False)].copy()
    pm = pm_all.copy()
    pm["period"] = np.where(pm["game_id"].isin(focus["game_id"]), "match",
                            np.where(pm["game_id"].isin(base["game_id"]), "base", "other"))
    pm = pm[pm["period"] != "other"]
    metric_cols = [c for c in PLAYER_METRICS if c in pm.columns and pm[c].notna().any()]
    if metric_cols:
        tot = pm.groupby(["player", "period"])[metric_cols + ["minutes"]].sum()
        per90 = tot[metric_cols].div(tot["minutes"] / 90, axis=0)
        long = per90.stack().unstack("period").reindex(columns=["base", "match"])
        long.index.names = ["player", "metric"]
    else:
        long = pd.DataFrame(columns=["base", "match"])
    ratings = _rating_rows(pm)
    if not ratings.empty:
        long = pd.concat([long, ratings])
    if not long.empty:
        mins = pm.groupby(["player", "period"])["minutes"].sum().unstack("period").reindex(columns=["base", "match"])
        long = long.join(mins.rename(columns={"base": "minutes_base", "match": "minutes_match"}))
        long = long[long["minutes_match"].fillna(0) > 0]
        long["change_pct"] = _pct(long["base"], long["match"])
        players = long.reset_index().sort_values("change_pct", na_position="last").reset_index(drop=True)
    else:
        players = pd.DataFrame(columns=["player", "metric", "base", "match", "minutes_base", "minutes_match", "change_pct"])
    role = pm_all["role"] if "role" in pm_all.columns else pd.Series("", index=pm_all.index)
    goalkeepers = set(pm_all.loc[(pm_all["position"].astype(str) == "GK") | (role == "G"), "player"])
    pm_opp = player_match[player_match["game_id"].isin(tm["game_id"]) & ~player_match["team"].str.contains(team, regex=False)]
    team_player = _team_player_table(pm_all, base["game_id"], focus["game_id"].tolist(), pm_opp)

    parts = [team_table.rename(columns={"base_avg": "base"})]
    if not team_player.empty:
        parts.append(team_player)
    team_changes = pd.concat(parts).dropna(subset=["change_pct"]).sort_values("change_pct")

    roles = relevance.player_roles(pm_all)
    min_base = min_base_minutes(len(base))
    min_focus = MIN_BASE_MINUTES          # at least one full match of minutes in each period
    eligible = players[(players["minutes_match"] >= min_focus) & (players["minutes_base"] >= min_base)
                       & ~players["player"].isin(goalkeepers) & players["change_pct"].notna()].copy()
    if not eligible.empty:
        keep = [relevance.relevant(roles.get(r.player), r.metric) and relevance.significant(r.metric, r.base, r.match)
                for r in eligible.itertuples()]
        eligible = eligible[keep]
    player_changes = eligible.sort_values("change_pct").reset_index(drop=True)

    comps = sorted(set(focus["competition"]) | set(base["competition"]))
    span = lambda d: f"{_day(d['date'].min())} to {_day(d['date'].max())}"
    meta = {"mode": "periods", "team": team, "focus_label": focus_label, "base_label": base_label,
            "n_focus": len(focus), "n_base": len(base), "focus_record": f_rec, "base_record": b_rec,
            "focus_ppg": f_ppg, "base_ppg": b_ppg, "focus_span": span(focus), "base_span": span(base),
            "baseline_label": f"{base_label} ({len(base)} matches)", "competition": ", ".join(comps),
            "season": int(focus["season"].max()), "opponent": None, "date": None, "result": None, "score": None,
            "venue": None, "baseline": "periods", "period": None,
            "focus_ids": sorted(focus["game_id"].astype(str)), "base_ids": sorted(base["game_id"].astype(str)),
            "n_base_with_shots": 0, "has_shots": False, "has_xg": has_xg, "has_players": not players.empty,
            "min_base_minutes": min_base, "min_match_minutes": min_focus,
            "sources": sorted(set(tm["source"].astype(str).str.split(":").str[0]))}
    empty_shots = shots.iloc[0:0]
    return {"meta": meta, "team_table": team_table, "shot_quality": pd.DataFrame(), "players": players,
            "team_player": team_player, "team_changes": team_changes, "player_changes": player_changes,
            "goalkeepers": goalkeepers, "lineup": pd.DataFrame(columns=["player", "position", "minutes"]), "roles": roles,
            "matches": tm, "focus_matches": focus, "base_matches": base,
            "shots_base": empty_shots, "shots_match": empty_shots, "opp_shots_base": empty_shots, "opp_shots_match": empty_shots}
