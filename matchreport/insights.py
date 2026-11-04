"""Deterministic findings: rules that turn the comparison tables into statements with numbers.
The language model only rewrites these as prose; it never decides what the story is.
Every rule tolerates missing data: a source without xG, shots, or player stats simply produces fewer findings."""
import numpy as np
import pandas as pd

from matchreport import relevance as rel
from matchreport.metrics import MIN_BASE_MINUTES, MIN_BASE_PER90, MIN_MATCH_MINUTES


def pct(a, b):
    return (b / a - 1) * 100 if (pd.notna(a) and a) else np.nan


def _g(row, key):
    """Safe lookup: NaN when the metric is not in this data source."""
    try:
        v = row[key]
    except KeyError:
        return np.nan
    return float(v) if pd.notna(v) else np.nan


def _ratio(a, b):
    return a / b if (pd.notna(a) and pd.notna(b) and b) else np.nan


# --------------------------------------------------------------------------------------------- team level
TEAM_PLAYER_TEXT = {
    "passes": ("attempted {m:.0f} passes versus {b:.0f} on average", "count"),
    "pass_accuracy_pct": ("had a pass accuracy of {m:.0f}% versus {b:.0f}% on average", "points"),
    "tackles": ("made {m:.0f} tackles versus {b:.1f} on average", "count"),
    "interceptions": ("made {m:.0f} interceptions versus {b:.1f} on average", "count"),
    "duel_win_pct": ("won {m:.0f}% of duels versus {b:.0f}% on average", "points"),
    "dribbles_completed": ("completed {m:.0f} dribbles versus {b:.1f} on average", "count"),
    "dribbled_past": ("were dribbled past {m:.0f} times versus {b:.1f} on average", "count"),
    "offsides": ("were caught offside {m:.0f} times versus {b:.1f} on average", "count"),
    "opp_offsides": ("caught their opponent offside {m:.0f} times versus {b:.1f} per match on average", "count"),
}


def team_player_findings(result: dict) -> list:
    tp, team = result.get("team_player"), result["meta"]["team"]
    if tp is None or tp.empty:
        return []
    cands = []
    for name, row in tp.iterrows():
        if name not in TEAM_PLAYER_TEXT:
            continue
        text, kind = TEAM_PLAYER_TEXT[name]
        size = abs(row["match"] - row["base"]) if kind == "points" else abs(row["change_pct"])
        threshold = 4 if kind == "points" else 15
        if pd.notna(size) and size >= threshold:
            cands.append((size / threshold, f"{team} " + text.format(m=row["match"], b=row["base"]) + "."))
    return [s for _, s in sorted(cands, reverse=True)[:4]]


def team_findings(result: dict) -> list:
    t, q, meta = result["team_table"], result["shot_quality"], result["meta"]
    team, opp, base = meta["team"], meta["opponent"], meta["baseline_label"]
    f = []
    if meta["has_xg"]:
        xf_b, xf_m = t.loc["xg_for", "base_avg"], t.loc["xg_for", "match"]
        xa_b, xa_m = t.loc["xg_against", "base_avg"], t.loc["xg_against", "match"]
        f.append(f"{team} created {xf_m:.2f} xG against {opp}, versus an average of {xf_b:.2f} across {base} "
                 f"({pct(xf_b, xf_m):+.0f}%).")
        f.append(f"{team} conceded {xa_m:.2f} xG, versus an average of {xa_b:.2f} ({pct(xa_b, xa_m):+.0f}%).")
        if xa_m > xf_m and meta["result"] == "L":
            f.append(f"{opp} out-created {team} on xG, so the result ({meta['result']} {meta['score']}) matched the balance "
                     f"of chances rather than bad luck.")
        elif xf_m > xa_m and meta["result"] == "L":
            f.append(f"{team} out-created {opp} on xG but still lost {meta['score']}, which points to finishing or "
                     f"goalkeeping rather than chance creation.")
        gf_m, ga_m = t.loc["goals_for", "match"], t.loc["goals_against", "match"]
        net = (gf_m - ga_m) - (xf_m - xa_m)
        verdict = "ran well ahead of" if net >= 1 else ("fell well short of" if net <= -1 else "was in line with")
        f.insert(0, f"{team} {'won' if meta['result'] == 'W' else 'lost' if meta['result'] == 'L' else 'drew'} "
                    f"{meta['score']}: they scored {gf_m:.0f} from {xf_m:.2f} xG and conceded {ga_m:.0f} from {xa_m:.2f} xG, "
                    f"so the result {verdict} the chances.")
    else:
        gf_b, gf_m = t.loc["goals_for", "base_avg"], t.loc["goals_for", "match"]
        ga_b, ga_m = t.loc["goals_against", "base_avg"], t.loc["goals_against", "match"]
        f.append(f"{team} scored {gf_m:.0f} against {opp} ({meta['result']} {meta['score']}), versus an average of "
                 f"{gf_b:.2f} across {base}.")
        f.append(f"{team} conceded {ga_m:.0f}, versus an average of {ga_b:.2f}. This data source has no xG.")
    f += team_player_findings(result)
    if not meta["has_shots"]:
        return f
    sh_b, sh_m = q.loc["shots", "for_base_avg"], q.loc["shots", "for_match"]
    qs_b, qs_m = q.loc["xg_per_shot", "for_base_avg"], q.loc["xg_per_shot", "for_match"]
    d_b, d_m = q.loc["mean_distance_m", "for_base_avg"], q.loc["mean_distance_m", "for_match"]
    box_b, box_m = q.loc["box_share", "for_base_avg"], q.loc["box_share", "for_match"]
    ot_b, ot_m = q.loc["on_target", "for_base_avg"], q.loc["on_target", "for_match"]
    shots_change, quality_change = pct(sh_b, sh_m), pct(qs_b, qs_m)
    if shots_change > -25 and quality_change < -35:
        f.append(f"Shot volume held up ({sh_b:.1f} shots per match in the baseline, {sh_m:.0f} in this match) but shot "
                 f"quality collapsed: xG per shot fell from {qs_b:.2f} to {qs_m:.2f} ({quality_change:+.0f}%). "
                 f"That is the signature of a team pushed into low-value shots.")
    elif quality_change < -35:
        f.append(f"Shot quality collapsed: xG per shot fell from {qs_b:.2f} to {qs_m:.2f} ({quality_change:+.0f}%), "
                 f"with shots down from {sh_b:.1f} per match to {sh_m:.0f}.")
    elif shots_change < -35:
        f.append(f"{team} took far fewer shots: {sh_m:.0f} versus {sh_b:.1f} per match ({shots_change:+.0f}%).")
    elif shots_change > 35 and quality_change > 25:
        f.append(f"{team} shot more and better: {sh_m:.0f} shots versus {sh_b:.1f} per match, at {qs_m:.2f} xG per shot "
                 f"versus {qs_b:.2f}.")
    else:
        f.append(f"Shots: {sh_b:.1f} per match in the baseline, {sh_m:.0f} in this match; xG per shot {qs_b:.2f} -> {qs_m:.2f}.")
    f.append(f"Average shot distance went from {d_b:.1f} m to {d_m:.1f} m, and the share of shots from inside the box "
             f"from {box_b*100:.0f}% to {box_m*100:.0f}%.")
    f.append(f"Shots on target: {ot_m:.0f}, versus {ot_b:.1f} per match.")
    oq_b, oq_m = q.loc["xg_per_shot", "against_base_avg"], q.loc["xg_per_shot", "against_match"]
    osh_b, osh_m = q.loc["shots", "against_base_avg"], q.loc["shots", "against_match"]
    f.append(f"{opp} took {osh_m:.0f} shots (opponents averaged {osh_b:.1f}) at {oq_m:.2f} xG per shot "
             f"(opponents averaged {oq_b:.2f}).")
    return f


# ------------------------------------------------------------------------------------------- player level
def _eligible(p: pd.DataFrame, result: dict) -> pd.DataFrame:
    """Starters who played most of this match and enough of the baseline to make per-90 numbers meaningful."""
    min_base = result.get("meta", {}).get("min_base_minutes", MIN_BASE_MINUTES)
    return p[(p["minutes_match"] >= MIN_MATCH_MINUTES) & (p["minutes_base"] >= min_base)]


def rating_findings(result: dict) -> list:
    p = result["players"]
    if p.empty or "rating" not in set(p["metric"]):
        return []
    rt = _eligible(p[p["metric"] == "rating"], result).dropna(subset=["base", "match"])
    if len(rt) < 3:
        return []
    return [f"The {len(rt)} regulars averaged a rating of {rt['match'].mean():.2f} in this match, against an average "
            f"of {rt['base'].mean():.2f} in the earlier matches."]


def _pattern(x: dict, role):
    """The strongest interpretable pattern for one player, as (sentence, metrics it already describes), or None.
    Shooting patterns are never applied to defenders or goalkeepers."""
    name = x["player"]
    can_shoot, can_create = role in (None, "F", "M"), role in (None, "F", "M", "D")
    kp_change = pct(x["kp_b"], x["kp_m"]) if x["kp_b"] >= MIN_BASE_PER90 else np.nan
    sh_change = pct(x["shots_b"], x["shots_m"]) if x["shots_b"] >= MIN_BASE_PER90 else np.nan
    q_change = pct(x["qps_b"], x["qps_m"]) if pd.notna(x["qps_b"]) and pd.notna(x["qps_m"]) else np.nan
    if can_create and pd.notna(kp_change) and kp_change <= -70 and (pd.isna(sh_change) or sh_change > -30) and pd.notna(x["xa_b"]):
        if can_shoot and x["shots_b"] >= MIN_BASE_PER90:
            return (f"{name} was shut out as a creator, not as a shooter: key passes per 90 went from an average of "
                    f"{x['kp_b']:.2f} before to {x['kp_m']:.2f} and xA from {x['xa_b']:.2f} to {x['xa_m']:.2f}, while his "
                    f"shots went from {x['shots_b']:.2f} to {x['shots_m']:.2f} per 90.", {"key_passes", "xa", "shots"})
        return (f"{name}'s creation dried up: key passes per 90 went from an average of {x['kp_b']:.2f} before to "
                f"{x['kp_m']:.2f} and xA from {x['xa_b']:.2f} to {x['xa_m']:.2f}.", {"key_passes", "xa"})
    if can_shoot and pd.notna(sh_change) and sh_change >= 25 and pd.notna(q_change) and q_change <= -50:
        return (f"{name} shot more but worse: shots per 90 went from an average of {x['shots_b']:.2f} before to "
                f"{x['shots_m']:.2f}, with xG per shot falling from {x['qps_b']:.2f} to {x['qps_m']:.2f}.", {"shots", "xg"})
    if can_create and pd.notna(kp_change) and kp_change >= 50 and x["kp_m"] and pd.notna(x["xa_b"]) and pd.notna(x["xa_m"]) \
            and (x["xa_m"] / x["kp_m"]) < 0.6 * (x["xa_b"] / x["kp_b"]):
        return (f"{name} created more often but less danger: key passes per 90 rose from an average of {x['kp_b']:.2f} "
                f"before to {x['kp_m']:.2f}, yet xA only went from {x['xa_b']:.2f} to {x['xa_m']:.2f}. Busy, not dangerous.",
                {"key_passes", "xa"})
    if can_create and pd.notna(kp_change) and kp_change >= 50 and pd.notna(x["xa_b"]) and pct(x["xa_b"], x["xa_m"]) >= 50:
        return (f"{name} drove the creation: key passes per 90 rose from an average of {x['kp_b']:.2f} before to "
                f"{x['kp_m']:.2f} and xA from {x['xa_b']:.2f} to {x['xa_m']:.2f}.", {"key_passes", "xa"})
    if can_shoot and pd.notna(q_change) and q_change <= -60:
        verb = "kept shooting" if pd.notna(sh_change) and sh_change > -30 else "shot less"
        return (f"{name} {verb} (shots per 90 from an average of {x['shots_b']:.2f} before to {x['shots_m']:.2f}) "
                f"and from worse positions: xG per shot fell from {x['qps_b']:.2f} to {x['qps_m']:.2f}.", {"shots", "xg"})
    if can_shoot and pd.notna(sh_change) and sh_change >= 50 and pd.notna(q_change) and q_change >= 25:
        return (f"{name} shot more and better: shots per 90 rose from an average of {x['shots_b']:.2f} before to "
                f"{x['shots_m']:.2f}, with xG per shot rising from {x['qps_b']:.2f} to {x['qps_m']:.2f}.", {"shots", "xg"})
    return None


def player_findings(result: dict, max_players: int = 7) -> list:
    """Per player: the significant changes in the statistics that matter for his position (up to three, biggest
    first, phrased as 'from an average of X before to Y'), plus one interpretation when a pattern applies.
    Players are ordered by the size of their biggest significant change."""
    p = result["players"]
    if p.empty:
        return []
    roles = result.get("roles", {})
    el = _eligible(p, result).dropna(subset=["base", "match"])
    f = rating_findings(result)
    wide = p[p["metric"] != "rating"].pivot_table(index="player", columns="metric", values=["base", "match"])

    # team-wide build-up pattern
    bu_rows = []
    for player in el["player"].unique():
        if player in wide.index and roles.get(player) != "G":
            b, m = wide.loc[player, "base"], wide.loc[player, "match"]
            bu_rows.append((_g(b, "xg_buildup"), _g(m, "xg_buildup")))
    bu = pd.DataFrame(bu_rows, columns=["b", "m"]).dropna()
    bu = bu[bu["b"] >= MIN_BASE_PER90]
    if len(bu) >= 3:
        change = (bu["m"] / bu["b"] - 1) * 100
        if (change <= -50).mean() >= 0.75:
            c = change[change <= -50]
            f.append(f"xGBuildup fell for {len(c)} of the {len(bu)} starters (60+ minutes) who normally contribute to "
                     f"build-up, by between {abs(c.max()):.0f}% and {abs(c.min()):.0f}%. A drop that uniform points to the "
                     f"build-up being disrupted as a team, not one player having a bad day.")
        elif (change >= 50).mean() >= 0.75:
            c = change[change >= 50]
            f.append(f"xGBuildup rose for {len(c)} of the {len(bu)} starters who normally contribute to build-up, "
                     f"by between {c.min():.0f}% and {c.max():.0f}%: the whole team was more involved in dangerous possessions.")

    entries = []
    for player, grp in el.groupby("player"):
        role = roles.get(player)
        pattern = None
        if player in wide.index:
            b, m = wide.loc[player, "base"], wide.loc[player, "match"]
            x = {"player": player, "shots_b": _g(b, "shots"), "shots_m": _g(m, "shots"),
                 "qps_b": _ratio(_g(b, "xg"), _g(b, "shots")), "qps_m": _ratio(_g(m, "xg"), _g(m, "shots")),
                 "kp_b": _g(b, "key_passes"), "kp_m": _g(m, "key_passes"), "xa_b": _g(b, "xa"), "xa_m": _g(m, "xa")}
            pattern = _pattern(x, role)
        covered = pattern[1] if pattern else set()
        sig = [(rel.strength(r.metric, r.base, r.match), r) for r in grp.itertuples()
               if rel.relevant(role, r.metric) and rel.significant(r.metric, r.base, r.match)]
        if not sig and not pattern:
            continue
        score = max([s for s, _ in sig] + [1.0 if pattern else 0.0])
        parts = [f"{rel.LABEL[r.metric]} {'fell' if r.match < r.base else 'rose'} from an average of "
                 f"{rel.fmt(r.metric, r.base)} before to {rel.fmt(r.metric, r.match)}"
                 for s, r in sorted(sig, key=lambda t: -t[0]) if r.metric not in covered][:3]
        lines = []
        if parts:
            who = f"{player} ({rel.ROLE_NAME[role]})" if role in rel.ROLE_NAME else player
            lines.append(f"{who}: " + "; ".join(parts) + ".")
        if pattern:
            lines.append(pattern[0])
        entries.append((score, lines))
    for _, lines in sorted(entries, key=lambda e: -e[0])[:max_players]:
        f += lines
    return f


def findings(result: dict) -> list:
    team, players = team_findings(result), player_findings(result)
    meta = result["meta"]
    if players:
        if meta["n_base"] < 4:
            players.append(f"Caution: the player comparison rests on only {meta['n_base']} earlier matches, "
                           f"so per-90 changes are noisy.")
    elif meta.get("has_players", True):
        players = [f"No player-level findings: no outfield player met the minimum of {MIN_MATCH_MINUTES} minutes in this "
                   f"match and {meta['min_base_minutes']} minutes in the baseline of {meta['n_base']} earlier matches, "
                   f"or none changed enough to matter."]
    return team + players
