"""Charts: xG bars, shot maps, and biggest-change bars. All return the saved path."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from matchreport import relevance

TEAM_COLOR, OPP_COLOR, MATCH_COLOR = "#004D98", "#9AA5B1", "#A50044"
UP, DOWN = "#2E8B57", "#C0392B"
LABELS = {"points_per_game": "Points per match", "goals_for": "Goals scored", "goals_against": "Goals conceded", "xg_for": "xG created",
          "xg_against": "xG conceded", "shots_for": "Shots", "shots_against": "Shots conceded",
          "xg_per_shot_for": "xG per shot", "xg_per_shot_against": "xG per shot conceded",
          "box_share_for": "Share of shots in the box", "box_share_against": "Opp. share in the box",
          "on_target_for": "Shots on target", "on_target_against": "Opp. shots on target",
          "mean_distance_m_for": "Mean shot distance", "mean_distance_m_against": "Opp. mean shot distance",
          "passes": "Passes attempted", "pass_accuracy_pct": "Pass accuracy %", "tackles": "Tackles",
          "interceptions": "Interceptions", "duel_win_pct": "Duels won %", "dribbles_completed": "Dribbles completed",
          "dribbled_past": "Times dribbled past",
          "offsides": "Caught offside", "opp_offsides": "Opponent caught offside"}


def _credit(fig, meta):
    fig.text(0.99, 0.01, "Data: " + ", ".join(meta["sources"]), ha="right", fontsize=8, color="gray")


def xg_bars(result: dict, out: Path) -> Path:
    t, meta = result["team_table"], result["meta"]
    if meta.get("mode") == "periods":
        labels = [f"{meta['base_label']}\n({meta['n_base']} matches, avg per match)",
                  f"{meta['focus_label']}\n({meta['n_focus']} matches, avg per match)"]
    else:
        labels = [f"Average of {meta['baseline_label']}", f"vs {meta['opponent']} ({meta['result']} {meta['score']})"]
    fig, ax = plt.subplots(figsize=(7, 5))
    w = 0.38
    b1 = ax.bar([-w / 2, 1 - w / 2], [t.loc["xg_for", "base_avg"], t.loc["xg_for", "match"]], w,
                color=TEAM_COLOR, label=f"{meta['team']} xG")
    b2 = ax.bar([w / 2, 1 + w / 2], [t.loc["xg_against", "base_avg"], t.loc["xg_against", "match"]], w,
                color=OPP_COLOR, label="Opponent xG")
    ax.bar_label(b1, fmt="%.2f"); ax.bar_label(b2, fmt="%.2f")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Expected goals (xG)")
    ax.set_title(f"{meta['team']}: chances created vs conceded")
    ax.legend(frameon=False)
    _credit(fig, meta)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return out


def shot_maps(result: dict, out: Path) -> Path:
    from mplsoccer import VerticalPitch
    meta = result["meta"]
    base, match = result["shots_base"], result["shots_match"]
    pitch = VerticalPitch(pitch_type="opta", half=True, line_color="#666666")
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(11, 6))
    for ax, s, title, color in [
        (axes[0], base, f"Baseline: {meta['n_base_with_shots']} matches, {len(base)} shots", TEAM_COLOR),
        (axes[1], match, f"vs {meta['opponent']}: {len(match)} shots", MATCH_COLOR),
    ]:
        goals, others = s[s["result"] == "Goal"], s[s["result"] != "Goal"]
        pitch.scatter(others["x"] * 100, others["y"] * 100, s=others["xg"] * 600 + 15, color=color,
                      alpha=0.45, edgecolors="black", linewidth=0.4, ax=ax)
        pitch.scatter(goals["x"] * 100, goals["y"] * 100, s=goals["xg"] * 600 + 15, color="gold",
                      edgecolors="black", linewidth=0.8, marker="*", ax=ax)
        ax.set_title(title, fontsize=11)
    fig.suptitle(f"{meta['team']} shots (size = xG, star = goal)", fontsize=13)
    _credit(fig, meta)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


# Measures where a rise is bad for the team, so a rise is drawn red and a fall green
WORSE_WHEN_UP = {"goals_against", "xg_against", "dribbled_past", "offsides", "shots_against", "xg_per_shot_against",
                 "box_share_against", "on_target_against"}


def change_bars(changes: pd.DataFrame, labels: list, out: Path, title: str, meta: dict, limit: int = 12) -> Path:
    """Horizontal bars of percentage change, biggest falls at the top and rises at the bottom.
    Red means worse for the team, green better."""
    df = changes.copy()
    df["label"] = labels
    df["worse_up"] = [str(i) in WORSE_WHEN_UP for i in changes.index]
    df = pd.concat([df.head(limit // 2), df.tail(limit // 2)]).drop_duplicates("label")
    vals = df["change_pct"].clip(-100, 300)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(df) + 1.5))
    ax.barh(df["label"], vals, color=[(UP if v < 0 else DOWN) if w else (DOWN if v < 0 else UP)
                                      for v, w in zip(df["change_pct"], df["worse_up"])])
    for y, (v, b, m) in enumerate(zip(df["change_pct"], df["base"], df["match"])):
        ax.text(315, y, f"{b:.2f} -> {m:.2f}  ({v:+.0f}%)", va="center", ha="left", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"% change, {meta['focus_label']} vs {meta['base_label']}" if meta.get("mode") == "periods"
                  else "% change vs baseline")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlim(-110, 480)
    ax.set_xticks([-100, 0, 100, 200, 300])
    _credit(fig, meta)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return out


def team_change_bars(result: dict, out: Path) -> Path:
    tc = result["team_changes"]
    return change_bars(tc, [LABELS.get(i, i) for i in tc.index], out,
                       f"{result['meta']['team']}: biggest team-level changes", result["meta"])


def player_change_bars(result: dict, out: Path) -> Path:
    pc = result["player_changes"]
    names = {"shots": "shots", "shots_on_target": "shots on target", "key_passes": "key passes", "xg": "xG",
             "xa": "xA", "xg_chain": "xGChain", "xg_buildup": "xGBuildup"}
    return change_bars(pc, [f"{r.player}: {names.get(r.metric, r.metric)}" for r in pc.itertuples()], out,
                       f"{result['meta']['team']}: biggest player changes (per 90"
                       + (")" if result["meta"].get("mode") == "periods" else ", starters only)"), result["meta"])


# ---------------------------------------------------------------------------------------------
# Starting XI on a pitch: each starter's baseline average vs this match, for one metric per 90.
# Position codes are Understat's (R = right, L = left, from the team's own point of view).
# ---------------------------------------------------------------------------------------------
import numpy as np  # noqa: E402

POSITION_XY = {  # code: (depth 0-100 from own goal, width 0-100 where small = right wing)
    "GK": (6, 50), "DR": (22, 15), "DC": (22, 50), "DL": (22, 85),
    "DMR": (38, 25), "DMC": (38, 50), "DML": (38, 75), "MR": (52, 15), "MC": (52, 50), "ML": (52, 85),
    "AMR": (68, 18), "AMC": (68, 50), "AML": (68, 82), "FWR": (84, 30), "FW": (84, 50), "FWL": (84, 70),
}
LINEUP_METRICS = {"rating": "Match rating (as given by the data source)",
                  "xg_chain": "xGChain (involvement in attacks that end in a shot)",
                  "xg_buildup": "xGBuildup (involvement before the final pass)",
                  "xg_xa": "xG + xA (shooting and chance creation)",
                  "key_passes": "Key passes", "shots": "Shots", "passes": "Passes attempted",
                  "passes_accurate": "Passes completed", "tackles": "Tackles", "interceptions": "Interceptions",
                  "dribbles_completed": "Dribbles completed", "duels_won": "Duels won"}
ROLE_DEPTH = {"D": 24, "M": 52, "F": 80}


def available_lineup_metrics(result: dict) -> list:
    """Only offer metrics this data source actually has."""
    have = set(result["players"]["metric"]) if not result["players"].empty else set()
    out = [m for m in LINEUP_METRICS if (m in have and m != "xg_xa") or (m == "xg_xa" and {"xg", "xa"} <= have)]
    return out


def _grid_positions(lineup: pd.DataFrame) -> pd.DataFrame:
    """Place starters from API-Football grid codes 'row:col' (row 1 = goalkeeper). Column 1 is treated as the
    team's right-hand side, which is the usual order of a lineup listing; check the first chart against a lineup you know."""
    g = lineup.dropna(subset=["grid"]).copy()
    g[["r", "c"]] = g["grid"].astype(str).str.split(":", expand=True).astype(float)
    n_rows = g["r"].max()
    rows = []
    for r_val, grp in g.groupby("r"):
        depth = 6 if r_val == 1 else 6 + (r_val - 1) / max(n_rows - 1, 1) * 78
        grp = grp.sort_values("c")
        n = len(grp)
        ys = [50.0] if n == 1 else list(np.linspace(14, 86, n))
        for (_, p), y in zip(grp.iterrows(), ys):
            rows.append({"player": p["player"], "position": p["position"], "x": float(depth), "y": float(y)})
    return pd.DataFrame(rows, columns=["player", "position", "x", "y"])


def _role_positions(lineup: pd.DataFrame) -> pd.DataFrame:
    """Fallback without grid codes: goalkeeper, then defenders / midfielders / forwards spread across each line.
    The order within a line is arbitrary."""
    rows = []
    for code, grp in lineup.groupby("position", sort=False):
        if code == "GK":
            rows += [{"player": r["player"], "position": code, "x": 6.0, "y": 50.0} for _, r in grp.iterrows()]
        elif code in ROLE_DEPTH:
            n = len(grp)
            ys = [50.0] if n == 1 else list(np.linspace(14, 86, n))
            rows += [{"player": r["player"], "position": code, "x": float(ROLE_DEPTH[code]), "y": float(y)}
                     for (_, r), y in zip(grp.iterrows(), ys)]
    return pd.DataFrame(rows, columns=["player", "position", "x", "y"])


def lineup_positions(lineup: pd.DataFrame) -> pd.DataFrame:
    """Place each starter on the pitch: from grid codes if the source has them, from Understat position codes,
    or (fallback) from goalkeeper/defender/midfielder/forward lines."""
    if "grid" in lineup.columns and lineup["grid"].notna().sum() >= 8:
        return _grid_positions(lineup)
    if lineup["position"].isin(list(ROLE_DEPTH)).any():
        return _role_positions(lineup)
    rows = []
    for code, grp in lineup.groupby("position", sort=False):
        if code not in POSITION_XY:
            continue
        x, y = POSITION_XY[code]
        offsets = (np.arange(len(grp)) - (len(grp) - 1) / 2) * 24
        for (_, r), off in zip(grp.iterrows(), offsets):
            rows.append({"player": r["player"], "position": code, "x": x, "y": float(np.clip(y + off, 8, 92))})
    return pd.DataFrame(rows, columns=["player", "position", "x", "y"])


def _short_name(name: str) -> str:
    parts = name.split()
    if len(parts) >= 3 and parts[-2].islower():
        return " ".join(parts[-2:])
    return parts[-1]


def lineup_values(result: dict, metric: str) -> pd.DataFrame:
    """This match's value for each starter, plus his usual value and % change when he has enough earlier
    minutes (otherwise base and change_pct are NaN but the match value is kept)."""
    p = result["players"]
    names = ["xg", "xa"] if metric == "xg_xa" else [metric]
    sub = p[p["metric"].isin(names)]
    if sub.empty:
        return pd.DataFrame(columns=["player", "base", "match", "minutes_base", "change_pct"])
    agg = sub.groupby("player").agg(base=("base", lambda s: s.sum(min_count=1)), match=("match", lambda s: s.sum(min_count=1)),
                                    minutes_base=("minutes_base", "first"))
    agg["change_pct"] = np.where(agg["base"] > 0, (agg["match"] / agg["base"] - 1) * 100, np.nan)
    low = agg["minutes_base"].fillna(0) < result["meta"]["min_base_minutes"]
    agg.loc[low, ["base", "change_pct"]] = np.nan
    return agg.reset_index()


def lineup_chart(result: dict, out: Path, metric: str = "xg_chain") -> Path:
    from matplotlib.colors import Normalize
    from mplsoccer import VerticalPitch
    meta = result["meta"]
    pos = lineup_positions(result["lineup"]).merge(lineup_values(result, metric), on="player", how="left")
    gk = set(result.get("goalkeepers", set()))
    is_rating = metric == "rating"
    pitch = VerticalPitch(pitch_type="opta", line_color="#666666", pitch_color="#f4f6f8")
    fig, ax = pitch.draw(figsize=(7, 10))
    norm, cmap = Normalize(-25 if is_rating else -100, 25 if is_rating else 100), matplotlib.colormaps["RdYlGn"]
    roles = result.get("roles", {})
    for r in pos.itertuples():
        is_gk = r.player in gk
        applies = relevance.relevant(roles.get(r.player), metric)
        shown = applies and (not is_gk or is_rating)            # goalkeepers are only judged on rating
        has_match = shown and pd.notna(r.match)
        compared = has_match and pd.notna(r.change_pct)         # enough earlier minutes for a usual value
        color = cmap(norm(float(np.clip(r.change_pct, norm.vmin, norm.vmax)))) if compared else "#c9ced4"
        pitch.scatter(r.x, r.y, s=1700, color=color, edgecolors="black", linewidth=1.2, zorder=3, ax=ax)
        if compared and not is_rating:
            inner = f"{r.change_pct:+.0f}%"
        elif has_match:                                          # rating, or this match's value when nothing to compare
            inner = f"{r.match:.1f}" if is_rating else relevance.fmt(metric, r.match)
        else:
            inner = "GK" if is_gk else "-"
        strong = compared and (abs(r.change_pct) > (15 if is_rating else 45))
        pitch.annotate(inner, (r.x, r.y), ha="center", va="center", fontsize=10, fontweight="bold",
                       color="white" if strong else "black", zorder=4, ax=ax)
        if compared:
            label = f"usual {r.base:.1f} -> {r.match:.1f}" if is_rating else f"{r.base:.2f} -> {r.match:.2f}"
        elif has_match:
            label = "this match only"
        else:
            label = None
        if is_gk:  # no room below the goalkeeper and the back line sits above: put the name beside the circle
            pitch.annotate(_short_name(r.player), (r.x + 1.5, r.y - 17), ha="center", va="center", fontsize=10, fontweight="bold", ax=ax)
            if label:
                pitch.annotate(label, (r.x - 2, r.y - 17), ha="center", va="center", fontsize=8.5, ax=ax)
            continue
        pitch.annotate(_short_name(r.player), (r.x - 6.5, r.y), ha="center", va="center", fontsize=10, fontweight="bold", ax=ax)
        if label:
            pitch.annotate(label, (r.x - 10, r.y), ha="center", va="center", fontsize=8.5, ax=ax)
    unit = "" if is_rating else " per 90"
    ax.set_title(f"{meta['team']} starting XI vs {meta['opponent']} ({meta['result']} {meta['score']})\n"
                 f"{LINEUP_METRICS[metric]}{unit}: baseline average -> this match", fontsize=11, pad=14)
    note = ("Number in the circle = rating in this match; colour = change vs the player's usual rating."
            if is_rating else "Colour and % = change vs baseline (red = down, green = up).")
    fig.text(0.5, 0.015, note + " Grey circle = this match's value only (under 90 earlier minutes); '-' = not judged for this position.",
             ha="center", fontsize=8, color="#555555")
    how = (f"Rating = the match rating supplied by {', '.join(meta['sources'])}; usual = minutes-weighted average of the "
           f"player's earlier ratings."
           if is_rating else f"Computed from {', '.join(meta['sources'])} data; this is not a match rating.")
    fig.text(0.5, -0.008, f"Baseline: {meta['baseline_label']}. {how}", ha="center", fontsize=8, color="#555555")
    fig.text(0.99, -0.03, "Data: " + ", ".join(meta["sources"]), ha="right", fontsize=7, color="gray")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


def lineup_summary(result: dict, metric: str) -> list:
    """A few sentences for the chosen lineup metric: the starters' average before vs this match, then the
    biggest significant drops and rises, counting only players whose position the metric applies to."""
    label = LINEUP_METRICS[metric].split(" (")[0]
    unit = "" if metric == "rating" else " per 90"
    roles = result.get("roles", {})
    gk = set(result.get("goalkeepers", set()))
    vals = lineup_values(result, metric)
    starters = set(result["lineup"]["player"])
    v = vals[vals["player"].isin(starters)].dropna(subset=["base", "match"])
    mask = pd.Series([relevance.relevant(roles.get(p), metric) and (metric == "rating" or p not in gk) for p in v["player"]],
                     index=v.index, dtype=bool)          # a boolean Series, so an empty frame keeps its columns
    v = v[mask]
    nobase = sorted(p for p in starters if relevance.relevant(roles.get(p), metric) and (metric == "rating" or p not in gk)
                    and p in set(vals["player"]) and p not in set(v["player"]))
    nobase_note = (f"No usual value yet (under {result['meta']['min_base_minutes']} earlier minutes), so only this match "
                   f"is shown: {', '.join(nobase)}." if nobase else None)
    if v.empty:
        return [f"No starter for whom {label.lower()} matters has earlier minutes to compare with."] + ([nobase_note] if nobase_note else [])
    f = lambda x: relevance.fmt(metric, x)
    out = [f"The {len(v)} starters it applies to averaged {f(v['base'].mean())}{unit} in the earlier matches "
           f"and {f(v['match'].mean())} in this one."]
    sig = v[[relevance.significant(metric, b, m) for b, m in zip(v["base"], v["match"])]].copy()
    if sig.empty:
        min_abs, min_pct = relevance.SIGNIFICANCE.get(metric, (0, 0))
        out.append(f"No starter changed enough to matter (at least {min_abs}{unit} and {min_pct}% from his average).")
    else:
        sig["strength"] = [relevance.strength(metric, b, m) for b, m in zip(sig["base"], sig["match"])]
        for direction, rows in (("drops", sig[sig["match"] < sig["base"]]), ("rises", sig[sig["match"] > sig["base"]])):
            if not rows.empty:
                top = rows.sort_values("strength", ascending=False).head(3)
                out.append(f"Biggest {direction}: " + ", ".join(
                    f"{r.player} (from an average of {f(r.base)} before to {f(r.match)})" for r in top.itertuples()) + ".")
    if nobase_note:
        out.append(nobase_note)
    skipped = [p for p in starters if not relevance.relevant(roles.get(p), metric) and p not in gk]
    if skipped:
        out.append(f"Not judged on {label.lower()} because of their position: {', '.join(sorted(skipped))}.")
    return out
