"""Share-ready charts in the style of the published articles (dark, 16:9, large type), for any comparison.

Grey is always the baseline (the usual level, or the earlier period); red is the match or the period being judged."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matchreport import charts, relevance

BG, GRID, TEXT, SUB, OLD, NEW, GOOD = "#0b0b0b", "#222222", "#eeeeee", "#bbbbbb", "#7a7f87", "#DA291C", "#2E8B57"


def _names(meta: dict) -> tuple:
    if meta.get("mode") == "periods":
        return f"{meta['base_label']} ({meta['n_base']} matches)", f"{meta['focus_label']} ({meta['n_focus']} matches)"
    return f"Average of {meta['n_base']} earlier matches", f"vs {meta['opponent']} ({meta['result']} {meta['score']})"


def _frame(meta, title, subtitle):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.suptitle(title, x=0.05, y=0.96, ha="left", color="white", fontsize=30, fontweight="bold")
    fig.text(0.05, 0.875, subtitle, ha="left", color=SUB, fontsize=15)
    old, new = _names(meta)
    fig.text(0.05, 0.035, "\u25CF", color=OLD, fontsize=16, va="center"); fig.text(0.065, 0.035, old, color=SUB, fontsize=13, va="center")
    fig.text(0.42, 0.035, "\u25CF", color=NEW, fontsize=16, va="center"); fig.text(0.435, 0.035, new, color=SUB, fontsize=13, va="center")
    fig.text(0.98, 0.035, f"Data: {', '.join(meta['sources'])}  |  Match Compare", ha="right", color="#888888", fontsize=11, va="center")
    return fig


def _style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=SUB, labelsize=14, length=0)


def _pct(b, a):
    return (a / b - 1) * 100 if b else np.nan


def _grouped(ax, groups, labels, fmt="{:.2f}"):
    w = 0.34
    top = max(max(g) for g in groups) or 1
    for i, (o, n) in enumerate(groups):
        ax.bar(i - w / 2 - 0.02, o, w, color=OLD); ax.bar(i + w / 2 + 0.02, n, w, color=NEW)
        ax.text(i - w / 2 - 0.02, o, fmt.format(o), ha="center", va="bottom", color=SUB, fontsize=20)
        ax.text(i + w / 2 + 0.02, n, fmt.format(n), ha="center", va="bottom", color=NEW, fontsize=22, fontweight="bold")
        p = _pct(o, n)
        if pd.notna(p):
            ax.text(i, max(o, n) + top * 0.13, f"{p:+.0f}%", ha="center", color=TEXT, fontsize=24, fontweight="bold")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(labels, color=TEXT, fontsize=18, fontweight="bold")
    ax.set_yticks([]); ax.set_ylim(0, top * 1.35); ax.set_xlim(-0.6, len(groups) - 0.4)


def _team_pair(result, key):
    t = result["team_table"]
    return float(t.loc[key, "base_avg"]), float(t.loc[key, "match"])


def chances_and_goals(result: dict, out: Path) -> Path:
    meta = result["meta"]
    fig = _frame(meta, "Chances created and goals scored", f"{meta['team']}, per match.")
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.66]); _style(ax)
    _grouped(ax, [_team_pair(result, "xg_for"), _team_pair(result, "goals_for")], ["xG created", "Goals scored"])
    fig.savefig(out, dpi=110, facecolor=BG); plt.close(fig)
    return Path(out)


def defence(result: dict, out: Path) -> Path:
    meta = result["meta"]
    fig = _frame(meta, "Chances and goals conceded", f"{meta['team']}, per match. Lower is better.")
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.66]); _style(ax)
    _grouped(ax, [_team_pair(result, "xg_against"), _team_pair(result, "goals_against")], ["xG conceded", "Goals conceded"])
    fig.savefig(out, dpi=110, facecolor=BG); plt.close(fig)
    return Path(out)


def finishing(result: dict, out: Path) -> Path:
    """Goals scored minus xG created: above zero means scoring more than the chances deserved."""
    meta = result["meta"]
    (xb, xa), (gb, ga) = _team_pair(result, "xg_for"), _team_pair(result, "goals_for")
    old_v, new_v = gb - xb, ga - xa
    old, new = _names(meta)
    fig = _frame(meta, "Goals against chances", "Goals scored minus xG created, per match. Above zero = scoring more than the chances deserved.")
    ax = fig.add_axes([0.30, 0.22, 0.62, 0.52]); _style(ax)
    ax.barh([1], [old_v], 0.5, color=OLD); ax.barh([0], [new_v], 0.5, color=NEW)
    ax.axvline(0, color="#999999", lw=1.2)
    span = max(abs(old_v), abs(new_v), 0.3)
    for y, v, col in ((1, old_v, SUB), (0, new_v, NEW)):
        ax.text(v + (0.03 if v >= 0 else -0.03) * span * 3, y, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                color=col, fontsize=28, fontweight="bold")
    ax.set_yticks([1, 0]); ax.set_yticklabels([old, new], color=TEXT, fontsize=16, fontweight="bold")
    ax.set_xlim(-span * 1.6, span * 1.6); ax.set_xticks([]); ax.set_ylim(-0.6, 1.6)
    fig.text(0.05, 0.12, f"A swing of {new_v - old_v:+.2f} goals a match. Goals went from {gb:.2f} to {ga:.2f}; xG from {xb:.2f} to {xa:.2f}.",
             color=TEXT, fontsize=16)
    fig.savefig(out, dpi=110, facecolor=BG); plt.close(fig)
    return Path(out)


def same_opponents(result: dict, out: Path):
    """Period comparisons only: results against the opponents met in both periods. None if fewer than two."""
    meta = result["meta"]
    if meta.get("mode") != "periods":
        return None
    fm, bm = result["focus_matches"], result["base_matches"]
    shared = [o for o in dict.fromkeys(bm["opponent"]) if o in set(fm["opponent"])]
    if len(shared) < 2:
        return None
    cell = {"W": GOOD, "D": "#8a8f98", "L": NEW}

    def pick(df, opp):
        r = df[df["opponent"] == opp].iloc[0]
        return r["result"], f"{int(r['goals_for'])}-{int(r['goals_against'])}", "H" if r["venue"] == "H" else "A"

    def total(df):
        d = df[df["opponent"].isin(shared)].drop_duplicates("opponent")
        w, dr, l = (d["result"] == "W").sum(), (d["result"] == "D").sum(), (d["result"] == "L").sum()
        return f"{w}W {dr}D {l}L, {3 * w + dr} points", f"goals {int(d['goals_for'].sum())}-{int(d['goals_against'].sum())}"

    old, new = meta["base_label"], meta["focus_label"]
    fig = _frame(meta, f"The same {len(shared)} opponents", f"{meta['team']}'s results against teams met in both periods (H = home, A = away).")
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.76]); ax.set_facecolor(BG); ax.axis("off")
    n = len(shared)
    ax.set_xlim(0, 10); ax.set_ylim(-1.8, n + 1.4)
    ax.text(3.9, n + 0.9, old, color=SUB, fontsize=17, ha="center", fontweight="bold")
    ax.text(7.5, n + 0.9, new, color=NEW, fontsize=17, ha="center", fontweight="bold")
    for i, opp in enumerate(shared):
        y = n - 0.1 - i
        ax.text(0.1, y, opp, color=TEXT, fontsize=17, va="center", fontweight="bold")
        for x, df in ((3.9, bm), (7.5, fm)):
            res, score, ven = pick(df, opp)
            ax.add_patch(plt.Rectangle((x - 1.45, y - 0.4), 2.9, 0.8, color=cell.get(res, "#8a8f98")))
            ax.text(x, y, f"{res}  {score}  ({ven})", color="white", fontsize=18, ha="center", va="center", fontweight="bold")
    ax.plot([0.1, 9.9], [-0.15, -0.15], color="#444444", lw=1)
    ax.text(0.1, -0.85, "Total", color=TEXT, fontsize=18, va="center", fontweight="bold")
    for x, df, col in ((3.9, bm, SUB), (7.5, fm, NEW)):
        line1, line2 = total(df)
        ax.text(x, -0.65, line1, color=col, fontsize=17, ha="center", va="center", fontweight="bold")
        ax.text(x, -1.3, line2, color=col, fontsize=15, ha="center", va="center")
    fig.savefig(out, dpi=110, facecolor=BG); plt.close(fig)
    return Path(out)


def player_stat(result: dict, out: Path, metric: str, limit: int = 10) -> Path:
    """The players who moved most on one stat, as arrows from their usual value to the new one."""
    meta = result["meta"]
    label = charts.LINEUP_METRICS[metric].split(" (")[0]
    unit = "" if metric == "rating" else " per 90"
    if meta.get("mode") == "periods":
        t = charts.period_metric_table(result, metric).rename(columns={"base": "b", "match": "a"})
    else:
        rows = charts.match_metric_rows(result, metric)
        t = pd.DataFrame(columns=["player", "b", "a"]) if rows.empty else \
            rows[(rows["Note"] == "") & rows["Usual"].notna()].rename(columns={"Player": "player", "Usual": "b", "This match": "a"})
    t = t.assign(move=(t["a"] - t["b"]).abs()).sort_values("move", ascending=False).head(limit) if not t.empty else t
    fig = _frame(meta, f"{label}: the biggest movers", f"{meta['team']}, {label.lower()}{unit}. Arrow from the usual value to the new one.")
    ax = fig.add_axes([0.22, 0.12, 0.72, 0.68]); _style(ax)
    if t.empty:
        ax.axis("off"); ax.text(0.5, 0.5, "No player has enough minutes to compare on this stat.", color=TEXT, ha="center", fontsize=18)
    else:
        t = t.iloc[::-1]
        f = lambda v: relevance.fmt(metric, v)
        lo, hi = float(min(t["a"].min(), t["b"].min())), float(max(t["a"].max(), t["b"].max()))
        pad = (hi - lo) * 0.12 or 0.5
        for i, r in enumerate(t.itertuples()):
            col = NEW if r.a < r.b else GOOD
            ax.annotate("", xy=(r.a, i), xytext=(r.b, i), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5, mutation_scale=18, shrinkA=8, shrinkB=8))
            ax.scatter([r.b], [i], s=160, color=OLD, zorder=3); ax.scatter([r.a], [i], s=160, color=col, zorder=3)
            ax.text(hi + pad * 0.6, i, f"{f(r.b)} \u2192 {f(r.a)}", va="center", color=col, fontsize=15, fontweight="bold")
        ax.set_yticks(range(len(t))); ax.set_yticklabels(t["player"], color=TEXT, fontsize=15, fontweight="bold")
        ax.set_xlim(lo - pad, hi + pad * 4); ax.set_xticks([]); ax.grid(False)
    fig.savefig(out, dpi=110, facecolor=BG); plt.close(fig)
    return Path(out)
