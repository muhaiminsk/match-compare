"""Before vs after chart: each team's average xG created and conceded in its earlier matches, against one match.

Uses your Match Compare data (Highlightly). Run from the match-compare folder:
  python before_after_xg.py --home Barcelona --away "Real Madrid" --date 2026-10-25
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAM_COLORS = {"Barcelona": "#E0457B", "Real Madrid": "#FEBE10"}


def points_from_data(home, away, date, source="highlightly", data_dir="data"):
    from matchreport import metrics, store
    m, s, p = store.load_all(Path(data_dir), sources=[source])
    out = []
    for team, opp in ((home, away), (away, home)):
        r = metrics.compare(m, s, p, team, opp, date=date)
        t = r["team_table"]
        out.append({"team": team, "n": r["meta"]["n_base"],
                    "before": (t.loc["xg_for", "base_avg"], t.loc["xg_against", "base_avg"]),
                    "after": (t.loc["xg_for", "match"], t.loc["xg_against", "match"])})
    return out


def draw(points, title, subtitle, out_path, source_note):
    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor="#0b0b0b")
    ax.set_facecolor("#0b0b0b")
    xs = [v for p in points for v in (p["before"][0], p["after"][0])]
    ys = [v for p in points for v in (p["before"][1], p["after"][1])]
    x_lo, x_hi = min(xs) - 0.5, max(xs) + 0.5
    y_lo, y_hi = min(ys) - 0.35, max(ys) + 0.35
    d_lo, d_hi = max(x_lo, y_lo), min(x_hi, y_hi)
    ax.plot([d_lo, d_hi], [d_lo, d_hi], ls="--", lw=1, color="#555555")
    mid = (d_lo + d_hi) / 2
    ax.text(mid + 0.06, mid - 0.03, "created = conceded", color="#777777", fontsize=9, ha="left", va="center")
    for p in points:
        c = TEAM_COLORS.get(p["team"], "#FFFFFF")
        (bx, by), (axx, ayy) = p["before"], p["after"]
        ax.annotate("", xy=(axx, ayy), xytext=(bx, by),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2, mutation_scale=18, shrinkA=9, shrinkB=9))
        ax.scatter([bx], [by], s=170, facecolors="none", edgecolors=c, linewidths=2.2, zorder=3)
        ax.scatter([axx], [ayy], s=170, color=c, zorder=3)
        # labels go on the side of each point facing away from the plot edge
        bx_right = bx < (x_lo + x_hi) / 2
        ax.annotate(f"{p['team']}: average of {p['n']} earlier matches\n{bx:.2f} created, {by:.2f} conceded",
                    (bx, by), xytext=(14 if bx_right else -14, 12), textcoords="offset points", color=c, fontsize=10,
                    ha="left" if bx_right else "right", va="bottom")
        above_left = p.get("after_label") == "above-left"     # optional override when a label would cross an arrow
        ax_right = axx < (x_lo + x_hi) / 2 and not above_left
        ax.annotate(f"{p['team']} in this match\n{axx:.2f} created, {ayy:.2f} conceded",
                    (axx, ayy), xytext=(14 if ax_right else -14, 14 if above_left else -14), textcoords="offset points",
                    color=c, fontsize=10, fontweight="bold", ha="left" if ax_right else "right",
                    va="bottom" if above_left else "top")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_hi, y_lo)                  # conceding less is better: flip so better is up
    ax.set_xlabel("xG created per match", color="#eeeeee", fontsize=12)
    ax.set_ylabel("xG conceded per match", color="#eeeeee", fontsize=12)
    ax.text(0.99, 0.02, "Better: top right", transform=ax.transAxes, ha="right", va="bottom", color="#F2A541", fontsize=10)
    ax.tick_params(colors="#cccccc")
    ax.grid(color="#222222", lw=0.6)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    fig.suptitle(title, x=0.07, ha="left", color="white", fontsize=17, fontweight="bold")
    ax.set_title(subtitle, loc="left", color="#bbbbbb", fontsize=10)
    fig.text(0.99, 0.01, f"Hollow = earlier average, filled = this match. {source_note}  |  @SamitOTB",
             ha="right", color="#888888", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default="Barcelona")
    ap.add_argument("--away", default="Real Madrid")
    ap.add_argument("--date", default="2026-10-25")
    ap.add_argument("--source", default="highlightly")
    ap.add_argument("--out", default="clasico_before_after_xg.png")
    a = ap.parse_args()
    pts = points_from_data(a.home, a.away, a.date, a.source)
    pts[1]["after_label"] = "above-left"
    draw(pts, f"{a.home} vs {a.away}: usual xG against this match",
         f"xG per match: each team's average in its earlier same-competition matches this season vs the match on {a.date}.",
         a.out, f"Data: {a.source.capitalize()}")
    print("written", a.out)
