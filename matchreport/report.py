"""Build a full report folder: tables, charts, findings, optional LLM narrative, and a markdown summary."""
import json
from pathlib import Path

from matchreport import charts, insights, metrics, narrative, store


def build(team, opponent, data_dir: Path, out_root: Path, date=None, llm=True, same_competition=True,
          baseline="before", source=None, period=None) -> Path:
    matches, shots, pm = store.load_all(data_dir, sources=[source] if source else None)
    result = metrics.compare(matches, shots, pm, team, opponent, date, same_competition, baseline, period=period)
    return build_from_result(result, out_root, llm)


def build_from_result(result: dict, out_root: Path, llm: bool = True) -> Path:
    meta = result["meta"]
    if meta.get("mode") == "periods":
        slug = lambda s: "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")
        out = out_root / f"{slug(meta['team'])}_{slug(meta['focus_label'])}_vs_{slug(meta['base_label'])}"
    else:
        which = f"range_{meta['period'][0]}_to_{meta['period'][1]}" if meta.get("period") else meta["baseline"]
        out = out_root / f"{meta['team']}_vs_{meta['opponent']}_{meta['date']}_{which}".replace(" ", "_")
    out.mkdir(parents=True, exist_ok=True)

    result["team_table"].round(2).to_csv(out / "team_table.csv")
    result["shot_quality"].round(3).to_csv(out / "shot_quality.csv")
    result["players"].round(2).to_csv(out / "players.csv", index=False)
    result["team_changes"].round(3).to_csv(out / "team_changes.csv")
    result["player_changes"].round(2).to_csv(out / "player_changes.csv", index=False)
    if meta["has_xg"]:
        charts.xg_bars(result, out / "xg_bars.png")
    if not result["team_changes"].empty:
        charts.team_change_bars(result, out / "team_changes.png")
    if not result["player_changes"].empty:
        charts.player_change_bars(result, out / "player_changes.png")
    if meta["has_shots"]:
        charts.shot_maps(result, out / "shot_maps.png")
    has_lineup = not result["lineup"].empty
    if has_lineup:
        metrics_here = charts.available_lineup_metrics(result)
        if metrics_here:
            charts.lineup_chart(result, out / "lineup.png", metrics_here[0])
        else:
            has_lineup = False

    found = insights.findings(result)
    (out / "findings.txt").write_text("\n".join(found))
    text, check = "", None
    if llm and narrative.BACKEND not in ("none", "stub"):
        nar = narrative.write(result)
        text, check = nar["text"], nar["check"]
        (out / "prompt.txt").write_text(nar["prompt"])
        (out / "validation.json").write_text(json.dumps(check, indent=2))

    if meta.get("mode") == "periods":
        fm, bm = result["focus_matches"], result["base_matches"]
        listing = lambda d: [f"- {r.date} vs {r.opponent} ({r.venue}) {r.result} {int(r.goals_for)}-{int(r.goals_against)}"
                             for r in d.itertuples()]
        head = ([f"# {meta['team']}: {meta['focus_label']} vs {meta['base_label']}", "",
                 f"Competitions: {meta['competition']}. Sources: {', '.join(meta['sources'])}.", "",
                 f"## {meta['focus_label']} ({meta['n_focus']} matches)", ""] + listing(fm) +
                ["", f"## {meta['base_label']} ({meta['n_base']} matches)", ""] + listing(bm) + [""])
    else:
        head = [f"# {meta['team']} vs {meta['opponent']}, {meta['date']} ({meta['competition']}, {meta['result']} {meta['score']})", "",
                f"Baseline: {meta['baseline_label']}. Sources: {', '.join(meta['sources'])}.", ""]
    md = head + [
          "## Findings (computed by rules in insights.py)", ""] + [f"- {s}" for s in found] + ["",
          ] + (["![xG](xg_bars.png)", ""] if meta["has_xg"] else []) \
        + (["## Starting XI", "", "![Starting XI](lineup.png)", ""]
           + [f"- {s}" for s in charts.lineup_summary(result, charts.available_lineup_metrics(result)[0])] + [""]
           if has_lineup else []) + [
          "## Biggest team-level changes", "", result["team_changes"].round(2).to_markdown(), ""] \
        + (["![Team changes](team_changes.png)", ""] if not result["team_changes"].empty else [])
    if meta["has_shots"]:
        md += ["## Shot quality", "", result["shot_quality"].round(3).to_markdown(), "", "![Shots](shot_maps.png)", ""]
    md += ["## Biggest player changes (per 90, starters with enough minutes)", "",
           result["player_changes"].round(2).to_markdown(index=False), ""]
    if not result["player_changes"].empty:
        md += ["![Player changes](player_changes.png)", ""]
    if text:
        status = ("every number checked against the data" if check["passed"]
                  else f"UNVERIFIED numbers: {check['unverified']}")
        md += ["## Written summary", "", f"*Written by {narrative.model_name()} from the findings above; {status}.*", "", text, ""]
    (out / "report.md").write_text("\n".join(md))
    return out
