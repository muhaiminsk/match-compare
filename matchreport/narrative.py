"""Write the report with a local LLM from code-computed findings, then check every number it used."""
import os
import re

import numpy as np
import pandas as pd
import requests

from matchreport import insights

# Backends, chosen with LLM_BACKEND:
#   "ollama": local model (OLLAMA_HOST, OLLAMA_MODEL)
#   "openai": any OpenAI-compatible API, e.g. Groq's free tier (OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL)
#   "none":   skip the prose; the findings are shown on their own
BACKEND = os.environ.get("LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "openai/gpt-oss-20b")


def model_name() -> str:
    return {"ollama": OLLAMA_MODEL, "openai": OPENAI_MODEL}.get(BACKEND, "no model")

SYSTEM = ("You are a football analyst. You will be given findings that were computed from match data. "
          "Rewrite them as a short, readable analysis for fans. Rules: use only the facts and numbers in the "
          "findings; never compute or round new numbers; keep every number exactly as written; do not add "
          "players, positions (forwards, midfield, defence, goalkeeper), tactics, injuries, or context that are not in the findings. "
          "If the findings do not mention a group of players, do not mention it either. Keep every number attached to the same "
          "team or player and the same measure as in the findings: never say a team limited or caused something that the "
          "findings say it suffered, and never describe a defensive statistic as offensive. When a finding compares a player's "
          "average in the earlier matches with this match, keep that framing explicitly (for example: 'his rating fell from "
          "an average of 7.50 in the earlier matches to 5.51'), never just 'from 7.50 to 5.51'. Mention only the "
          "statistics the findings mention for each player. Where a finding offers an "
          "interpretation, you may keep it, phrased as what the numbers suggest. No bullet points, no headings.")


def build_prompt(result: dict) -> tuple:
    meta = result["meta"]
    found = insights.findings(result)
    if meta.get("mode") == "periods":
        head = (f"Comparison of two periods for {meta['team']}: {meta['focus_label']} ({meta['n_focus']} matches) "
                f"against {meta['base_label']} ({meta['n_base']} matches). All team figures are averages per match; "
                f"player figures are per 90 minutes. Always name which period a number belongs to.")
    else:
        head = (f"Match: {meta['team']} vs {meta['opponent']} ({meta['venue']}), {meta['date']}, "
                f"result {meta['result']} {meta['score']}. Baseline: {meta['baseline_label']}.")
    body = "\n".join(f"- {s}" for s in found)
    has_players = bool(insights.player_findings(result))
    if has_players:
        task = ("Write 3 paragraphs, about 200 words in total. Paragraph 1: the team-level picture. "
                "Paragraph 2: shots, volume versus quality. Paragraph 3: the players named in the findings. "
                "Start with the most important point, not with the score.")
        if meta.get("mode") == "periods":
            task = ("Write 3 paragraphs, about 220 words in total. Paragraph 1: results and the team-level picture. "
                    "Paragraph 2: chances created and conceded, and finishing. Paragraph 3: the players named in the "
                    "findings. Start with the most important difference between the two periods.")
    else:
        task = ("Write 2 paragraphs, about 150 words in total. Paragraph 1: the team-level picture. "
                "Paragraph 2: shots, volume versus quality. There are no player-level findings, so do not discuss "
                "individual players, positions, or lines of the team. Start with the most important point, not with the score.")
    return f"{head}\n\nFINDINGS:\n{body}\n\n{task}", found


def _numbers_in(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", text.replace(",", "")))


def allowed_numbers(result: dict, found: list) -> set:
    """Every number in the computed tables (in a few rounding forms) plus every number in the findings."""
    vals = set()
    def add(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        if pd.isna(f) or not np.isfinite(f):
            return
        for s in (f"{f:.2f}", f"{f:.1f}", f"{f:.0f}", str(int(f)) if f == int(f) else None):
            if s is not None:
                vals.add(s.lstrip("-"))
    for df in (result["team_table"], result["shot_quality"], result["players"]):
        for v in df.apply(pd.to_numeric, errors="coerce").values.ravel():
            add(v)
    meta = result["meta"]
    for part in (meta.get("score") or "").split("-"):      # a period comparison has no single score
        add(part)
    add(meta["n_base"])
    add(meta.get("n_focus"))
    for s in found:
        vals |= _numbers_in(s)
    return vals


POSITION_WORDS = re.compile(r"\b(forwards?|strikers?|wingers?|midfield(?:ers?)?|defenders?|defen[cs]e|back ?line|"
                            r"full-?backs?|centre-?backs?|center-?backs?|goalkeepers?|keepers?)\b", re.I)


def validate(text: str, result: dict, found: list) -> dict:
    """Two checks: every number exists in the data, and no position or line-of-the-team claim appears
    unless the findings themselves talk about it."""
    nums = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
    ok = allowed_numbers(result, found)
    year = (result["meta"].get("date") or "")[:4]
    unverified = [n for n in nums if n not in ok and n not in {year, "90", "60"}]
    allowed_words = {w.lower() for w in POSITION_WORDS.findall(" ".join(found))}
    unsupported = sorted({w.lower() for w in POSITION_WORDS.findall(text)} - allowed_words)
    return {"numbers_found": len(nums), "unverified": unverified, "unsupported_terms": unsupported,
            "passed": not unverified and not unsupported}


def _call(prompt: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    try:
        if BACKEND == "ollama":
            r = requests.post(f"{OLLAMA_HOST}/api/chat", timeout=300,
                              json={"model": OLLAMA_MODEL, "stream": False, "options": {"temperature": 0}, "messages": msgs})
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        if BACKEND == "openai":
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set")
            url, headers = f"{OPENAI_BASE_URL}/chat/completions", {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            payload = {"model": OPENAI_MODEL, "temperature": 0, "messages": msgs}
            reasoning = "gpt-oss" in OPENAI_MODEL
            if reasoning:  # reasoning models can spend the whole budget thinking and return nothing
                payload.update({"reasoning_effort": "low", "max_completion_tokens": 3000})
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            if r.status_code == 400 and reasoning:  # provider rejected the extra settings: retry plainly
                payload.pop("reasoning_effort"); payload.pop("max_completion_tokens")
                r = requests.post(url, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            choice = r.json()["choices"][0]
            text = (choice["message"].get("content") or "").strip()
            if not text:
                raise RuntimeError(f"The model returned an empty reply (finish_reason={choice.get('finish_reason')}, "
                                   f"model={OPENAI_MODEL}). Try again, or set OPENAI_MODEL to a different model.")
            return text
    except requests.RequestException as e:
        hint = ""
        if getattr(e, "response", None) is not None and e.response.status_code == 404 and BACKEND == "openai":
            hint = (f" A 404 from an OpenAI-compatible API usually means the model name '{OPENAI_MODEL}' is not "
                    f"available (providers retire models). Check the provider's model list and set OPENAI_MODEL.")
        raise RuntimeError(f"The language model call failed ({BACKEND}, {model_name()}): {e}.{hint}")
    raise RuntimeError(f"Unknown LLM_BACKEND '{BACKEND}'")


def write(result: dict) -> dict:
    prompt, found = build_prompt(result)
    if BACKEND in ("none", "stub"):
        return {"prompt": prompt, "text": "", "check": None, "findings": found}
    text = _call(prompt)
    if not text.strip():
        raise RuntimeError(f"The model ({model_name()}) returned an empty reply.")
    check = validate(text, result, found)
    return {"prompt": prompt, "text": text, "check": check, "findings": found}
