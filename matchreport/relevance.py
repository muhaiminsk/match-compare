"""Which statistics matter for which position, and when a change is big enough to mention.

A percentage alone is not enough: 1.0 tackle per 90 falling to 0 is a 100% drop and means nothing.
A change counts only if it clears BOTH an absolute threshold (per 90 minutes, or rating points) and a
percentage threshold, and only for statistics that matter for the player's position."""
import pandas as pd

ROLE_NAME = {"G": "goalkeeper", "D": "defender", "M": "midfielder", "F": "forward"}

# Statistics worth discussing for each position. Attackers are never judged on defending,
# centre-backs never on shooting, goalkeepers only on their rating.
RELEVANT = {
    "F": {"rating", "goals", "assists", "shots_on_target", "xg", "shots", "key_passes", "xa", "dribbles_completed", "xg_chain", "passes_accurate"},
    "M": {"rating", "passes", "passes_accurate", "key_passes", "xa", "xg_buildup", "xg_chain", "dribbles_completed",
          "duels_won", "tackles", "interceptions", "shots", "xg", "goals", "assists", "shots_on_target"},
    "D": {"rating", "tackles", "interceptions", "duels_won", "passes", "passes_accurate", "xg_buildup", "key_passes", "xa"},
    "G": {"rating"},
}

# metric: (minimum absolute change, minimum % change). Absolute values are per 90 minutes (rating: points).
SIGNIFICANCE = {
    "rating": (0.5, 0), "xg": (0.25, 40), "shots": (1.5, 40), "shots_on_target": (1.0, 50),
    "key_passes": (1.5, 40), "xa": (0.2, 40), "dribbles_completed": (1.5, 40), "xg_chain": (0.4, 40),
    "xg_buildup": (0.3, 40), "passes": (12, 15), "passes_accurate": (10, 15), "duels_won": (2.5, 30),
    "tackles": (1.5, 50), "interceptions": (1.5, 50), "goals": (0.25, 40), "assists": (0.2, 40), "xg_xa": (0.3, 40),
}

LABEL = {"rating": "rating", "goals": "goals per 90", "assists": "assists per 90", "xg": "xG per 90", "shots": "shots per 90", "shots_on_target": "shots on target per 90",
         "key_passes": "key passes per 90", "xa": "xA per 90", "dribbles_completed": "completed dribbles per 90",
         "xg_chain": "xGChain per 90", "xg_buildup": "xGBuildup per 90", "passes": "passes attempted per 90",
         "passes_accurate": "completed passes per 90", "duels_won": "duels won per 90", "tackles": "tackles per 90",
         "interceptions": "interceptions per 90"}
DECIMALS = {"rating": 2, "goals": 2, "assists": 2, "xg": 2, "xa": 2, "xg_chain": 2, "xg_buildup": 2, "xg_xa": 2}


def fmt(metric: str, v: float) -> str:
    return f"{v:.{DECIMALS.get(metric, 1)}f}"


def _role_from_code(code: str):
    code = str(code)
    if code in {"G", "D", "M", "F"}:
        return code
    if code == "GK":
        return "G"
    if code.startswith("DM") or code in {"MC", "ML", "MR", "AMC"}:
        return "M"
    if code.startswith("D"):
        return "D"
    if code.startswith("FW") or code in {"AML", "AMR"}:
        return "F"
    return None


def player_roles(pm: pd.DataFrame) -> dict:
    """player -> G/D/M/F: the role column when the source has it (Highlightly, API-Football), otherwise derived
    from Understat position codes; the most common role across the player's matches wins."""
    if pm.empty:
        return {}
    role = pm["role"] if "role" in pm.columns else pd.Series(None, index=pm.index)
    derived = role.where(role.isin(["G", "D", "M", "F"]), pm["position"].map(_role_from_code))
    d = pd.DataFrame({"player": pm["player"], "r": derived}).dropna()
    return d.groupby("player")["r"].agg(lambda s: s.mode().iat[0]).to_dict()


def relevant(role, metric: str) -> bool:
    """Unknown role: everything counts (better than silently dropping a player)."""
    if metric == "xg_xa":
        metric = "xg"
    return True if role is None else metric in RELEVANT.get(role, set())


def significant(metric: str, base: float, match: float) -> bool:
    if metric not in SIGNIFICANCE or pd.isna(base) or pd.isna(match):
        return False
    min_abs, min_pct = SIGNIFICANCE[metric]
    diff = abs(match - base)
    pct = abs(match / base - 1) * 100 if base else (100 if match else 0)
    return diff >= min_abs and pct >= min_pct


def strength(metric: str, base: float, match: float) -> float:
    """How many times the change exceeds its absolute threshold (used to rank what to mention first)."""
    return abs(match - base) / SIGNIFICANCE[metric][0] if metric in SIGNIFICANCE else 0.0
