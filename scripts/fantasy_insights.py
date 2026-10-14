"""Fantasy-facing outputs built on top of calculate_metrics.py's per-player data.

All of these work from the accumulated data/processed/gw*_combined.csv
history (FPL + Understat), so none of them depend on FBref -- they're the
most reliable layer of the pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import calculate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fantasy_insights")


def value_picks(xgi_form_df: pd.DataFrame, latest_gw_df: pd.DataFrame, min_minutes: int = 180, top_n: int = 20) -> pd.DataFrame:
    """Best xGI-per-90-per-price picks, filtered by a minimum minutes floor.

        value_score = xgi_per90 / price_in_millions

    `xgi_form_df` is calculate_metrics.compute_player_xgi_form()'s output.
    `latest_gw_df` is the most recent data/processed/gw*_combined.csv (for
    current price and ownership).
    """
    if xgi_form_df.empty or latest_gw_df.empty:
        return pd.DataFrame()

    price_lookup = latest_gw_df.drop_duplicates("id").set_index("id")[["now_cost", "selected_by_percent"]]
    df = xgi_form_df.join(price_lookup, on="id")
    df = df[df["minutes_total"] >= min_minutes].copy()
    if df.empty:
        logger.warning("value_picks: no players met the %d-minute floor.", min_minutes)
        return pd.DataFrame()

    df["price_m"] = df["now_cost"] / 10
    df["value_score"] = (df["xgi_per90"] / df["price_m"]).round(3)
    cols = ["id", "full_name", "team_name", "position", "price_m", "xgi_per90", "selected_by_percent", "value_score"]
    return df[cols].sort_values("value_score", ascending=False).head(top_n).reset_index(drop=True)


def rotation_risk_alerts(n: int = 5, min_xgi_per90: float = 0.3, minutes_drop_ratio: float = 0.5) -> pd.DataFrame:
    """Flag players whose minutes are trending down despite good underlying performance.

    Splits each player's appearances in the last `n` gameweeks into an
    earlier half and a more recent half, and flags anyone whose
    recent-half average minutes have dropped below `minutes_drop_ratio`
    (50% by default) of their earlier-half average, while their overall
    xGI/90 across the window stays at or above `min_xgi_per90` -- i.e.
    they're still performing when they do play, but playing less. Needs
    at least 2 gameweeks of history for a player to be considered.
    """
    history = calculate_metrics.load_recent_gw_files(n)
    if history.empty:
        logger.warning("rotation_risk_alerts: no accumulated gameweek files in data/processed/.")
        return pd.DataFrame()
    history = calculate_metrics.attach_xgi_source(history)

    rows = []
    for (pid, name, team, pos), grp in history.groupby(["id", "full_name", "team_name", "position"]):
        grp = grp.sort_values("gameweek")
        if len(grp) < 2:
            continue
        mid = len(grp) // 2
        earlier, recent = grp.iloc[:mid], grp.iloc[mid:]
        earlier_min_avg = earlier["gw_minutes"].mean()
        recent_min_avg = recent["gw_minutes"].mean()
        if earlier_min_avg <= 0:
            continue

        minutes_total = grp["gw_minutes"].sum()
        xgi_per90 = (grp["xgi_source"].sum() / minutes_total * 90) if minutes_total else 0
        drop_ratio = recent_min_avg / earlier_min_avg

        if drop_ratio < minutes_drop_ratio and xgi_per90 >= min_xgi_per90:
            rows.append(
                {
                    "id": pid, "full_name": name, "team_name": team, "position": pos,
                    "earlier_minutes_avg": round(earlier_min_avg, 1),
                    "recent_minutes_avg": round(recent_min_avg, 1),
                    "minutes_drop_ratio": round(drop_ratio, 2),
                    "xgi_per90": round(xgi_per90, 2),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("minutes_drop_ratio").reset_index(drop=True)


def captaincy_ranking(
    xgi_form_df: pd.DataFrame, fdr_df: pd.DataFrame, min_minutes: int = 180, top_n: int = 10
) -> pd.DataFrame:
    """Suggested captaincy ranking for the next gameweek.

        captaincy_score = xgi_per90 * minutes_reliability / attack_fdr

    Where:
      - xgi_per90: the player's xGI form over the lookback window
        (calculate_metrics.compute_player_xgi_form).
      - minutes_reliability = min(1, minutes_total / (90 * gameweeks_counted)):
        penalizes players who've been heavily rotated even if their
        per-90 numbers look good.
      - attack_fdr: the player's team's custom attack FDR for their next
        fixture (calculate_metrics.compute_fdr; 1 = easiest, 5 = hardest).
        Teams with no resolved fixture default to FDR 3 (neutral).

    This rewards in-form attacking players with a favourable, minutes-secure
    fixture. It's a transparent heuristic, not a points-prediction model.
    """
    if xgi_form_df.empty:
        return pd.DataFrame()

    df = xgi_form_df[xgi_form_df["minutes_total"] >= min_minutes].copy()
    if df.empty:
        logger.warning("captaincy_ranking: no players met the %d-minute floor.", min_minutes)
        return pd.DataFrame()

    df["minutes_reliability"] = (df["minutes_total"] / (90 * df["gameweeks_counted"])).clip(upper=1.0)

    if not fdr_df.empty:
        team_attack_fdr = fdr_df.drop_duplicates("team").set_index("team")["attack_fdr"]
        df["attack_fdr"] = df["team_name"].map(team_attack_fdr).fillna(3.0)
    else:
        logger.warning("captaincy_ranking: no FDR data supplied; defaulting every team to neutral FDR 3.")
        df["attack_fdr"] = 3.0

    df["captaincy_score"] = (df["xgi_per90"] * df["minutes_reliability"] / df["attack_fdr"]).round(3)
    cols = ["id", "full_name", "team_name", "position", "xgi_per90", "minutes_reliability", "attack_fdr", "captaincy_score"]
    return df[cols].sort_values("captaincy_score", ascending=False).head(top_n).reset_index(drop=True)


def player_comparator(player_ids: list[int], n: int = 5) -> pd.DataFrame:
    """Side-by-side comparison of 2-3 players' totals/per-90 numbers over the last `n` gameweeks.

    Row order follows `player_ids`. For a match-by-match breakdown instead
    of totals, call calculate_metrics.load_recent_gw_files(n) directly and
    filter it yourself.
    """
    if not 2 <= len(player_ids) <= 3:
        logger.warning("player_comparator: expected 2-3 player ids, got %d.", len(player_ids))

    history = calculate_metrics.load_recent_gw_files(n)
    if history.empty:
        return pd.DataFrame()
    history = calculate_metrics.attach_xgi_source(history)
    subset = history[history["id"].isin(player_ids)]
    if subset.empty:
        return pd.DataFrame()

    agg = subset.groupby(["id", "full_name", "team_name", "position"], as_index=False).agg(
        gameweeks=("gameweek", "nunique"),
        minutes_total=("gw_minutes", "sum"),
        points_total=("gw_total_points", "sum"),
        goals_total=("gw_goals_scored", "sum"),
        assists_total=("gw_assists", "sum"),
        xgi_total=("xgi_source", "sum"),
    )
    agg["xgi_per90"] = (agg["xgi_total"] / agg["minutes_total"].replace(0, pd.NA) * 90).round(2)
    agg["points_per90"] = (agg["points_total"] / agg["minutes_total"].replace(0, pd.NA) * 90).round(2)

    present_ids = [pid for pid in player_ids if pid in agg["id"].values]
    return agg.set_index("id").loc[present_ids].reset_index()
