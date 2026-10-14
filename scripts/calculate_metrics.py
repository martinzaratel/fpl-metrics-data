"""Compute Premier League / FPL metrics for a gameweek.

What's implemented, and the honest limits of each:

  - Team rolling xG/xGA (last N matches) -- from Understat's full-season
    match list. Reliable, Understat-only.
  - Custom Fixture Difficulty Rating (FDR) -- each opponent's rolling-6
    xG/xGA, split home/away, scaled 1-5. Understat + FPL fixtures.
  - DefCon (Defensive Contribution) -- mirrors FPL's own 2025-26 scoring
    rule exactly (see compute_defcon's docstring for the formula). Sourced
    from the FPL API fields fetch_data.py already pulls per gameweek.
  - Player xGI form (xG + xA) over the last N gameweeks -- from the
    accumulated data/processed/gw*_combined.csv files fetch_data.py writes.
  - PPDA (Passes Per Defensive Action) -- APPROXIMATE. soccerdata's FBref
    match-level API only exposes the "summary" table (whole-pitch tackles
    and interceptions; no zone split, no opponent-half passes), so this is
    a whole-pitch approximation of the canonical zone-restricted metric.
    See compute_ppda's docstring.
  - Average XI positions -- APPROXIMATE. Maps each FBref starting-lineup
    position tag to a static pitch-slot coordinate; this is a formation
    diagram, not a true touch-weighted average position (that needs
    FBref's Possession table, which soccerdata doesn't fetch at match
    level).
  - Attack zones (% of actions per pitch third) -- NOT IMPLEMENTED.
    soccerdata's FBref API doesn't expose the zone-split Possession table
    at match level, and there's no honest proxy for it from the other
    sources here. compute_attack_zones() returns an empty DataFrame and
    logs why, rather than faking a number.

All FBref-dependent functions are best-effort: FBref now requires a full
headless-browser session and is known to block automated/CI IPs, so every
call is wrapped to degrade gracefully rather than break the pipeline.
"""

from __future__ import annotations

import difflib
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from scripts import fetch_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("calculate_metrics")


# --------------------------------------------------------------------------
# Team name matching (FPL <-> Understat)
# --------------------------------------------------------------------------

def match_team_name(fpl_name: str, understat_titles: set[str]) -> str | None:
    """Map an FPL team name to its Understat title, static map first, fuzzy fallback second."""
    mapped = config.FPL_TO_UNDERSTAT_TEAM.get(fpl_name)
    if mapped in understat_titles:
        return mapped

    close = difflib.get_close_matches(fpl_name, list(understat_titles), n=1, cutoff=0.6)
    if close:
        logger.warning(
            "Team name fuzzy-matched: FPL '%s' -> Understat '%s'. Not in config.FPL_TO_UNDERSTAT_TEAM "
            "-- add it there if this is a new/promoted club.",
            fpl_name, close[0],
        )
        return close[0]

    logger.warning("Team name '%s' could not be matched to any Understat team.", fpl_name)
    return None


# --------------------------------------------------------------------------
# Team-level rolling xG/xGA + custom FDR (Understat)
# --------------------------------------------------------------------------

def fetch_season_matches(season: str) -> pd.DataFrame:
    """All finished matches this season, one row per match, from Understat."""
    try:
        data = fetch_data._understat_get(f"getLeagueData/{config.UNDERSTAT_LEAGUE}/{season}")
    except Exception:
        logger.exception("Understat: failed to fetch season match list for season %s", season)
        return pd.DataFrame()

    rows = []
    for m in data.get("dates", []):
        if not m.get("isResult"):
            continue
        try:
            rows.append(
                {
                    "match_id": m["id"],
                    "datetime": m["datetime"],
                    "home_team": m["h"]["title"],
                    "away_team": m["a"]["title"],
                    "home_xg": float(m["xG"]["h"]),
                    "away_xg": float(m["xG"]["a"]),
                    "home_goals": int(m["goals"]["h"]),
                    "away_goals": int(m["goals"]["a"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def to_team_match_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape one-row-per-match to one-row-per-team-per-match (for/against + venue)."""
    if matches.empty:
        return pd.DataFrame()

    cols = ["match_id", "datetime", "team", "opponent", "venue", "xg_for", "xg_against", "goals_for", "goals_against"]
    home = matches.rename(
        columns={
            "home_team": "team", "away_team": "opponent",
            "home_xg": "xg_for", "away_xg": "xg_against",
            "home_goals": "goals_for", "away_goals": "goals_against",
        }
    ).assign(venue="H")
    away = matches.rename(
        columns={
            "away_team": "team", "home_team": "opponent",
            "away_xg": "xg_for", "home_xg": "xg_against",
            "away_goals": "goals_for", "home_goals": "goals_against",
        }
    ).assign(venue="A")
    return pd.concat([home[cols], away[cols]], ignore_index=True).sort_values("datetime").reset_index(drop=True)


def rolling_team_form(team_long: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Each team's rolling xG/xGA over its last `n` matches (any venue)."""
    if team_long.empty:
        return pd.DataFrame()

    def _tail_stats(group: pd.DataFrame) -> pd.Series:
        tail = group.tail(n)
        return pd.Series(
            {
                "matches_considered": len(tail),
                "xg_for_total": tail["xg_for"].sum(),
                "xg_against_total": tail["xg_against"].sum(),
                "xg_for_per_match": tail["xg_for"].mean(),
                "xg_against_per_match": tail["xg_against"].mean(),
            }
        )

    return (
        team_long.sort_values("datetime")
        .groupby("team", group_keys=True)
        .apply(_tail_stats, include_groups=False)
        .reset_index()
    )


def rolling_team_form_by_venue(team_long: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """Each team's rolling xG/xGA over its last `n` matches, split by venue (for FDR)."""
    if team_long.empty:
        return pd.DataFrame()

    def _tail_stats(group: pd.DataFrame) -> pd.Series:
        tail = group.tail(n)
        return pd.Series(
            {
                "matches_considered": len(tail),
                "xg_for_avg": tail["xg_for"].mean(),
                "xg_against_avg": tail["xg_against"].mean(),
            }
        )

    return (
        team_long.sort_values("datetime")
        .groupby(["team", "venue"], group_keys=True)
        .apply(_tail_stats, include_groups=False)
        .reset_index()
    )


def _scale_1_5(series: pd.Series, invert: bool) -> pd.Series:
    """Percentile-rank a series to a 1-5 scale (5 = highest in the set)."""
    if series.isna().all() or series.nunique() <= 1:
        return pd.Series(3.0, index=series.index)
    ranked = series.rank(pct=True)
    if invert:
        ranked = 1 - ranked
    return (1 + ranked * 4).round(1)


def compute_fdr(
    team_long: pd.DataFrame,
    fixtures: list[dict],
    teams_lookup: dict[int, str],
    n: int = 6,
) -> pd.DataFrame:
    """Custom Fixture Difficulty Rating for a set of upcoming fixtures.

    For a fixture where `team` plays `opponent`, two separate 1-5 ratings
    (5 = hardest) are computed from the opponent's rolling last-`n`-match
    form AT THE VENUE THEY WILL BE PLAYING (i.e. if `team` is away, the
    opponent is at home, so we use the opponent's home form):

      - attack_fdr:   how hard it is for `team` to SCORE, driven by the
                       opponent's rolling xG AGAINST (low xGA conceded =
                       hard to score past them = high attack_fdr).
      - defence_fdr:  how hard it is for `team` to keep a clean sheet,
                       driven by the opponent's rolling xG FOR (high xG
                       scored by them = high defence_fdr).

    Both are percentile-scaled 1-5 across the fixtures passed in, not
    against the whole league -- pass all 20 teams' next fixture to rate
    a full gameweek consistently.
    """
    if team_long.empty or not fixtures:
        return pd.DataFrame()

    venue_form = rolling_team_form_by_venue(team_long, n=n)
    understat_titles = set(team_long["team"].unique())

    rows = []
    for fx in fixtures:
        home_name = teams_lookup.get(fx.get("team_h"))
        away_name = teams_lookup.get(fx.get("team_a"))
        if not home_name or not away_name:
            continue
        rows.append({"team": home_name, "opponent": away_name, "venue": "H", "event": fx.get("event")})
        rows.append({"team": away_name, "opponent": home_name, "venue": "A", "event": fx.get("event")})

    out = []
    for row in rows:
        opp_understat = match_team_name(row["opponent"], understat_titles)
        if opp_understat is None:
            continue
        opp_venue = "A" if row["venue"] == "H" else "H"  # opponent plays at the opposite venue
        opp_form = venue_form[(venue_form["team"] == opp_understat) & (venue_form["venue"] == opp_venue)]
        if opp_form.empty:
            continue
        out.append(
            {
                **row,
                "opponent_xg_for_avg": opp_form.iloc[0]["xg_for_avg"],
                "opponent_xg_against_avg": opp_form.iloc[0]["xg_against_avg"],
            }
        )

    fdr_df = pd.DataFrame(out)
    if fdr_df.empty:
        return fdr_df

    fdr_df["attack_fdr"] = _scale_1_5(fdr_df["opponent_xg_against_avg"], invert=True)
    fdr_df["defence_fdr"] = _scale_1_5(fdr_df["opponent_xg_for_avg"], invert=False)
    return fdr_df


# --------------------------------------------------------------------------
# DefCon (Defensive Contribution) -- FPL's own 2025-26 scoring rule
# --------------------------------------------------------------------------

def compute_defcon(gw_df: pd.DataFrame) -> pd.DataFrame:
    """Defensive Contribution, mirroring FPL's own 2025-26 bonus-point rule exactly.

    FPL awards 2 bonus points when a player's combined defensive actions in
    a single match reach a position-specific threshold:
        Defenders:            CBIT  = tackles + clearances_blocks_interceptions          >= 10
        Midfielders/Forwards: CBIRT = tackles + clearances_blocks_interceptions + recoveries >= 12
        Goalkeepers are not eligible for this bonus.

    If FPL's own `gw_defensive_contribution` field is present (FPL computes
    it server-side), it's trusted directly. Otherwise it's derived from the
    raw `gw_tackles` / `gw_clearances_blocks_interceptions` / `gw_recoveries`
    components using the formula above. Field names are assumed from FPL's
    2025-26 API; this is NOT yet verified against a live payload (see the
    module's "untested" notes) -- if FPL's schema differs, this logs a
    warning and returns an empty DataFrame rather than guessing.
    """
    if gw_df.empty:
        return pd.DataFrame()

    df = gw_df.copy()
    has_official = "gw_defensive_contribution" in df.columns and df["gw_defensive_contribution"].notna().any()

    if has_official:
        df["defcon_raw"] = pd.to_numeric(df["gw_defensive_contribution"], errors="coerce")
        logger.info("DefCon: using FPL's own gw_defensive_contribution field.")
    else:
        required = {"gw_tackles", "gw_clearances_blocks_interceptions"}
        if not required.issubset(df.columns):
            logger.warning("DefCon: raw component fields not found in this gameweek's data; skipping.")
            return pd.DataFrame()
        tackles = pd.to_numeric(df["gw_tackles"], errors="coerce").fillna(0)
        cbi = pd.to_numeric(df["gw_clearances_blocks_interceptions"], errors="coerce").fillna(0)
        recoveries = pd.to_numeric(df.get("gw_recoveries", 0), errors="coerce").fillna(0)
        is_def = df["position"] == "DEF"
        df["defcon_raw"] = tackles + cbi + recoveries.where(~is_def, 0)
        logger.info("DefCon: derived from raw tackles/CBI/recoveries components.")

    threshold = df["position"].map(
        {"DEF": config.DEFCON_THRESHOLD_DEF, "MID": config.DEFCON_THRESHOLD_MID_FWD, "FWD": config.DEFCON_THRESHOLD_MID_FWD}
    )
    df = df[df["position"].isin(["DEF", "MID", "FWD"])].copy()
    df["defcon_threshold"] = threshold
    df["defcon_per90"] = df["defcon_raw"] / df["gw_minutes"].replace(0, pd.NA) * 90
    df["defcon_threshold_pct"] = (df["defcon_raw"] / df["defcon_threshold"]).clip(upper=2.0)
    df["defcon_bonus_hit"] = df["defcon_raw"] >= df["defcon_threshold"]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Player xGI form over the last N gameweeks
# --------------------------------------------------------------------------

def load_recent_gw_files(n: int = 5) -> pd.DataFrame:
    """Concatenate the last `n` data/processed/gw*_combined.csv files written by fetch_data.py."""
    files = list(config.PROCESSED_DIR.glob("gw*_combined.csv"))
    gw_files = []
    for f in files:
        match = re.search(r"gw(\d+)_combined", f.stem)
        if match:
            gw_files.append((int(match.group(1)), f))
    gw_files.sort(key=lambda t: t[0])
    recent = gw_files[-n:]
    if not recent:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for _, f in recent], ignore_index=True)


def compute_player_xgi_form(n: int = 5) -> pd.DataFrame:
    """Player xGI (xG + xA) form, summed/averaged over the last `n` available gameweeks.

    Reads from the accumulated data/processed/gw*_combined.csv files, so it
    only covers gameweeks fetch_data.py has already been run for. Prefers
    Understat's xG/xA; falls back to FPL's own gw_expected_goal_involvements
    for player-gameweeks Understat didn't match.
    """
    history = load_recent_gw_files(n)
    if history.empty:
        logger.warning("xGI form: no accumulated gameweek files in data/processed/. Run fetch_data.py first.")
        return pd.DataFrame()

    understat_xgi = history.get("understat_xg", pd.Series(dtype=float)).fillna(0) + history.get(
        "understat_xa", pd.Series(dtype=float)
    ).fillna(0)
    history["xgi_source"] = understat_xgi

    if "gw_expected_goal_involvements" in history.columns:
        missing_understat = history.get("understat_xg", pd.Series(dtype=float)).isna() & history.get(
            "understat_xa", pd.Series(dtype=float)
        ).isna()
        history.loc[missing_understat, "xgi_source"] = history.loc[missing_understat, "gw_expected_goal_involvements"].fillna(0)

    agg = history.groupby(["id", "full_name", "team_name", "position"], as_index=False).agg(
        gameweeks_counted=("gameweek", "nunique"),
        minutes_total=("gw_minutes", "sum"),
        xgi_total=("xgi_source", "sum"),
        points_total=("gw_total_points", "sum"),
    )
    agg["xgi_per90"] = agg["xgi_total"] / agg["minutes_total"].replace(0, pd.NA) * 90
    return agg.sort_values("xgi_total", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# PPDA -- approximate (see module docstring)
# --------------------------------------------------------------------------

def compute_ppda(fbref_match_stats: pd.DataFrame) -> pd.DataFrame:
    """Approximate PPDA (Passes Per Defensive Action) per team per match.

    Canonical PPDA restricts both sides to the pressing team's defensive
    and middle thirds. soccerdata's FBref match-level API only exposes the
    whole-pitch "summary" table, so this computes:

        PPDA(team) = opponent's total passes attempted (that match)
                     / (team's tackles + interceptions + fouls, that match)

    This is a whole-pitch approximation, not the canonical zone-restricted
    metric, and will generally read LOWER (i.e. look like more pressing
    than reality) because the denominator includes defensive actions
    anywhere on the pitch. Returns an empty DataFrame if the expected
    FBref columns aren't present in the scraped table.
    """
    if fbref_match_stats.empty or "game_id" not in fbref_match_stats.columns:
        return pd.DataFrame()

    df = fbref_match_stats.copy()

    def _find_col(keywords: list[str]) -> str | None:
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in keywords):
                return c
        return None

    passes_col = _find_col(["passes_total_att", "total_att", "passes_att"])
    tkl_col = _find_col(["tackles_tkl"]) or _find_col(["tkl"])
    int_col = _find_col(["int"])
    fouls_col = _find_col(["fls", "fouls"])

    if not passes_col or not tkl_col:
        logger.warning("PPDA: required FBref columns (passes attempted / tackles) not found; skipping.")
        return pd.DataFrame()

    agg_spec = {"passes_attempted": (passes_col, "sum"), "tackles": (tkl_col, "sum")}
    team_match = df.groupby(["game_id", "team"], as_index=False).agg(**agg_spec)
    team_match["interceptions"] = df.groupby(["game_id", "team"])[int_col].sum().values if int_col else 0
    team_match["fouls"] = df.groupby(["game_id", "team"])[fouls_col].sum().values if fouls_col else 0

    rows = []
    for game_id, group in team_match.groupby("game_id"):
        if len(group) != 2:
            continue
        teams = group["team"].tolist()
        for i, team in enumerate(teams):
            opponent = teams[1 - i]
            own = group[group["team"] == team].iloc[0]
            opp_passes = group[group["team"] == opponent].iloc[0]["passes_attempted"]
            defensive_actions = own["tackles"] + own["interceptions"] + own["fouls"]
            ppda = opp_passes / defensive_actions if defensive_actions else float("nan")
            rows.append({"game_id": game_id, "team": team, "opponent": opponent, "ppda": ppda})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Average XI positions -- approximate formation diagram (see module docstring)
# --------------------------------------------------------------------------

# Static pitch-slot coordinates (mplsoccer StatsBomb 120x80 convention),
# keyed by the position tag FBref assigns each starter for that match.
# NOT true touch-weighted average positions -- see module docstring.
POSITION_COORDS = {
    "GK": (6, 40),
    "LB": (20, 10), "LWB": (28, 8),
    "CB": (18, 40), "RCB": (18, 52), "LCB": (18, 28),
    "RB": (20, 70), "RWB": (28, 72),
    "DM": (38, 40), "LDM": (38, 28), "RDM": (38, 52),
    "CM": (55, 40), "LCM": (55, 28), "RCM": (55, 52),
    "AM": (75, 40), "LAM": (75, 28), "RAM": (75, 52),
    "LM": (60, 12), "RM": (60, 68),
    "LW": (85, 12), "RW": (85, 68),
    "CF": (95, 40), "FW": (95, 40), "LF": (95, 25), "RF": (95, 55),
}
DEFAULT_COORD = (50, 40)


def compute_average_positions(start: datetime, end: datetime) -> pd.DataFrame:
    """Best-effort starting-XI formation slots for matches in a date window.

    See module docstring: approximate only, not a true average touch
    position.
    """
    try:
        import soccerdata as sd
    except ImportError:
        logger.warning("Average positions: soccerdata not installed, skipping.")
        return pd.DataFrame()

    try:
        fbref = sd.FBref(leagues=config.LEAGUE, seasons=config.SEASON)
        schedule = fbref.read_schedule().reset_index()
        schedule["date"] = pd.to_datetime(schedule["date"], utc=True)
        start_utc = pd.Timestamp(start, tz="UTC") if start.tzinfo is None else start
        end_utc = pd.Timestamp(end, tz="UTC") if end.tzinfo is None else end
        window = schedule[(schedule["date"] >= start_utc) & (schedule["date"] <= end_utc)]
        game_ids = window["game_id"].dropna().tolist()
        if not game_ids:
            logger.warning("Average positions: no matches found in the gameweek date window.")
            return pd.DataFrame()
        lineups = fbref.read_lineup(match_id=game_ids).reset_index()
    except Exception:
        logger.exception("Average positions: FBref lineup fetch failed (likely blocked).")
        return pd.DataFrame()

    starters = lineups[lineups["is_starter"] == True].copy()  # noqa: E712 (pandas bool column)
    coords = starters["position"].map(lambda p: POSITION_COORDS.get(p, DEFAULT_COORD))
    starters["x"] = coords.map(lambda c: c[0])
    starters["y"] = coords.map(lambda c: c[1])
    return starters[["game", "team", "player", "position", "x", "y", "minutes_played"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# Attack zones -- not implemented (see module docstring)
# --------------------------------------------------------------------------

def compute_attack_zones(*_args, **_kwargs) -> pd.DataFrame:
    """Not implemented: see the module docstring for why (no zone-split touch
    data is reachable through soccerdata's FBref API at match level)."""
    logger.warning(
        "Attack zones (%% actions by pitch third) is not implemented: soccerdata's FBref API doesn't "
        "expose the zone-split Possession table at match level, and there's no honest proxy for it "
        "from Understat or the FPL API. Returning an empty DataFrame."
    )
    return pd.DataFrame()
