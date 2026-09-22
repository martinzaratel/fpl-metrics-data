"""Fetch and combine raw data for a single FPL gameweek.

Sources, in priority order:
  1. FPL public API (bootstrap-static, fixtures, event/<gw>/live).
     Plain JSON, no bot protection. This is the backbone of the combined
     DataFrame (minutes, price, points, FPL's own xG/xA per GW).
  2. Understat, via `understatapi` (plain `requests`, CI-safe). Primary
     source for cross-checked xG/xA/shots at player-match level.
  3. FBref, via `soccerdata`. BEST EFFORT ONLY: FBref now requires a full
     headless-browser session (seleniumbase) and is known to block
     datacenter/CI IPs. Every call here is wrapped so a block or scraping
     failure never takes down the rest of the pipeline -- it just leaves
     the fbref_* columns empty.

Player-name matching across sources is done on a normalized name key
(ASCII-folded, lowercased, punctuation stripped). This is a best-effort
join, not a guaranteed one: FPL, Understat and FBref sometimes spell the
same player differently. Unmatched rows keep their FPL data with NaNs in
the understat_*/fbref_* columns rather than being dropped.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from unidecode import unidecode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fetch_data")


def normalize_name(name: str) -> str:
    """Fold a player name to a matchable key: ascii, lowercase, no punctuation."""
    if not isinstance(name, str):
        return ""
    name = unidecode(name).lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return re.sub(r"\s+", " ", name).strip()


# --------------------------------------------------------------------------
# FPL
# --------------------------------------------------------------------------

def fetch_fpl_bootstrap() -> dict:
    resp = requests.get(config.FPL_BOOTSTRAP_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_fpl_fixtures() -> list[dict]:
    resp = requests.get(config.FPL_FIXTURES_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_fpl_event_live(event_id: int) -> dict:
    url = config.FPL_EVENT_LIVE_URL.format(event_id=event_id)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def pick_target_gameweek(bootstrap: dict, gw_override: int | None = None) -> int:
    if gw_override is not None:
        return gw_override
    finished = [e for e in bootstrap["events"] if e["finished"]]
    if not finished:
        raise RuntimeError("No finished gameweeks yet this season; pass --gw explicitly.")
    return max(e["id"] for e in finished)


def get_gw_fixture_window(fixtures: list[dict], gw: int) -> tuple[datetime, datetime]:
    """Return a padded [start, end] UTC datetime window covering all of a GW's kickoffs."""
    gw_fixtures = [f for f in fixtures if f["event"] == gw and f.get("kickoff_time")]
    if not gw_fixtures:
        raise RuntimeError(f"No fixtures with a kickoff time found for gameweek {gw}.")
    kickoffs = [datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00")) for f in gw_fixtures]
    return min(kickoffs) - timedelta(hours=6), max(kickoffs) + timedelta(hours=6)


def build_fpl_gw_dataframe(bootstrap: dict, live: dict, gw: int) -> pd.DataFrame:
    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name", "short_name"]].rename(
        columns={"id": "team_id", "name": "team_name", "short_name": "team_short"}
    )
    positions = pd.DataFrame(bootstrap["element_types"])[["id", "singular_name_short"]].rename(
        columns={"id": "element_type", "singular_name_short": "position"}
    )

    players = players.merge(teams, left_on="team", right_on="team_id", how="left")
    players = players.merge(positions, on="element_type", how="left")

    live_rows = [{"id": e["id"], **e["stats"]} for e in live["elements"]]
    live_df = pd.DataFrame(live_rows).add_prefix("gw_").rename(columns={"gw_id": "id"})

    df = players.merge(live_df, on="id", how="inner")
    df = df[df["gw_minutes"] > 0].copy()

    df["full_name"] = (df["first_name"] + " " + df["second_name"]).str.strip()
    df["name_key"] = df["full_name"].apply(normalize_name)
    df["gameweek"] = gw

    numeric_cols = [
        "gw_expected_goals", "gw_expected_assists", "gw_expected_goal_involvements",
        "gw_expected_goals_conceded", "gw_ict_index", "selected_by_percent",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = [
        "id", "full_name", "web_name", "name_key", "team_name", "team_short",
        "position", "now_cost", "selected_by_percent", "gameweek",
        "gw_minutes", "gw_starts", "gw_total_points", "gw_goals_scored", "gw_assists",
        "gw_expected_goals", "gw_expected_assists", "gw_expected_goal_involvements",
        "gw_expected_goals_conceded", "gw_bonus", "gw_bps", "gw_ict_index",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# Understat (primary xG/xA cross-check; plain requests, no browser needed)
# --------------------------------------------------------------------------

def fetch_understat_gw(season: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        import understatapi
    except ImportError:
        logger.warning("Understat: understatapi not installed, skipping.")
        return pd.DataFrame()

    client = understatapi.UnderstatClient()
    try:
        matches = client.league(league=config.UNDERSTAT_LEAGUE).get_match_data(season=season)
    except Exception:
        logger.exception("Understat: failed to fetch league match list for season %s", season)
        return pd.DataFrame()

    target_matches = []
    for m in matches:
        if not m.get("isResult"):
            continue
        try:
            match_dt = datetime.fromisoformat(m["datetime"])
        except (KeyError, ValueError, TypeError):
            continue
        if start <= match_dt <= end:
            target_matches.append(m)

    if not target_matches:
        logger.warning("Understat: no matches fell inside the gameweek date window %s -> %s", start, end)
        return pd.DataFrame()

    rows = []
    for m in target_matches:
        match_id = m["id"]
        try:
            roster = client.match(match=match_id).get_roster_data()
        except Exception:
            logger.exception("Understat: failed to fetch roster for match id=%s", match_id)
            continue
        for side_key in ("h", "a"):
            for player in roster.get(side_key, {}).values():
                rows.append(
                    {
                        "player": player.get("player"),
                        "minutes": player.get("time"),
                        "understat_xg": player.get("xG"),
                        "understat_xa": player.get("xA"),
                        "understat_shots": player.get("shots"),
                        "understat_key_passes": player.get("key_passes"),
                        "match_id": match_id,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ("understat_xg", "understat_xa", "understat_shots", "understat_key_passes"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["name_key"] = df["player"].apply(normalize_name)

    return df.groupby("name_key", as_index=False).agg(
        understat_xg=("understat_xg", "sum"),
        understat_xa=("understat_xa", "sum"),
        understat_shots=("understat_shots", "sum"),
        understat_key_passes=("understat_key_passes", "sum"),
    )


# --------------------------------------------------------------------------
# FBref (best-effort only -- see module docstring)
# --------------------------------------------------------------------------

def fetch_fbref_gw(start: datetime, end: datetime) -> pd.DataFrame:
    try:
        import soccerdata as sd
    except ImportError:
        logger.warning("FBref: soccerdata not installed, skipping.")
        return pd.DataFrame()

    try:
        fbref = sd.FBref(leagues=config.LEAGUE, seasons=config.SEASON)
        schedule = fbref.read_schedule().reset_index()
    except Exception:
        logger.exception("FBref: could not load the schedule (likely blocked, or no browser available).")
        return pd.DataFrame()

    schedule["date"] = pd.to_datetime(schedule["date"], utc=True)
    start_utc = pd.Timestamp(start, tz="UTC") if start.tzinfo is None else start
    end_utc = pd.Timestamp(end, tz="UTC") if end.tzinfo is None else end
    window = schedule[(schedule["date"] >= start_utc) & (schedule["date"] <= end_utc)]
    game_ids = window["game_id"].dropna().tolist()
    if not game_ids:
        logger.warning("FBref: no matches found in the gameweek date window.")
        return pd.DataFrame()

    try:
        stats = fbref.read_player_match_stats(stat_type="summary", match_id=game_ids).reset_index()
    except Exception:
        logger.exception("FBref: failed to fetch player match stats (likely blocked mid-scrape).")
        return pd.DataFrame()

    stats["name_key"] = stats["player"].apply(normalize_name)
    signal_cols = [
        c for c in stats.columns
        if any(k in c.lower() for k in ("tkl", "int", "blocks", "clr", "prgp", "prgc", "sca", "gca", "xg", "xa"))
    ]
    if not signal_cols:
        logger.warning("FBref: expected stat columns not found in scraped table; skipping merge.")
        return pd.DataFrame()

    agg = stats.groupby("name_key", as_index=False)[signal_cols].sum(numeric_only=True)
    return agg.add_prefix("fbref_").rename(columns={"fbref_name_key": "name_key"})


# --------------------------------------------------------------------------
# Combine
# --------------------------------------------------------------------------

def combine(fpl_df: pd.DataFrame, understat_df: pd.DataFrame, fbref_df: pd.DataFrame) -> pd.DataFrame:
    df = fpl_df.copy()
    if not understat_df.empty:
        df = df.merge(understat_df, on="name_key", how="left")
    if not fbref_df.empty:
        df = df.merge(fbref_df, on="name_key", how="left")
    return df


def main() -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Fetch and combine one FPL gameweek of data.")
    parser.add_argument("--gw", type=int, default=None, help="Gameweek to fetch. Defaults to latest finished GW.")
    parser.add_argument("--skip-fbref", action="store_true", help="Skip FBref entirely (fastest, most reliable).")
    args = parser.parse_args()

    logger.info("Fetching FPL bootstrap-static ...")
    bootstrap = fetch_fpl_bootstrap()
    gw = pick_target_gameweek(bootstrap, args.gw)
    logger.info("Target gameweek: GW%d", gw)

    fixtures = fetch_fpl_fixtures()
    start, end = get_gw_fixture_window(fixtures, gw)
    logger.info("Gameweek date window: %s -> %s", start, end)

    live = fetch_fpl_event_live(gw)
    fpl_df = build_fpl_gw_dataframe(bootstrap, live, gw)
    logger.info("FPL: %d players with minutes in GW%d", len(fpl_df), gw)

    (config.RAW_DIR / f"fpl_bootstrap_gw{gw}.json").write_text(json.dumps(bootstrap))
    (config.RAW_DIR / f"fpl_live_gw{gw}.json").write_text(json.dumps(live))

    understat_season = str(int(config.SEASON[:2]) + 2000)  # "2526" -> "2025"
    logger.info("Fetching Understat data for season %s ...", understat_season)
    understat_df = fetch_understat_gw(understat_season, start, end)
    logger.info("Understat: matched stats for %d players", len(understat_df))

    fbref_df = pd.DataFrame()
    if not args.skip_fbref:
        logger.info("Attempting FBref (best-effort, may fail or get blocked) ...")
        fbref_df = fetch_fbref_gw(start, end)
        logger.info("FBref: matched stats for %d players", len(fbref_df))

    combined = combine(fpl_df, understat_df, fbref_df)

    out_path = config.PROCESSED_DIR / f"gw{gw}_combined.csv"
    combined.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows x %d cols)", out_path, *combined.shape)

    understat_match_rate = combined["understat_xg"].notna().mean() if "understat_xg" in combined else 0.0
    logger.info("Understat match rate: %.0f%%", understat_match_rate * 100)

    print(combined.head(15).to_string())
    return combined


if __name__ == "__main__":
    main()
