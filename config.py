"""Shared configuration for the fpl-engine pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- League / season -------------------------------------------------------
LEAGUE = "ENG-Premier League"
SEASON = "2526"  # soccerdata short format for 2025-26

# --- FPL public API --------------------------------------------------------
FPL_BASE_URL = "https://fantasy.premierleague.com/api"
FPL_BOOTSTRAP_URL = f"{FPL_BASE_URL}/bootstrap-static/"
FPL_FIXTURES_URL = f"{FPL_BASE_URL}/fixtures/"
FPL_ELEMENT_SUMMARY_URL = FPL_BASE_URL + "/element-summary/{player_id}/"
FPL_EVENT_LIVE_URL = FPL_BASE_URL + "/event/{event_id}/live/"

# --- Understat ---------------------------------------------------------------
# Called directly via requests (see scripts/fetch_data.py) rather than through
# the understatapi package -- see the note in requirements.txt.
UNDERSTAT_BASE_URL = "https://understat.com"
UNDERSTAT_LEAGUE = "EPL"

# --- Team name mapping (FPL <-> Understat) ----------------------------------
# FPL's bootstrap `teams[].name` and Understat's match `title` fields spell
# several clubs differently. This is a best-effort static map for the
# 2025-26 Premier League; `scripts/calculate_metrics.match_team_name` falls
# back to fuzzy matching (and logs a warning) for anything not listed here,
# e.g. a club that was promoted/relegated in a season this map wasn't
# updated for.
FPL_TO_UNDERSTAT_TEAM = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Leeds": "Leeds",
    "Liverpool": "Liverpool",
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Sunderland": "Sunderland",
    "Spurs": "Tottenham",
    "West Ham": "West Ham",
    "Wolves": "Wolverhampton Wanderers",
}

# --- DefCon (Defensive Contribution) -----------------------------------------
# Mirrors FPL's own 2025-26 scoring rule for the Defensive Contribution bonus
# point, so this isn't an invented metric -- see calculate_metrics.compute_defcon
# for the exact formula and the raw FPL stat fields it reads.
DEFCON_THRESHOLD_DEF = 10   # CBIT  (clearances+blocks+interceptions+tackles)
DEFCON_THRESHOLD_MID_FWD = 12  # CBIRT (+ recoveries)

# --- GitHub publishing -------------------------------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "martinzaratel/fpl-metrics-data")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_METRICS_PATH = os.getenv("GITHUB_METRICS_PATH", "metrics.json")
