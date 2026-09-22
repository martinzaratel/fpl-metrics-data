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

# --- Understat --------------------------------------------------------------
UNDERSTAT_LEAGUE = "EPL"

# --- GitHub publishing -------------------------------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "martinzaratel/fpl-metrics-data")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_METRICS_PATH = os.getenv("GITHUB_METRICS_PATH", "metrics.json")
