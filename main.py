"""Entry point: run the full FPL metrics pipeline for one gameweek.

    fetch_data  ->  calculate_metrics  ->  fantasy_insights
                 ->  generate_visuals (best-effort)
                 ->  metrics.json  ->  publish_to_github

Every stage past step 1 (the FPL+Understat fetch/combine, the most
reliable part of the pipeline) is wrapped so a failure there degrades
that stage's output rather than aborting the run -- see each module's own
docstring for exactly what's reliable vs. best-effort/approximate, and
the project's known-limitations notes for what hasn't been exercised
against live data yet (network access to fbref.com/understat.com/
fantasy.premierleague.com has been unavailable throughout development).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config
from scripts import calculate_metrics, fantasy_insights, fetch_data, generate_visuals, publish_to_github

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("main")


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of plain-Python dicts, NaN -> null, via pandas' own
    JSON encoder (handles numpy dtypes correctly, unlike a bare .to_dict())."""
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def build_metrics_payload(
    gw: int,
    combined: pd.DataFrame,
    team_form: pd.DataFrame,
    fdr_df: pd.DataFrame,
    defcon_df: pd.DataFrame,
    xgi_form_df: pd.DataFrame,
    value_picks_df: pd.DataFrame,
    rotation_df: pd.DataFrame,
    captaincy_df: pd.DataFrame,
    visual_paths: list[Path],
) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": f"20{config.SEASON[:2]}-{config.SEASON[2:]}",
        "gameweek": gw,
        "players": _records(combined),
        "team_form_last5": _records(team_form),
        "fixture_difficulty_next_gw": _records(fdr_df),
        "defcon": _records(defcon_df),
        "player_xgi_form": _records(xgi_form_df),
        "fantasy": {
            "value_picks": _records(value_picks_df),
            "rotation_risk_alerts": _records(rotation_df),
            "captaincy_ranking": _records(captaincy_df),
        },
        "visuals": [p.name for p in visual_paths],
    }


def run_pipeline(gw: int | None = None, skip_fbref: bool = False, publish: bool = True) -> dict:
    logger.info("=== FPL Metrics Pipeline: starting ===")

    # --- 1. Fetch + combine this gameweek (FPL + Understat, FBref best-effort) ---
    bootstrap = fetch_data.fetch_fpl_bootstrap()
    target_gw = fetch_data.pick_target_gameweek(bootstrap, gw)
    fixtures = fetch_data.fetch_fpl_fixtures()
    start, end = fetch_data.get_gw_fixture_window(fixtures, target_gw)
    live = fetch_data.fetch_fpl_event_live(target_gw)
    fpl_df = fetch_data.build_fpl_gw_dataframe(bootstrap, live, target_gw)

    understat_season = str(int(config.SEASON[:2]) + 2000)
    understat_df = fetch_data.fetch_understat_gw(understat_season, start, end)
    fbref_df = pd.DataFrame() if skip_fbref else fetch_data.fetch_fbref_gw(start, end)
    combined = fetch_data.combine(fpl_df, understat_df, fbref_df)

    (config.RAW_DIR / f"fpl_bootstrap_gw{target_gw}.json").write_text(json.dumps(bootstrap))
    (config.RAW_DIR / f"fpl_live_gw{target_gw}.json").write_text(json.dumps(live))
    combined.to_csv(config.PROCESSED_DIR / f"gw{target_gw}_combined.csv", index=False)
    logger.info("Step 1/5 done: fetched + combined GW%d (%d players with minutes).", target_gw, len(combined))

    # --- 2. Metrics: team form, custom FDR for next GW, DefCon, player xGI form ---
    matches = calculate_metrics.fetch_season_matches(understat_season)
    team_long = calculate_metrics.to_team_match_long(matches)
    team_form = calculate_metrics.rolling_team_form(team_long, n=5)

    teams_lookup = {t["id"]: t["name"] for t in bootstrap["teams"]}
    next_gw_fixtures = [f for f in fixtures if f.get("event") == target_gw + 1]
    fdr_df = calculate_metrics.compute_fdr(team_long, next_gw_fixtures, teams_lookup, n=6)

    defcon_df = calculate_metrics.compute_defcon(combined)
    xgi_form_df = calculate_metrics.compute_player_xgi_form(n=5)
    logger.info("Step 2/5 done: team form (%d teams), FDR (%d fixture-sides), DefCon (%d players), xGI form (%d players).",
                len(team_form), len(fdr_df), len(defcon_df), len(xgi_form_df))

    # --- 3. Fantasy insights ---
    value_picks_df = fantasy_insights.value_picks(xgi_form_df, combined, min_minutes=180)
    rotation_df = fantasy_insights.rotation_risk_alerts(n=5)
    captaincy_df = fantasy_insights.captaincy_ranking(xgi_form_df, fdr_df, min_minutes=180)
    logger.info("Step 3/5 done: %d value picks, %d rotation-risk alerts, %d captaincy candidates.",
                len(value_picks_df), len(rotation_df), len(captaincy_df))

    # --- 4. Visuals (best-effort end to end -- see generate_visuals.generate_all_visuals) ---
    visual_paths: list[Path] = []
    try:
        visual_paths = generate_visuals.generate_all_visuals(target_gw)
    except Exception:
        logger.exception("Step 4/5: visual generation failed entirely; continuing without visuals.")
    logger.info("Step 4/5 done: %d visuals written.", len(visual_paths))

    # --- 5. Build metrics.json, write locally, publish to GitHub ---
    payload = build_metrics_payload(
        target_gw, combined, team_form, fdr_df, defcon_df, xgi_form_df,
        value_picks_df, rotation_df, captaincy_df, visual_paths,
    )
    metrics_path = config.PROCESSED_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("metrics.json written locally to %s", metrics_path)

    if publish:
        if publish_to_github.publish_metrics(payload):
            logger.info("Step 5/5 done: published to GitHub.")
        else:
            logger.error(
                "Step 5/5: GitHub publish FAILED -- see the error(s) above for exactly what went wrong. "
                "metrics.json is still saved locally at %s and the rest of the pipeline completed normally.",
                metrics_path,
            )
    else:
        logger.info("Step 5/5 skipped (--no-publish).")

    logger.info("=== FPL Metrics Pipeline: finished (GW%d) ===", target_gw)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full FPL metrics pipeline for one gameweek.")
    parser.add_argument("--gw", type=int, default=None, help="Gameweek to run. Defaults to the latest finished GW.")
    parser.add_argument("--skip-fbref", action="store_true", help="Skip FBref entirely (fastest, most reliable).")
    parser.add_argument("--no-publish", action="store_true", help="Write metrics.json locally but skip the GitHub push.")
    args = parser.parse_args()

    run_pipeline(gw=args.gw, skip_fbref=args.skip_fbref, publish=not args.no_publish)


if __name__ == "__main__":
    main()
