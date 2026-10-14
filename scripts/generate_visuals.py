"""Branded visualizations for a gameweek, built with mplsoccer/matplotlib.

Brand palette: adapted from this project's validated default data-viz
palette (see the dataviz skill's references/palette.md) -- the hex values
below are used unchanged from that reference, so no further CVD/contrast
validation is needed. Swap BRAND's values to re-skin the whole module.

Every `plot_*` function takes an already-prepared DataFrame (so each is
independently testable with synthetic data) and writes a PNG plus the
underlying data as CSV, sized at >=1080x1080px.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from scripts import calculate_metrics, fetch_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate_visuals")

# --------------------------------------------------------------------------
# Brand
# --------------------------------------------------------------------------

BRAND = {
    "bg": "#fcfcfb",
    "ink_primary": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "gridline": "#e1e0d9",
    "baseline": "#c3c2b7",
    "team_a": "#2a78d6",   # categorical slot 1 (blue)
    "team_b": "#eb6834",   # categorical slot 2 (orange)
    "accent": "#6250d6",   # categorical slot 7 (violet) -- this project's signature color
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
    "diverging_cold": "#2a78d6",
    "diverging_hot": "#e34948",
    "diverging_mid": "#898781",
}

WATERMARK_TEXT = "FPL METRICS ENGINE"


def apply_brand_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "figure.facecolor": BRAND["bg"],
            "axes.facecolor": BRAND["bg"],
            "axes.edgecolor": BRAND["baseline"],
            "axes.labelcolor": BRAND["ink_secondary"],
            "text.color": BRAND["ink_primary"],
            "xtick.color": BRAND["ink_muted"],
            "ytick.color": BRAND["ink_muted"],
            "grid.color": BRAND["gridline"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )


def add_watermark(fig: plt.Figure) -> None:
    fig.text(0.99, 0.01, WATERMARK_TEXT, ha="right", va="bottom", fontsize=8, color=BRAND["ink_muted"], alpha=0.6, fontweight="bold")


def save_figure(fig: plt.Figure, out_path: Path, min_px: int = 1080) -> None:
    """Save at a DPI that guarantees >= min_px in both dimensions.

    `bbox_inches="tight"` crops the canvas *after* the DPI is chosen, which
    can shrink the final image below the figsize-implied pixel count -- so
    this targets a 25% margin over min_px to stay clear of that crop, then
    verifies the actual saved size and re-saves once at a higher DPI in the
    rare case that isn't enough (e.g. an unusually tall legend/title).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w_in, h_in = fig.get_size_inches()
    dpi = max(150, int(min_px * 1.25 / min(w_in, h_in)) + 1)
    fig.savefig(out_path, dpi=dpi, facecolor=BRAND["bg"], bbox_inches="tight")

    actual_w, actual_h = _png_dimensions(out_path)
    if actual_w and (actual_w < min_px or actual_h < min_px):
        scale = min_px / min(actual_w, actual_h) * 1.1
        dpi = int(dpi * scale) + 1
        fig.savefig(out_path, dpi=dpi, facecolor=BRAND["bg"], bbox_inches="tight")
        actual_w, actual_h = _png_dimensions(out_path)

    plt.close(fig)
    logger.info("Saved %s (dpi=%d, %sx%s px)", out_path, dpi, actual_w, actual_h)


def _png_dimensions(path: Path) -> tuple[int, int] | tuple[None, None]:
    """Read a PNG's pixel dimensions from its IHDR chunk, no Pillow dependency."""
    try:
        with open(path, "rb") as f:
            header = f.read(24)
        if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
            return None, None
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        return width, height
    except OSError:
        return None, None


def export_raw(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


# --------------------------------------------------------------------------
# Average positions + xG/PPDA per match
# --------------------------------------------------------------------------

def plot_average_positions(positions_df: pd.DataFrame, match_meta: dict, out_path: Path) -> None:
    """Starting-XI formation diagram for one match (see calculate_metrics.compute_average_positions
    for why this is an approximate formation slot, not a true average touch position), with
    the match's xG and PPDA for each team as a subtitle.

    `positions_df` columns: team, player, x, y (StatsBomb 120x80 coordinates).
    `match_meta`: {"home_team", "away_team", "home_xg", "away_xg", "home_ppda", "away_ppda"}.
    """
    apply_brand_style()
    # Generous padding so player-name labels near the touchline have room
    # inside the axes data limits -- otherwise they can get cropped by
    # savefig's tight-bbox pass regardless of clip_on.
    pitch = Pitch(pitch_type="statsbomb", pitch_color=BRAND["bg"], line_color=BRAND["baseline"], pad_left=10, pad_right=10, pad_top=6, pad_bottom=6)
    fig, ax = pitch.draw(figsize=(10, 7.5))

    home, away = match_meta["home_team"], match_meta["away_team"]
    home_pts = positions_df[positions_df["team"] == home]
    away_pts = positions_df[positions_df["team"] == away]

    pitch.scatter(home_pts["x"], home_pts["y"], ax=ax, s=400, color=BRAND["team_a"], edgecolors=BRAND["bg"], linewidth=2, label=home, zorder=3)
    pitch.scatter(120 - away_pts["x"], 80 - away_pts["y"], ax=ax, s=400, color=BRAND["team_b"], edgecolors=BRAND["bg"], linewidth=2, label=away, zorder=3)

    # Labels sit below each marker in dark ink rather than inside it: the
    # markers are too small to fit most surnames, and white text extending
    # past a small colored circle onto the near-white pitch background
    # becomes invisible.
    for _, row in home_pts.iterrows():
        ax.annotate(row["player"].split()[-1], (row["x"], row["y"] - 4.5), ha="center", va="top", fontsize=7, color=BRAND["ink_primary"], zorder=4, clip_on=False)
    for _, row in away_pts.iterrows():
        ax.annotate(row["player"].split()[-1], (120 - row["x"], 80 - row["y"] - 4.5), ha="center", va="top", fontsize=7, color=BRAND["ink_primary"], zorder=4, clip_on=False)

    subtitle = (
        f"xG {match_meta.get('home_xg', float('nan')):.2f} - {match_meta.get('away_xg', float('nan')):.2f}  |  "
        f"PPDA (approx.) {match_meta.get('home_ppda', float('nan')):.1f} - {match_meta.get('away_ppda', float('nan')):.1f}"
    )
    ax.set_title(f"{home} vs {away}", fontsize=16, color=BRAND["ink_primary"], pad=28)
    fig.text(0.5, 0.93, subtitle, ha="center", fontsize=11, color=BRAND["ink_secondary"])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, labelcolor=BRAND["ink_secondary"])

    add_watermark(fig)
    save_figure(fig, out_path)
    export_raw(positions_df, out_path.with_suffix(".csv"))


# --------------------------------------------------------------------------
# Shot maps
# --------------------------------------------------------------------------

def plot_shot_map(shots_df: pd.DataFrame, match_meta: dict, out_path: Path) -> None:
    """Shot map for one match, from Understat's getMatchData shots payload.

    `shots_df` columns: x, y (Understat's 0-1 normalized coords, attacking
    goal at x=1), xG, player, team, result ("Goal" or anything else).
    `match_meta`: {"home_team", "away_team"}.
    """
    apply_brand_style()
    pitch = Pitch(pitch_type="statsbomb", pitch_color=BRAND["bg"], line_color=BRAND["baseline"], half=False)
    fig, ax = pitch.draw(figsize=(10, 7.5))

    home, away = match_meta["home_team"], match_meta["away_team"]
    df = shots_df.copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["xG"] = pd.to_numeric(df["xG"], errors="coerce")

    # Understat: x=1 is always the shooting team's attacking end. Home team
    # shoots left-to-right (as-is); away team's x/y are mirrored so both
    # shot clouds point at their true attacking goal on a single pitch.
    home_shots = df[df["team"] == home].copy()
    away_shots = df[df["team"] == away].copy()
    home_shots["px"], home_shots["py"] = home_shots["x"] * 120, home_shots["y"] * 80
    away_shots["px"], away_shots["py"] = (1 - away_shots["x"]) * 120, (1 - away_shots["y"]) * 80

    for shots, color, label in ((home_shots, BRAND["team_a"], home), (away_shots, BRAND["team_b"], away)):
        goals = shots[shots["result"] == "Goal"]
        others = shots[shots["result"] != "Goal"]
        pitch.scatter(others["px"], others["py"], ax=ax, s=others["xG"] * 900 + 60, color=color, edgecolors=BRAND["ink_primary"], alpha=0.75, linewidth=0.5, label=label, zorder=3)
        pitch.scatter(goals["px"], goals["py"], ax=ax, s=goals["xG"] * 900 + 60, color=color, edgecolors=BRAND["ink_primary"], alpha=0.95, linewidth=1.8, marker="*", zorder=4)

    total_xg_home, total_xg_away = home_shots["xG"].sum(), away_shots["xG"].sum()
    ax.set_title(f"{home} {total_xg_home:.2f}xG - {total_xg_away:.2f}xG {away}", fontsize=16, color=BRAND["ink_primary"], pad=28)
    fig.text(0.5, 0.93, "Marker size = shot xG  |  ★ = goal", ha="center", fontsize=10, color=BRAND["ink_secondary"])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, labelcolor=BRAND["ink_secondary"])

    add_watermark(fig)
    save_figure(fig, out_path)
    export_raw(shots_df, out_path.with_suffix(".csv"))


# --------------------------------------------------------------------------
# Top-10 xGI vs FPL points
# --------------------------------------------------------------------------

def plot_xgi_vs_points(xgi_form_df: pd.DataFrame, out_path: Path, top_n: int = 10) -> None:
    """Top-N players by xGI form, as two side-by-side panels (xGI, then actual FPL
    points) sharing the same player order -- never a dual-axis chart, since the two
    measures live on different scales.

    `xgi_form_df`: calculate_metrics.compute_player_xgi_form()'s output.
    """
    apply_brand_style()
    df = xgi_form_df.sort_values("xgi_total", ascending=False).head(top_n).iloc[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    y = range(len(df))

    ax1.barh(y, df["xgi_total"], color=BRAND["accent"], height=0.6)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(df["full_name"], fontsize=10, color=BRAND["ink_primary"])
    ax1.set_title("xGI (xG + xA)", fontsize=12, color=BRAND["ink_secondary"])
    ax1.grid(axis="x", color=BRAND["gridline"])
    ax1.spines["left"].set_visible(False)

    ax2.barh(y, df["points_total"], color=BRAND["team_a"], height=0.6)
    ax2.set_title("Actual FPL points", fontsize=12, color=BRAND["ink_secondary"])
    ax2.grid(axis="x", color=BRAND["gridline"])
    ax2.spines["left"].set_visible(False)

    fig.suptitle(f"Top {top_n} xGI form vs. FPL points", fontsize=16, color=BRAND["ink_primary"])
    add_watermark(fig)
    save_figure(fig, out_path)
    export_raw(df, out_path.with_suffix(".csv"))


# --------------------------------------------------------------------------
# PPDA summary
# --------------------------------------------------------------------------

def plot_ppda_bar(ppda_df: pd.DataFrame, out_path: Path) -> None:
    """Per-team PPDA for the gameweek (approximate -- see calculate_metrics.compute_ppda),
    lower = more aggressive pressing, sorted ascending (most pressing team first)."""
    apply_brand_style()
    df = ppda_df.dropna(subset=["ppda"]).groupby("team", as_index=False)["ppda"].mean().sort_values("ppda")
    if df.empty:
        logger.warning("plot_ppda_bar: no PPDA data to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, max(6, 0.4 * len(df))))
    ax.barh(df["team"], df["ppda"], color=BRAND["accent"], height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("PPDA (approx.) -- lower = more pressing")
    ax.set_title("Passes Per Defensive Action", fontsize=16, color=BRAND["ink_primary"])
    ax.grid(axis="x", color=BRAND["gridline"])
    ax.spines["left"].set_visible(False)

    add_watermark(fig)
    save_figure(fig, out_path)
    export_raw(df, out_path.with_suffix(".csv"))


# --------------------------------------------------------------------------
# Orchestrator -- wires the plot_* functions to live data for one gameweek.
#
# Every step beyond the FPL-only pieces (shot maps, average positions, PPDA)
# depends on Understat/FBref, so each is wrapped independently: one failing
# (or blocked) data source skips that visual rather than aborting the rest.
# This function has NOT been exercised against live data -- network access
# to fbref.com/understat.com has been unavailable throughout development
# (see the project's known-limitations notes). Each plot_* function it
# calls has been smoke-tested with synthetic data; this wiring has only
# been compile- and import-checked.
# --------------------------------------------------------------------------

def generate_all_visuals(gw: int, out_dir: Path | None = None) -> list[Path]:
    out_dir = Path(out_dir) if out_dir else config.PROCESSED_DIR / f"visuals_gw{gw}"
    written: list[Path] = []

    fixtures = fetch_data.fetch_fpl_fixtures()
    start, end = fetch_data.get_gw_fixture_window(fixtures, gw)

    # xGI vs points doesn't need Understat/FBref beyond what's already in
    # the accumulated per-gameweek history, so it's the most reliable visual.
    try:
        xgi_form = calculate_metrics.compute_player_xgi_form(n=5)
        if not xgi_form.empty:
            path = out_dir / f"gw{gw}_top10_xgi_vs_points.png"
            plot_xgi_vs_points(xgi_form, path, top_n=10)
            written.append(path)
    except Exception:
        logger.exception("generate_all_visuals: xGI vs points failed.")

    understat_season = str(int(config.SEASON[:2]) + 2000)
    matches = calculate_metrics.fetch_season_matches(understat_season)
    gw_matches = pd.DataFrame()
    if not matches.empty:
        start_naive = pd.Timestamp(start).tz_localize(None) if start.tzinfo else pd.Timestamp(start)
        end_naive = pd.Timestamp(end).tz_localize(None) if end.tzinfo else pd.Timestamp(end)
        gw_matches = matches[(matches["datetime"] >= start_naive) & (matches["datetime"] <= end_naive)].copy()

    if gw_matches.empty:
        logger.warning("generate_all_visuals: no Understat matches found for GW%d's date window; skipping shot maps/avg positions/PPDA.", gw)
        return written

    gw_matches["combined_xg"] = gw_matches["home_xg"] + gw_matches["away_xg"]
    top_matches = gw_matches.nlargest(3, "combined_xg")

    for _, match in top_matches.iterrows():
        try:
            match_data = fetch_data._understat_get(f"getMatchData/{match['match_id']}")
            shots = match_data.get("shots", {})
            rows = [dict(s, team=match["home_team"] if side == "h" else match["away_team"]) for side in ("h", "a") for s in shots.get(side, {}).values()]
            if rows:
                shots_df = pd.DataFrame(rows)
                path = out_dir / f"gw{gw}_shotmap_{match['match_id']}.png"
                plot_shot_map(shots_df, {"home_team": match["home_team"], "away_team": match["away_team"]}, path)
                written.append(path)
        except Exception:
            logger.exception("generate_all_visuals: shot map failed for match %s.", match["match_id"])

    try:
        ppda_raw = fetch_data.fetch_fbref_match_stats(start, end)
        ppda_df = calculate_metrics.compute_ppda(ppda_raw)
        if not ppda_df.empty:
            path = out_dir / f"gw{gw}_ppda.png"
            plot_ppda_bar(ppda_df, path)
            written.append(path)
    except Exception:
        logger.exception("generate_all_visuals: PPDA chart failed (FBref likely blocked).")
        ppda_df = pd.DataFrame()

    try:
        positions = calculate_metrics.compute_average_positions(start, end)
        if not positions.empty:
            for _, match in top_matches.iterrows():
                game_positions = positions[positions["team"].isin([match["home_team"], match["away_team"]])]
                if game_positions.empty:
                    continue
                match_ppda = ppda_df[ppda_df.get("team", pd.Series(dtype=str)).isin([match["home_team"], match["away_team"]])] if not ppda_df.empty else pd.DataFrame()
                home_ppda = match_ppda[match_ppda["team"] == match["home_team"]]["ppda"].mean() if not match_ppda.empty else float("nan")
                away_ppda = match_ppda[match_ppda["team"] == match["away_team"]]["ppda"].mean() if not match_ppda.empty else float("nan")
                meta = {
                    "home_team": match["home_team"], "away_team": match["away_team"],
                    "home_xg": match["home_xg"], "away_xg": match["away_xg"],
                    "home_ppda": home_ppda, "away_ppda": away_ppda,
                }
                path = out_dir / f"gw{gw}_avgpos_{match['match_id']}.png"
                plot_average_positions(game_positions, meta, path)
                written.append(path)
    except Exception:
        logger.exception("generate_all_visuals: average positions failed (FBref likely blocked).")

    return written
