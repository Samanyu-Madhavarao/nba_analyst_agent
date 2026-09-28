"""
run_full_injury_and_production_features.py

Full-dataset production run of Step 14 (injury_features.py) and Step 15
(player_production_features.py) — every regular-season game, every team,
every season in games.parquet / player_boxscores.parquet.

Only run this after test_injury_and_production_features.py has passed
cleanly at single-team and full-single-season scope — this script
assumes correctness is already established and does NOT repeat the
expensive per-team diagnostics (leakage check, roster consistency,
spot checks). It focuses on:
  1. Running both pipelines across the whole dataset, timed.
  2. Lightweight structural sanity checks — cheap enough to run at full
     scale as a final gut-check, NOT a substitute for the scoped
     verification already done.
  3. Saving output to parquet so this doesn't need to be re-run.

If anything here looks off (PROBLEM/MISMATCH, unexpected NaNs, values
out of range), stop and go back to
test_injury_and_production_features.py scoped to whichever
team/season looks responsible — don't debug a full-scale run directly.

Usage:
    python run_full_injury_and_production_features.py
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "features"))

import pandas as pd

from team_strength_features import build_team_stats_long
from rolling_stats import add_season_column
from injury_features import build_injury_features
from player_production_features import build_player_production_features, PRODUCTION_STATS

GAMES_PATH = Path("data/raw/games.parquet")
PLAYER_BOXSCORES_PATH = Path("data/raw/player_boxscores.parquet")

INJURY_OUTPUT_PATH = Path("data/processed/injury_features.parquet")
PRODUCTION_OUTPUT_PATH = Path("data/processed/player_production_features.parquet")


# ----------------------------------------------------------------------
# 1. Load everything — no team/season scoping, regular season only
# ----------------------------------------------------------------------

def load_full_data():
    games_df = pd.read_parquet(GAMES_PATH)
    games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"])

    # Regular season only — see the playoff-game investigation in
    # test_injury_and_production_features.py's history for why this
    # matters: playoff/preseason game_ids have no matching player rows
    # in player_boxscores.parquet (download_player_boxscores.py only
    # pulls "Regular Season"), which silently produces false all-zero
    # rows for any game_id that slips through unfiltered.
    games_df = games_df[games_df["GAME_ID"].str[:3] == "002"].copy()

    team_long = build_team_stats_long(games_df)
    team_long = add_season_column(team_long, date_col="GAME_DATE")
    print(team_long.groupby("season")["GAME_ID"].nunique().loc[["2019-20", "2020-21"]])

    player_long = pd.read_parquet(PLAYER_BOXSCORES_PATH)
    player_long["GAME_DATE"] = pd.to_datetime(player_long["GAME_DATE"])
    # Match player_long to the same (now regular-season-only) game set.
    player_long = player_long[player_long["GAME_ID"].isin(team_long["GAME_ID"])].copy()
    player_long = add_season_column(player_long, date_col="GAME_DATE")
    print(player_long.groupby("season")["GAME_ID"].nunique().loc[["2019-20", "2020-21"]])

    seasons = sorted(team_long["season"].unique())
    print(f"Loaded {team_long['GAME_ID'].nunique()} games, {len(team_long)} team-game rows, "
          f"{len(player_long)} player-game rows, across {len(seasons)} seasons: {seasons}")

    return team_long, player_long


# ----------------------------------------------------------------------
# 2. Run the pipelines, timed
# ----------------------------------------------------------------------

def run_pipelines(team_long: pd.DataFrame, player_long: pd.DataFrame):
    t0 = time.time()
    injury_out = build_injury_features(player_long, team_long)
    t1 = time.time()
    print(f"build_injury_features: {t1 - t0:.1f}s for {len(injury_out)} team-game rows "
          f"({(t1 - t0) / max(len(injury_out), 1):.4f}s/row)")

    t2 = time.time()
    production_out = build_player_production_features(player_long, team_long)
    t3 = time.time()
    print(f"build_player_production_features: {t3 - t2:.1f}s for {len(production_out)} team-game rows "
          f"({(t3 - t2) / max(len(production_out), 1):.4f}s/row)")

    print(f"Total: {t3 - t0:.1f}s")

    return injury_out, production_out


# ----------------------------------------------------------------------
# 3. Lightweight sanity checks — same shape as the test script's, kept
# minimal since correctness is already established
# ----------------------------------------------------------------------

def sanity_check(team_long: pd.DataFrame, injury_out: pd.DataFrame, production_out: pd.DataFrame):
    print("\n--- Sanity check ---")
    expected_rows = team_long[["GAME_ID", "TEAM_ABBREVIATION"]].drop_duplicates().shape[0]
    print(f"Expected rows: {expected_rows}  injury_out: {len(injury_out)}  production_out: {len(production_out)}")
    if len(injury_out) != expected_rows or len(production_out) != expected_rows:
        print("MISMATCH — stop here, do not save output, go back to the scoped test script.")
        return False

    for name, df in [("injury_out", injury_out), ("production_out", production_out)]:
        dup = df.duplicated(subset=["GAME_ID", "TEAM_ABBREVIATION"]).sum()
        nan_total = df.isna().sum().sum()
        print(f"{name}: {dup} duplicate rows, {nan_total} total NaNs "
              f"{'OK' if dup == 0 and nan_total == 0 else 'PROBLEM — investigate before trusting output'}")

    for stat in PRODUCTION_STATS:
        for prefix in ["expected_team", "missing_team"]:
            col = f"{prefix}_{stat}"
            infs = production_out[col].isin([float("inf"), float("-inf")]).sum()
            negs = (production_out[col] < 0).sum()
            if infs or negs:
                print(f"PROBLEM: {col} has {infs} inf and {negs} negative values")

    for stat in PRODUCTION_STATS:
        col = f"pct_{stat}_missing"
        out_of_range = ((production_out[col] < 0) | (production_out[col] > 1)).sum()
        if out_of_range:
            print(f"PROBLEM: {col} has {out_of_range} rows outside [0,1]")

    print("Sanity check complete — see PROBLEM/MISMATCH lines above, if any.")
    return True


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    team_long, player_long = load_full_data()
    player_long["SEASON_ID"].head()

    injury_out, production_out = run_pipelines(team_long, player_long)

    ok = sanity_check(team_long, injury_out, production_out)

    if ok:
        INJURY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        injury_out.to_parquet(INJURY_OUTPUT_PATH, index=False)
        production_out.to_parquet(PRODUCTION_OUTPUT_PATH, index=False)
        print(f"\nSaved:\n  {INJURY_OUTPUT_PATH}\n  {PRODUCTION_OUTPUT_PATH}")
    else:
        print("\nNOT saved — sanity check found a problem. Fix before re-running.")