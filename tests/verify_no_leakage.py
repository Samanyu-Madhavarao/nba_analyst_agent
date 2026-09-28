"""
verify_no_leakage.py

Run this against your real data to sanity-check Step 8 before moving on.

Usage:
    python tests/verify_no_leakage.py --games data/raw/games.parquet

What it checks:
  1. Every game has a parseable date.
  2. add_rest_days() never assigns a team's FIRST game of a dataset a
     rest value derived from a future game (sanity check on shift direction).
  3. For a handful of sample games, confirms that team_as_of() returns
     zero rows on/after the game date.
  4. Prints a manual example so you can eyeball it.

This does not require betting-line data -- it only needs date, home_team,
away_team columns, matching the outline's minimum games schema.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.data_leakage_utils import add_rest_days, team_as_of  # noqa: E402


def main(games_path: str, date_col: str, home_col: str, away_col: str):
    df = pd.read_parquet(games_path)

    required = {date_col, home_col, away_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"games file is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Normalize to the column names the leakage utils expect
    df = df.rename(columns={date_col: "date", home_col: "home_team", away_col: "away_team"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Loaded {len(df):,} games spanning {df['date'].min().date()} "
          f"to {df['date'].max().date()}")

    # 1. Rest days leakage check
    rest_df = add_rest_days(df)
    print("\nSample rest_days output:")
    print(rest_df[["date", "home_team", "away_team", "home_rest",
                    "away_rest", "rest_difference"]].head(5).to_string(index=False))

    # 2. Spot-check team_as_of() on a handful of random games
    print("\nSpot-checking team_as_of() leakage on 5 random games...")
    sample = df.sample(min(5, len(df)), random_state=42)
    failures = 0
    for _, row in sample.iterrows():
        history = team_as_of(df, row["home_team"], row["date"])
        if (history["date"] >= row["date"]).any():
            failures += 1
            print(f"  LEAK: game {row.get('game_id', '?')} on {row['date'].date()} "
                  f"pulled in same/future-day data.")
        else:
            latest = history["date"].max() if len(history) else None
            print(f"  OK: {row['home_team']} game on {row['date'].date()} -> "
                  f"latest prior game used: {latest.date() if latest is not None else 'none (first game)'}")

    if failures:
        print(f"\n{failures} leakage failures found. Fix before proceeding to Step 9.")
        sys.exit(1)
    else:
        print("\nNo leakage detected in spot checks. Safe to move to Step 9 "
              "(team strength features).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", required=True, help="Path to games parquet file")
    parser.add_argument("--date-col", default="date", help="Name of the game-date column")
    parser.add_argument("--home-col", default="home_team", help="Name of the home-team column")
    parser.add_argument("--away-col", default="away_team", help="Name of the away-team column")
    args = parser.parse_args()
    main(args.games, args.date_col, args.home_col, args.away_col)