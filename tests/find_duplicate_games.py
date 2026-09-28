"""
find_duplicate_games.py

Diagnoses the row-count mismatch from join_opponent_stats: finds any
GAME_ID in games.parquet that doesn't have exactly 2 rows (one per team).

Run from project root:
    python tests/find_duplicate_games.py
"""

import pandas as pd

games = pd.read_parquet("data/raw/games.parquet")

counts = games.groupby("GAME_ID").size()
bad_games = counts[counts != 2]

print(f"Total games.parquet rows: {len(games):,}")
print(f"Unique GAME_IDs: {games['GAME_ID'].nunique():,}")
print(f"GAME_IDs with != 2 rows: {len(bad_games)}")
print()

if len(bad_games) > 0:
    print("Offending GAME_IDs and their row counts:")
    print(bad_games)
    print()

    print("Full rows for the first few offending games:")
    for gid in bad_games.index[:5]:
        subset = games[games["GAME_ID"] == gid]
        print(f"\n--- GAME_ID {gid} ({len(subset)} rows) ---")
        print(subset[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP", "PTS"]].to_string(index=False))

    # Also check for exact full-row duplicates specifically
    exact_dupes = games[games.duplicated(keep=False)]
    print(f"\nExact full-row duplicates (every column identical): {len(exact_dupes)}")
else:
    print("No mismatches found.")
