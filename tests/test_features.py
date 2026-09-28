import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from src.features.schedule_features import add_games_in_window

GAMES_PATH = Path("data/processed/games_clean.parquet")
TEAM = "BOS"
SEASON = "2021-22"       # pick a season fully in the past, not in progress
N_ROWS_TO_SHOW = 20
WINDOW_DAYS = [7, 14]


df = pd.read_parquet(GAMES_PATH)
df = df[df["GAME_ID"].str.startswith("002")].copy()
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

result = add_games_in_window(df, date_col="GAME_DATE", windows_days=WINDOW_DAYS)

team_games = result[result["TEAM_ABBREVIATION"] == TEAM].sort_values("GAME_DATE")
team_games = team_games[team_games["season"] == SEASON]

cols = ["GAME_DATE"] + [f"games_last_{n}_days" for n in WINDOW_DAYS]
print(f"\n{TEAM} {SEASON} — first {N_ROWS_TO_SHOW} games:")
print(team_games[cols].head(N_ROWS_TO_SHOW).to_string(index=False))

# NaN check across the whole dataset
print("\nNaN counts (should be 0):")
for n in WINDOW_DAYS:
    print(f"  games_last_{n}_days: {result[f'games_last_{n}_days'].isna().sum()}")

# Manual recompute for a handful of specific rows, so you can compare
# the printed table above against real arithmetic instead of eyeballing dates.
print(f"\nManual recompute check for {TEAM} {SEASON}:")
dates = team_games["GAME_DATE"].tolist()
for i, d in enumerate(dates[:N_ROWS_TO_SHOW]):
    for n in WINDOW_DAYS:
        window_start = d - pd.Timedelta(days=n)
        prior_count = sum(1 for pd_date in dates[:i] if window_start <= pd_date < d)
        computed = team_games.iloc[i][f"games_last_{n}_days"]
        flag = "OK" if prior_count == computed else "MISMATCH"
        print(f"  game {i} ({d.date()}): last_{n}_days manual={prior_count} "
              f"computed={computed}  [{flag}]")