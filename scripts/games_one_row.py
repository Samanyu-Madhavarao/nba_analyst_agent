import pandas as pd

def pivot_games_to_one_row(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    games["is_home"] = games["MATCHUP"].str.contains("vs.")

    home = games[games["is_home"]].rename(columns={
        "TEAM_ABBREVIATION": "HOME_TEAM_ABBREVIATION",
        "PTS": "HOME_PTS",
        "WL": "HOME_WL",
    })
    away = games[~games["is_home"]].rename(columns={
        "TEAM_ABBREVIATION": "AWAY_TEAM_ABBREVIATION",
        "PTS": "AWAY_PTS",
        "WL": "AWAY_WL",
    })

    one_row_per_game = home[["GAME_ID", "GAME_DATE", "SEASON_ID",
                              "HOME_TEAM_ABBREVIATION", "HOME_PTS", "HOME_WL"]].merge(
        away[["GAME_ID", "AWAY_TEAM_ABBREVIATION", "AWAY_PTS", "AWAY_WL"]],
        on="GAME_ID",
        validate="1:1"
    )
    return one_row_per_game

games_raw = pd.read_parquet("data/raw/games.parquet")
games_raw = games_raw[games_raw["GAME_ID"].str.startswith("002")]  # regular season only

games_raw["is_home"] = games_raw["MATCHUP"].str.contains("vs.")

game_sizes = games_raw.groupby("GAME_ID").size()
home_counts = games_raw.groupby("GAME_ID")["is_home"].sum()

# A clean game has exactly 2 total rows AND exactly 1 home row
bad_game_ids = game_sizes[(game_sizes != 2) | (home_counts != 1)].index.tolist()

print(f"{len(bad_game_ids)} games with broken MATCHUP data")

before = len(games_raw)
games_raw = games_raw[~games_raw["GAME_ID"].isin(bad_game_ids)]
after = len(games_raw)
print(f"Dropped {before - after} rows across {len(bad_game_ids)} games with corrupted MATCHUP data")

games = pivot_games_to_one_row(games_raw)
print(games.shape)
print(games.head())

games.to_parquet("data/processed/games_one_row.parquet")