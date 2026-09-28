import pandas as pd

def filter_to_competitive_games(games_df):
    """Keep only regular season and playoff games (excludes preseason, All-Star)."""
    return games_df[games_df["GAME_ID"].str.startswith(("002", "004", "005"))]

def main():
    games = pd.read_parquet("data/raw/games.parquet")
    games_clean = filter_to_competitive_games(games)
    games_clean.to_parquet("data/processed/games_clean.parquet")
    print(f"Wrote {len(games_clean):,} rows (dropped {len(games) - len(games_clean)})")

if __name__ == "__main__":
    main()