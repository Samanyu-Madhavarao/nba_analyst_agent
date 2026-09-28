from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import time

def download_season_games(season: str):
    """season format: '2023-24'"""
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00"  # NBA
    )
    df = gamefinder.get_data_frames()[0]
    time.sleep(0.6)  # be polite to the API, avoid rate limiting
    return df

seasons = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
           "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
           "2025-26"]

all_games = []
for s in seasons:
    print(f"Downloading {s}...")
    all_games.append(download_season_games(s))

games_raw = pd.concat(all_games, ignore_index=True)
games_raw.to_parquet("data/raw/games.parquet")