# src/data/download_boxscores.py
import pandas as pd
import time
import os
from nba_api.stats.endpoints import boxscoretraditionalv3

RAW_DIR = "data/raw/boxscores"
os.makedirs(RAW_DIR, exist_ok=True)

def already_downloaded(game_id: str) -> bool:
    return os.path.exists(f"{RAW_DIR}/{game_id}.parquet")

def download_one_boxscore(game_id: str, max_retries=3):
    for attempt in range(max_retries):
        try:
            box = boxscoretraditionalv3.BoxScoreTraditionalV3(
                game_id=game_id, timeout=30
            )
            player_df = box.get_data_frames()[0]
            team_df = box.get_data_frames()[1]

            player_df["game_id"] = game_id
            team_df["game_id"] = game_id

            player_df.to_parquet(f"{RAW_DIR}/{game_id}_players.parquet")
            team_df.to_parquet(f"{RAW_DIR}/{game_id}_teams.parquet")
            return True

        except Exception as e:
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            print(f"  {game_id} failed (attempt {attempt+1}): {e} — retrying in {wait}s")
            time.sleep(wait)

    print(f"  {game_id} FAILED after {max_retries} attempts, skipping")
    return False

def download_all_boxscores(games_df: pd.DataFrame):
    game_ids = games_df["GAME_ID"].astype(str).str.zfill(10).unique()
    total = len(game_ids)

    for i, game_id in enumerate(game_ids):
        if already_downloaded(game_id):
            continue

        print(f"[{i+1}/{total}] Downloading {game_id}")
        download_one_boxscore(game_id)
        time.sleep(0.6)  # rate limit pacing

if __name__ == "__main__":
    games = pd.read_parquet("data/raw/games.parquet")
    download_all_boxscores(games)