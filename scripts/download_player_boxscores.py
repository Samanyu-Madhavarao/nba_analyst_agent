import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog

RAW_DIR = Path("data/raw")
PER_SEASON_DIR = RAW_DIR / "player_boxscores"
COMBINED_PATH = RAW_DIR / "player_boxscores.parquet"

FIRST_SEASON_START_YEAR = 2015
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 5
BETWEEN_SEASON_SLEEP_SECONDS = 0.6


def get_current_season_start_year(today: datetime = None) -> int:
    """
    NBA seasons start in October. If today is Oct or later, the current
    season started this calendar year; otherwise it started last year.
    """
    today = today or datetime.now()
    if today.month >= 10:
        return today.year
    return today.year - 1


def build_season_list(first_start_year: int = FIRST_SEASON_START_YEAR) -> list:
    """
    Returns season strings like '2015-16', '2016-17', ..., up through
    whatever season is currently in progress (or most recently completed).
    """
    last_start_year = get_current_season_start_year()
    seasons = []
    for start_year in range(first_start_year, last_start_year + 1):
        end_year_short = str(start_year + 1)[2:]
        seasons.append(f"{start_year}-{end_year_short}")
    return seasons


def download_season(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Fetches one season of player-level game log rows. Retries a few times
    on failure since this endpoint is prone to intermittent read timeouts.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="P",
            )
            df = result.get_data_frames()[0]
            return df
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed for {season}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"Failed to download season {season} after {MAX_RETRIES} attempts") from last_error


def download_all_seasons(seasons: list) -> None:
    """
    Downloads each season, saving it immediately to its own parquet file
    so a failure partway through doesn't lose seasons already fetched.
    """
    PER_SEASON_DIR.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        season_path = PER_SEASON_DIR / f"{season}.parquet"

        if season_path.exists():
            print(f"Skipping {season} (already downloaded at {season_path})")
            continue

        print(f"Downloading {season}...")
        df = download_season(season)
        print(f"  {len(df)} player-game rows")

        df.to_parquet(season_path, index=False)
        time.sleep(BETWEEN_SEASON_SLEEP_SECONDS)


def combine_seasons() -> pd.DataFrame:
    """
    Concatenates every per-season parquet file into one combined dataframe
    and writes it to data/raw/player_boxscores.parquet.
    """
    season_files = sorted(PER_SEASON_DIR.glob("*.parquet"))
    if not season_files:
        raise FileNotFoundError(f"No per-season files found in {PER_SEASON_DIR}")

    frames = [pd.read_parquet(f) for f in season_files]
    combined = pd.concat(frames, ignore_index=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(COMBINED_PATH, index=False)
    print(f"Combined {len(season_files)} seasons -> {len(combined)} rows -> {COMBINED_PATH}")

    return combined


def run_sanity_checks(df: pd.DataFrame) -> None:
    """
    Quick checks worth eyeballing before trusting this data:
    - Row count is in a plausible ballpark per season.
    - Per-team-per-game total minutes is close to 240 (or more with OT).
    """
    print("\n--- Sanity checks ---")

    rows_per_season = df.groupby("SEASON_ID").size()
    print("Rows per season:")
    print(rows_per_season)

    minutes_check = (
        df.groupby(["GAME_ID", "TEAM_ABBREVIATION"])["MIN"]
        .sum()
        .describe()
    )
    print("\nTeam-game total MIN distribution (expect mean/median near 240, "
          "higher with overtime games):")
    print(minutes_check)


if __name__ == "__main__":
    seasons = build_season_list()
    print(f"Seasons to download: {seasons}\n")

    download_all_seasons(seasons)
    combined = combine_seasons()
    run_sanity_checks(combined)