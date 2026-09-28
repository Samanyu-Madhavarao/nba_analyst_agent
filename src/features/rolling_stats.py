import pandas as pd

# Core step-9 stats to roll first (per earlier decision — expand later if needed)
ROLLING_STATS = ["net_rating", "offensive_rating", "defensive_rating", "pace"]
ROLLING_WINDOWS = [3, 5, 10]


def add_season_column(long_df: pd.DataFrame, date_col: str = "GAME_DATE") -> pd.DataFrame:
    long_df = long_df.copy()  # don't mutate the caller's dataframe
    if "SEASON_ID" in long_df.columns:
        start_year = long_df["SEASON_ID"].astype(str).str[1:].astype(int)
    else:
        date = pd.to_datetime(long_df[date_col])
        start_year = date.dt.year.where(date.dt.month >= 7, date.dt.year - 1)

    end_short = (start_year + 1) % 100
    long_df["season"] = start_year.astype(str) + "-" + end_short.astype(str).str.zfill(2)
    return long_df


def sort_for_rolling(long_df: pd.DataFrame, team_col: str = "TEAM_ABBREVIATION",
                      season_col: str = "season", date_col: str = "GAME_DATE") -> pd.DataFrame:
    long_df = long_df.sort_values([team_col, season_col, date_col]).reset_index(drop=True)
    return long_df


def add_rolling_stats(long_df: pd.DataFrame, team_col: str = "TEAM_ABBREVIATION",
                       season_col: str = "season", date_col: str = "GAME_DATE",
                       stats: list = ROLLING_STATS, windows: list = ROLLING_WINDOWS) -> pd.DataFrame:
    long_df = add_season_column(long_df, date_col=date_col)
    long_df = sort_for_rolling(long_df, team_col=team_col, season_col=season_col, date_col=date_col)

    for stat in stats:
        grouped = long_df.groupby([team_col, season_col])[stat]

        for n in windows:
            long_df[f"last_{n}_{stat}"] = grouped.transform(
                lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean()
            )

    return long_df


def add_games_available(long_df: pd.DataFrame, team_col: str = "TEAM_ABBREVIATION",
                         season_col: str = "season", date_col: str = "GAME_DATE",
                         stats: list = ROLLING_STATS, windows: list = ROLLING_WINDOWS) -> pd.DataFrame:
    long_df = add_season_column(long_df, date_col=date_col)
    long_df = sort_for_rolling(long_df, team_col=team_col, season_col=season_col, date_col=date_col)
 
    reference_stat = stats[0]
    grouped = long_df.groupby([team_col, season_col])[reference_stat]
 
    for n in windows:
        long_df[f"last_{n}_games_available"] = grouped.transform(
            lambda s, n=n: s.shift(1).rolling(n, min_periods=1).count()
        )
 
    return long_df


def handle_missing_rolling_stats(long_df: pd.DataFrame, stats: list = ROLLING_STATS,
                                  windows: list = ROLLING_WINDOWS, strategy: str = "drop") -> pd.DataFrame:
    rolling_cols = [f"last_{n}_{stat}" for stat in stats for n in windows]
    long_df = long_df.copy()

    if strategy == "drop":
        long_df = long_df.dropna(subset=rolling_cols)

    elif strategy == "league_average":
        if "season" not in long_df.columns:
            raise KeyError("league_average strategy requires a 'season' column — run add_season_column() first.")

        season_avg = long_df.groupby("season")[stats].mean()

        seasons_sorted = sorted(long_df["season"].unique())
        prev_season = {s: (seasons_sorted[i - 1] if i > 0 else None) for i, s in enumerate(seasons_sorted)}

        for stat in stats:
            for n in windows:
                col = f"last_{n}_{stat}"
                missing_mask = long_df[col].isna()
                if not missing_mask.any():
                    continue

                fill_values = long_df.loc[missing_mask, "season"].map(
                    lambda s: season_avg.loc[prev_season[s], stat] if prev_season[s] is not None else float("nan")
                )
                long_df.loc[missing_mask, col] = fill_values

        remaining_na = long_df[rolling_cols].isna().any(axis=1).sum()
        if remaining_na:
            print(f"Warning: {remaining_na} rows still NaN after league_average fill "
                  f"(likely the first season in the dataset, with no prior season to borrow from).")

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'drop' or 'league_average'.")

    return long_df


def test_no_leakage(long_df: pd.DataFrame, game_id_col: str = "GAME_ID",
                     team_col: str = "TEAM_ABBREVIATION", date_col: str = "GAME_DATE",
                     stat: str = "net_rating", window: int = 10) -> None:
    long_df = add_rolling_stats(long_df, team_col=team_col, date_col=date_col)
    rolling_col = f"last_{window}_{stat}"

    game_number = long_df.groupby([team_col, "season"]).cumcount() + 1
    candidates = long_df[(game_number > window) & long_df[rolling_col].notna()]

    if candidates.empty:
        raise ValueError(
            f"No row found with at least {window} prior games — check that "
            f"long_df actually has teams with {window}+ games in a season."
        )

    target_row = candidates.iloc[0]
    target_team = target_row[team_col]
    target_season = target_row["season"]
    target_date = target_row[date_col]

    prior_games = long_df[
        (long_df[team_col] == target_team)
        & (long_df["season"] == target_season)
        & (long_df[date_col] < target_date)
    ].sort_values(date_col)

    manual_value = prior_games[stat].tail(window).mean()
    computed_value = target_row[rolling_col]

    assert abs(manual_value - computed_value) < 1e-9, (
        f"Leakage check FAILED for {target_team} on {target_date}: "
        f"manual={manual_value}, computed={computed_value}"
    )
    print(f"Leakage check passed for {target_team} on {target_date.date()}: "
          f"manual={manual_value:.4f}, computed={computed_value:.4f}")