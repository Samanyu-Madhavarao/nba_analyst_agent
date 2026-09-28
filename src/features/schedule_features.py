import pandas as pd
from rolling_stats import add_season_column, sort_for_rolling

def add_rest_days(long_df: pd.DataFrame, team_col: str = "TEAM_ABBREVIATION",
                   season_col: str = "season", date_col: str = "GAME_DATE") -> pd.DataFrame:
    long_df = add_season_column(long_df, date_col)
    long_df = sort_for_rolling(long_df, team_col, season_col, date_col)
    grouped = long_df.groupby([team_col, season_col])[date_col]
    day_gap = grouped.diff().dt.days
    rest_days = day_gap - 1
    long_df['rest_days'] = rest_days
    return long_df


def add_back_to_back(long_df: pd.DataFrame, rest_col: str = "rest_days") -> pd.DataFrame:
    if rest_col not in long_df.columns:
        long_df = add_rest_days(long_df)
    long_df["is_back_to_back"] = (long_df[rest_col] == 0)
    return long_df


def handle_missing_rest_days(long_df: pd.DataFrame, rest_col: str = "rest_days",
                              strategy: str = "fill_default", default_rest: int = 3) -> pd.DataFrame:
    if strategy == 'fill_default':
        long_df[rest_col] = long_df[rest_col].fillna(default_rest)
    elif strategy == 'drop':
        long_df = long_df.dropna(subset=[rest_col])
    else:
        raise ValueError('unrecognized strategy')
    return long_df


def add_games_in_window(long_df: pd.DataFrame, team_col: str = "TEAM_ABBREVIATION",
                         season_col: str = "season", date_col: str = "GAME_DATE",
                         windows_days: list = [7, 14]) -> pd.DataFrame:
    long_df = add_season_column(long_df, date_col)
    long_df = long_df.sort_values([team_col, season_col, date_col])

    for n in windows_days:
        def _count_prior(group, n=n):
            s = pd.Series(1, index=group[date_col])
            return s.rolling(f"{n}D", closed="left", min_periods=0).sum()

        counts = long_df.groupby([team_col, season_col], group_keys=False).apply(_count_prior)
        long_df[f"games_last_{n}_days"] = counts.values

    return long_df


def add_rest_difference(wide_df: pd.DataFrame) -> pd.DataFrame:
    wide_df["rest_difference"] = wide_df["home_rest_days"] - wide_df["away_rest_days"]
    return wide_df
    