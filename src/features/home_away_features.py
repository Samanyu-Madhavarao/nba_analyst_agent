import pandas as pd

HOME_ROAD_STATS = ["net_rating"]

def add_win_column(long_df: pd.DataFrame, pts_col: str="PTS", opp_pts_col: str="opp_PTS") -> pd.DataFrame:
    team_win = (long_df[pts_col] > long_df[opp_pts_col]).astype(int)
    long_df["team_win"] = team_win
    return long_df

def add_home_road_split_stats(long_df: pd.DataFrame, team_col: str="TEAM_ABBREVIATION",
                               season_col: str="season", date_col: str="GAME_DATE", 
                               is_home_col: str="is_home", stats: list=HOME_ROAD_STATS,
                               window: int=10) -> pd.DataFrame:
    long_df = long_df.copy()
    home_games = long_df[long_df[is_home_col]].sort_values([team_col, season_col, date_col])
    road_games = long_df[~long_df[is_home_col]].sort_values([team_col, season_col, date_col])

    for stat in stats:
        home_grouped = home_games.groupby([team_col, season_col])[stat]
        road_grouped = road_games.groupby([team_col, season_col])[stat]

        home_rolled = home_grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        road_rolled = road_grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

        home_col = f"home_last_{window}_{stat}"
        road_col = f"road_last_{window}_{stat}"

        long_df[home_col] = float("nan")
        long_df[road_col] = float("nan")

        long_df.loc[home_games.index, home_col] = home_rolled
        long_df.loc[road_games.index, road_col] = road_rolled

    return long_df

def add_win_percentages(long_df: pd.DataFrame, team_col: str = "TEAM_ABBREVIATION",
                         season_col: str = "season", date_col: str = "GAME_DATE",
                         is_home_col: str = "is_home", window: int = 10) -> pd.DataFrame:

    if "team_win" not in long_df.columns:
        long_df = add_win_column(long_df)

    long_df = add_home_road_split_stats(
        long_df, team_col=team_col, season_col=season_col, date_col=date_col,
        is_home_col=is_home_col, stats=["team_win"], window=window
    )

    long_df = long_df.rename(columns={
        f"home_last_{window}_team_win": f"home_last_{window}_win_pct",
        f"road_last_{window}_team_win": f"road_last_{window}_win_pct"
    })

    return long_df