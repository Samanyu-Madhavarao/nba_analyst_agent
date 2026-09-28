import pandas as pd

STAT_COLS = ["PTS", "FGA", "FGM", "FG3M", "FG3A", "OREB", "DREB", "TOV", "FTA", "MIN"]


def get_pos(df, prefix=""):
    return df[prefix + 'FGA'] - df[prefix + 'OREB'] + df[prefix + 'TOV'] + (0.4 * df[prefix + 'FTA'])


def get_ortg(pts, pos):
    return 100 * pts / pos


def get_drtg(opp_pts, opp_pos):
    return 100 * opp_pts / opp_pos


def get_net_rating(ortg, drtg):
    return ortg - drtg


def get_efg_pct(fgm, fg3m, fga):
    return (fgm + (0.5 * fg3m)) / fga


def get_ts_pct(pts, fga, fta):
    return pts / (2 * (fga + (0.44 * fta)))


def get_oreb_pct(oreb, opp_dreb):
    return oreb / (oreb + opp_dreb)


def get_dreb_pct(dreb, opp_oreb):
    return dreb / (dreb + opp_oreb)


def get_tov_rate(tov, pos):
    return tov / pos


def get_three_point_attempt_rate(fg3a, fga):
    return fg3a / fga


def get_free_throw_rate(fta, fga):
    return fta / fga


def get_three_point_pct(fg3m, fg3a):
    return fg3m / fg3a


def get_pace(pos, opp_pos, team_minutes):
    """
    Pace = possessions per 48 minutes, averaging the team's and
    opponent's possession estimates (standard Basketball-Reference approach).
    team_minutes is the team's total minutes played in the game
    (240 for a regulation game with 5 players on court, more with OT).
    """
    return 48 * ((pos + opp_pos) / 2) / (team_minutes / 5)


def join_opponent_stats(games_df: pd.DataFrame, game_id_col: str = "GAME_ID", team_col: str = "TEAM_ABBREVIATION") -> pd.DataFrame:
    stat_cols = [c for c in games_df.columns if c in STAT_COLS]

    opponent_side = games_df[[game_id_col, team_col] + stat_cols].copy()
    opponent_side = opponent_side.rename(
        columns={team_col: f"opp_{team_col}", **{c: f"opp_{c}" for c in stat_cols}}
    )

    merged = games_df.merge(opponent_side, on=game_id_col, how="left")
    merged = merged.loc[merged[team_col] != merged[f"opp_{team_col}"]].reset_index(drop=True)

    # Sanity check: should not have grown beyond 2x the input (one opponent per team-game)
    if len(merged) > 2 * len(games_df):
        raise ValueError(
            f"join_opponent_stats produced {len(merged)} rows from {len(games_df)} "
            f"input rows. Expected 2x (one row per team per game). Check for "
            f"duplicate game_id/team_col pairs or games with more than 2 teams."
        )

    return merged


def add_home_away(df: pd.DataFrame, matchup_col: str = "MATCHUP", team_col="TEAM_ABBREVIATION") -> pd.DataFrame:
    """
    Flags each team-game row as home or away using the standard nba_api
    MATCHUP convention: 'vs.' means home, '@' means away.
    """
    df = df.copy()
    if matchup_col not in df.columns:
        raise KeyError(
            f"'{matchup_col}' column not found; cannot determine home/away. "
            f"Pass a games_df that includes MATCHUP, or supply your own "
            f"is_home flag before calling to_wide()."
        )
    is_vs = df[matchup_col].str.contains("vs.", regex=False)
    home_team = df[matchup_col].str.split(" @ ").str[1].where(
        ~is_vs, df[matchup_col].str.split(" vs. ").str[0]
    )
    df["is_home"] = df[team_col] == home_team
    return df


def to_wide(df: pd.DataFrame, game_id_col: str = "GAME_ID") -> pd.DataFrame:
    """
    Collapses the long team-game table (one row per team per game) into
    one row per game, with every stat column prefixed home_/away_.
    Requires add_home_away() to have been run first.
    """
    if "is_home" not in df.columns:
        raise KeyError("Call add_home_away(df) before to_wide(df).")

    home = df[df["is_home"]].drop(columns=["is_home"]).copy()
    away = df[~df["is_home"]].drop(columns=["is_home"]).copy()

    home = home.add_prefix("home_").rename(columns={f"home_{game_id_col}": game_id_col})
    away = away.add_prefix("away_").rename(columns={f"away_{game_id_col}": game_id_col})

    wide = home.merge(away, on=game_id_col, how="inner")

    if len(wide) > df[game_id_col].nunique():
        raise ValueError(
            f"to_wide produced {len(wide)} rows for {df[game_id_col].nunique()} "
            f"unique games. Expected exactly one row per game — check for "
            f"duplicate home or away rows per game_id."
        )

    return wide


def difference(df: pd.DataFrame, stat: str) -> pd.Series:
    return df["home_" + stat] - df["away_" + stat]


def build_team_stats_long(games_df: pd.DataFrame, game_id_col: str = "GAME_ID", team_col: str = "TEAM_ABBREVIATION") -> pd.DataFrame:
    """
    Step 9 only: one row per team per game, with per-game rate/rating stats
    computed (net_rating, pace, etc.) plus opponent ('opp_') columns attached.
    Stays in LONG form on purpose — this is the shape rolling_stats.py needs
    as input, since rolling averages are a per-team time series and don't
    make sense computed on a pivoted wide table.
    """
    df = join_opponent_stats(games_df, game_id_col=game_id_col, team_col=team_col)

    df['pos'] = get_pos(df)
    df['opp_pos'] = get_pos(df, "opp_")

    df['offensive_rating'] = get_ortg(df['PTS'], df['pos'])
    df['defensive_rating'] = get_drtg(df['opp_PTS'], df['opp_pos'])
    df['net_rating'] = get_net_rating(df['offensive_rating'], df['defensive_rating'])
    df['effective_fg_percentage'] = get_efg_pct(df['FGM'], df['FG3M'], df['FGA'])
    df['true_shooting_percentage'] = get_ts_pct(df['PTS'], df['FGA'], df['FTA'])
    df['offensive_rebound_pct'] = get_oreb_pct(df['OREB'], df['opp_DREB'])
    df['defensive_rebound_pct'] = get_dreb_pct(df['DREB'], df['opp_OREB'])
    df['turnover_rate'] = get_tov_rate(df['TOV'], df['pos'])
    df['pace'] = get_pace(df['pos'], df['opp_pos'], df['MIN'])
    df['three_point_attempt_rate'] = get_three_point_attempt_rate(df['FG3A'], df['FGA'])
    df['free_throw_rate'] = get_free_throw_rate(df['FTA'], df['FGA'])

    # Defensive/"opponent" stats: what this team allows its opponent to do.
    df['opponent_effective_fg_percentage'] = get_efg_pct(df['opp_FGM'], df['opp_FG3M'], df['opp_FGA'])
    df['opponent_three_point_percentage'] = get_three_point_pct(df['opp_FG3M'], df['opp_FG3A'])
    df['opponent_turnover_rate'] = get_tov_rate(df['opp_TOV'], df['opp_pos'])

    return df


def add_stat_differences(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Step 10: home_ vs away_ differences for every step-9 stat.
    Expects a WIDE table (i.e. already run through add_home_away() + to_wide()).
    """
    wide = wide.copy()
    wide["net_rating_difference"] = difference(wide, "net_rating")
    wide["offensive_rating_difference"] = difference(wide, "offensive_rating")
    wide["defensive_rating_difference"] = difference(wide, "defensive_rating")
    wide["pace_difference"] = difference(wide, "pace")
    wide["offensive_rebound_difference"] = difference(wide, "offensive_rebound_pct")
    wide["defensive_rebound_difference"] = difference(wide, "defensive_rebound_pct")
    wide["turnover_difference"] = difference(wide, "turnover_rate")
    wide["shooting_difference"] = difference(wide, "effective_fg_percentage")
    wide["three_point_attempt_rate_difference"] = difference(wide, "three_point_attempt_rate")
    wide["free_throw_rate_difference"] = difference(wide, "free_throw_rate")
    wide["opponent_efg_difference"] = difference(wide, "opponent_effective_fg_percentage")
    wide["opponent_three_point_pct_difference"] = difference(wide, "opponent_three_point_percentage")
    wide["opponent_turnover_rate_difference"] = difference(wide, "opponent_turnover_rate")

    return wide


def build_team_game_stats(games_df: pd.DataFrame, game_id_col: str = "GAME_ID", team_col: str = "TEAM_ABBREVIATION") -> pd.DataFrame:
    """
    Convenience wrapper: steps 9 + 10 in one call, long -> wide, with
    differences attached. Use this if you're NOT adding rolling stats
    (step 11) in between. If you ARE adding rolling stats, call
    build_team_stats_long() -> add_rolling_stats() -> add_home_away() ->
    to_wide() -> add_stat_differences() yourself instead, so rolling_stats.py
    can operate on the long form before the pivot happens.
    """
    df = build_team_stats_long(games_df, game_id_col=game_id_col, team_col=team_col)
    df = add_home_away(df, matchup_col="MATCHUP")
    wide = to_wide(df, game_id_col=game_id_col)
    wide = add_stat_differences(wide)
    return wide