import pandas as pd

# Trailing window (in team games, not calendar days) used to decide who's
# "expected" to play and who counts as a "starter". Matches the last_10
# window already used in rolling_stats.py.
EXPECTED_ROSTER_WINDOW = 10
EXPECTED_ROSTER_THRESHOLD = 0.5  # appeared in >= 50% of trailing window games
STARTER_COUNT = 5


def add_player_appearance_flag(player_long_df: pd.DataFrame) -> pd.DataFrame:
    df = player_long_df.copy()
 
    dup_mask = df.duplicated(subset=["GAME_ID", "PLAYER_ID"], keep=False)
    if dup_mask.any():
        raise ValueError(
            f"Found {dup_mask.sum()} duplicate (GAME_ID, PLAYER_ID) rows in "
            f"player_long_df — resolve before adding appearance flags, or "
            f"downstream appearance/minutes counts will be inflated."
        )
 
    df["appeared"] = 1
    return df



def build_team_game_calendar(team_long_df: pd.DataFrame, team_col: str = "TEAM_ABBREVIATION",
                              season_col: str = "season", date_col: str = "GAME_DATE",
                              game_id_col: str = "GAME_ID") -> pd.DataFrame:
    required_cols = [team_col, season_col, date_col, game_id_col]
    missing_cols = [c for c in required_cols if c not in team_long_df.columns]
    if missing_cols:
        raise KeyError(
            f"team_long_df is missing required column(s) {missing_cols}. "
            f"Run add_season_column() first if '{season_col}' is missing."
        )
 
    calendar = team_long_df[required_cols].drop_duplicates().copy()
    calendar = calendar.sort_values([team_col, season_col, date_col]).reset_index(drop=True)
 
    # One row per team per game_id, not two (home + away) — team_long_df
    # is already the per-team-per-game long form, so this should hold
    # naturally, but worth asserting rather than assuming.
    dup_check = calendar.duplicated(subset=[team_col, game_id_col])
    if dup_check.any():
        raise ValueError(
            f"Found {dup_check.sum()} duplicate (team, game_id) rows after "
            f"dedup — team_long_df may not be in the expected one-row-per-"
            f"team-per-game shape."
        )
 
    return calendar
    


def add_player_rolling_appearance_and_minutes(player_long_df: pd.DataFrame, team_game_calendar: pd.DataFrame,
                                               player_col: str = "PLAYER_ID",
                                               team_col: str = "TEAM_ABBREVIATION",
                                               season_col: str = "season",
                                               date_col: str = "GAME_DATE",
                                               game_id_col: str = "GAME_ID",
                                               window: int = EXPECTED_ROSTER_WINDOW) -> pd.DataFrame:
    player_team_season = player_long_df[[player_col, team_col, season_col]].drop_duplicates()
    tenure = player_long_df.groupby([player_col, team_col, season_col])[date_col].agg(
        tenure_start="min", tenure_end="max"
    ).reset_index()
    skeleton = player_team_season.merge(team_game_calendar, on=[team_col, season_col], how="left")
    skeleton = skeleton.merge(tenure, on=[player_col, team_col, season_col], how="left")
    skeleton = skeleton[
        (skeleton[date_col] >= skeleton["tenure_start"]) & (skeleton[date_col] <= skeleton["tenure_end"])
    ].drop(columns=["tenure_start", "tenure_end"])
    
    reindexed = skeleton.merge(
        player_long_df[[player_col, team_col, season_col, game_id_col, "MIN", "appeared"]],
        on=[player_col, team_col, season_col, game_id_col],
        how="left"
    )
    reindexed["appeared"] = reindexed["appeared"].fillna(0)
    reindexed["MIN"] = reindexed["MIN"].where(reindexed["appeared"] == 1)
    reindexed = reindexed.sort_values([player_col, team_col, season_col, date_col]).reset_index(drop=True)
    grouped_appeared = reindexed.groupby([player_col, team_col, season_col])["appeared"]
    grouped_minutes = reindexed.groupby([player_col, team_col, season_col])["MIN"]

    reindexed[f"player_games_played_last_{window}"] = grouped_appeared.transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )
    reindexed[f"player_avg_minutes_last_{window}"] = grouped_minutes.transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    reindexed[f"player_games_available_last_{window}"] = grouped_appeared.transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).count()
    )
 
    return reindexed



def get_expected_roster(player_rolling_df: pd.DataFrame, team: str, as_of_date,
                         player_col: str = "PLAYER_ID",
                         team_col: str = "TEAM_ABBREVIATION",
                         date_col: str = "GAME_DATE",
                         window: int = EXPECTED_ROSTER_WINDOW,
                         threshold: float = EXPECTED_ROSTER_THRESHOLD) -> pd.DataFrame:
    played_col = f"player_games_played_last_{window}"
    available_col = f"player_games_available_last_{window}"
    minutes_col = f"player_avg_minutes_last_{window}"
 
    for col in (played_col, available_col, minutes_col):
        if col not in player_rolling_df.columns:
            raise KeyError(
                f"'{col}' not found — run add_player_rolling_appearance_and_minutes() "
                f"with window={window} first."
            )
 
    subset = player_rolling_df[
        (player_rolling_df[team_col] == team) & (player_rolling_df[date_col] == as_of_date)
    ].copy()
 
    if subset.empty:
        raise ValueError(
            f"No rows found for team='{team}' as_of_date='{as_of_date}' — check "
            f"that this team/date combination exists in player_rolling_df's "
            f"team_game_calendar."
        )
 
    appearance_rate = subset[played_col] / subset[available_col].replace(0, float("nan"))
 
    subset["appearance_rate"] = appearance_rate
    mask = (subset["appearance_rate"] >= threshold).fillna(False)
    expected = subset[mask]
 
    return expected[[player_col, team_col, minutes_col, "appearance_rate"]].reset_index(drop=True)


def get_starters(expected_roster_df: pd.DataFrame, n: int = STARTER_COUNT,
                  minutes_col: str = f"player_avg_minutes_last_{EXPECTED_ROSTER_WINDOW}") -> pd.DataFrame:
    if expected_roster_df.empty:
        return expected_roster_df
    return expected_roster_df.nlargest(n, minutes_col)


def get_actual_game_roster(player_long_df: pd.DataFrame, game_id: str, team: str,
                           game_id_col: str = "GAME_ID", team_col: str = "TEAM_ABBREVIATION",
                           player_col: str = "PLAYER_ID") -> pd.DataFrame:
    mask = (player_long_df[game_id_col] == game_id) & (player_long_df[team_col] == team)
    actual = player_long_df.loc[mask, [player_col]].drop_duplicates().reset_index(drop=True)
    return actual


def get_players_out(expected_roster_df: pd.DataFrame, actual_roster_df: pd.DataFrame,
                     starters_df: pd.DataFrame, player_col: str='PLAYER_ID') -> dict:
    expected_ids = set(expected_roster_df[player_col])
    actual_ids = set(actual_roster_df[player_col])
    starter_ids = set(starters_df[player_col])
 
    missing_player_ids = expected_ids - actual_ids
    starters_out = missing_player_ids & starter_ids
 
    return {
        "missing_player_ids": missing_player_ids,
        "number_of_players_out": len(missing_player_ids),
        "starters_out": starters_out,
        "number_of_starters_out": len(starters_out),
    }


def get_team_missing_impact(missing_player_ids: set, expected_roster_df: pd.DataFrame,
                             minutes_col: str = f"player_avg_minutes_last_{EXPECTED_ROSTER_WINDOW}",
                             player_col: str="PLAYER_ID") -> float:
    if not missing_player_ids:
        return 0.0
    mask = expected_roster_df[player_col].isin(missing_player_ids)
    impact = expected_roster_df.loc[mask, minutes_col].sum()
    return float(impact)


def build_injury_features(player_long_df: pd.DataFrame, team_long_df: pd.DataFrame,
                           player_col: str = "PLAYER_ID", team_col: str = "TEAM_ABBREVIATION",
                           season_col: str = "season", date_col: str = "GAME_DATE",
                           game_id_col: str = "GAME_ID",
                           window: int = EXPECTED_ROSTER_WINDOW,
                           threshold: float = EXPECTED_ROSTER_THRESHOLD,
                           starter_count: int = STARTER_COUNT) -> pd.DataFrame:
    minutes_col = f"player_avg_minutes_last_{window}"
 
    player_long_df = add_player_appearance_flag(player_long_df)
 
    calendar = build_team_game_calendar(
        team_long_df, team_col=team_col, season_col=season_col,
        date_col=date_col, game_id_col=game_id_col
    )
 
    player_rolling = add_player_rolling_appearance_and_minutes(
        player_long_df, calendar, player_col=player_col, team_col=team_col,
        season_col=season_col, date_col=date_col, game_id_col=game_id_col,
        window=window
    )

    rows = []
    for _, cal_row in calendar.iterrows():
        team = cal_row[team_col]
        game_id = cal_row[game_id_col]
        as_of_date = cal_row[date_col]
 
        expected = get_expected_roster(
            player_rolling, team, as_of_date, player_col=player_col,
            team_col=team_col, date_col=date_col, window=window, threshold=threshold
        )
        starters = get_starters(expected, n=starter_count, minutes_col=minutes_col)
        actual = get_actual_game_roster(
            player_long_df, game_id, team, game_id_col=game_id_col,
            team_col=team_col, player_col=player_col
        )
        out_info = get_players_out(expected, actual, starters, player_col=player_col)
        missing_impact = get_team_missing_impact(
            out_info["missing_player_ids"], expected,
            player_col=player_col, minutes_col=minutes_col
        )
 
        rows.append({
            game_id_col: game_id,
            team_col: team,
            "number_of_players_out": out_info["number_of_players_out"],
            "number_of_starters_out": out_info["number_of_starters_out"],
            "team_missing_impact": missing_impact,
        })
 
    return pd.DataFrame(rows)