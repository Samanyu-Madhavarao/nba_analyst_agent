import pandas as pd

# Reuse Step 14's window/threshold so "expected roster" means the same
# thing here as it does in injury_features.py — no reason for these to
# drift apart.
from injury_features import EXPECTED_ROSTER_WINDOW, EXPECTED_ROSTER_THRESHOLD, STARTER_COUNT
import injury_features

PRODUCTION_STATS = ["PTS", "REB", "AST", "usage_rate"]


def add_reb_column(player_long_df: pd.DataFrame) -> pd.DataFrame:
    df = player_long_df.copy()
    df["REB"] = df["OREB"] + df["DREB"]
    return df


def add_player_usage_rate(player_long_df: pd.DataFrame, team_long_df: pd.DataFrame,
                           game_id_col: str = "GAME_ID", team_col: str = "TEAM_ABBREVIATION") -> pd.DataFrame:
    df = player_long_df.copy()

    team_side = team_long_df[[game_id_col, team_col, "pos", "MIN"]].copy()
    team_side = team_side.rename(columns={"MIN": "team_MIN"})

    df = df.merge(team_side, on=[game_id_col, team_col], how="left")

    assert len(df) == len(player_long_df), (
        f"Merge changed row count: {len(player_long_df)} -> {len(df)}. "
        f"Check for duplicate (game_id, team) rows in team_long_df."
    )

    player_min_denom = df["MIN"].replace(0, float("nan"))
    team_pos_denom = df["pos"].replace(0, float("nan"))

    df["usage_rate"] = (
        100 * ((df["FGA"] + 0.44 * df["FTA"] + df["TOV"]) * (df["team_MIN"] / 5))
        / (player_min_denom * team_pos_denom)
    )

    return df



def add_player_rolling_production(player_long_df: pd.DataFrame, team_game_calendar: pd.DataFrame,
                                   player_col: str = "PLAYER_ID",
                                   team_col: str = "TEAM_ABBREVIATION",
                                   season_col: str = "season",
                                   date_col: str = "GAME_DATE",
                                   game_id_col: str = "GAME_ID",
                                   stats: list = PRODUCTION_STATS,
                                   window: int = EXPECTED_ROSTER_WINDOW) -> pd.DataFrame:
    """ Generalizes injury_features.add_player_rolling_appearance_
    and_minutes() to roll multiple stat columns (PTS, REB, AST,
    usage_rate) instead of just MIN. """

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
        player_long_df[[player_col, team_col, season_col, game_id_col, *stats, "appeared"]],
        on=[player_col, team_col, season_col, game_id_col],
        how="left"
    )
    reindexed["appeared"] = reindexed["appeared"].fillna(0)
    reindexed[stats] = reindexed[stats].where(reindexed["appeared"] == 1)
    reindexed = reindexed.sort_values([player_col, team_col, season_col, date_col]).reset_index(drop=True)
    grouped = reindexed.groupby([player_col, team_col, season_col])
    for stat in stats:
        reindexed[f"player_avg_{stat}_last_{window}"] = grouped[stat].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )   

    return reindexed


def get_expected_rotation_production(player_rolling_df, production_rolling_df, team, as_of_date,
                                      player_col: str = "PLAYER_ID",
                                      team_col: str = "TEAM_ABBREVIATION",
                                      date_col: str = "GAME_DATE",
                                      stats: list = PRODUCTION_STATS,
                                      window: int = EXPECTED_ROSTER_WINDOW,
                                      threshold: float = EXPECTED_ROSTER_THRESHOLD) -> pd.DataFrame:
    """
    player_rolling_df: Step 14's output (add_player_rolling_appearance_and_minutes)
    production_rolling_df: Step 15's output (add_player_rolling_production)
    """
    expected = injury_features.get_expected_roster(
        player_rolling_df, team, as_of_date,
        player_col=player_col, team_col=team_col, date_col=date_col,
        window=window, threshold=threshold
    )
    # expected has: [player_col, team_col, minutes_col, "appearance_rate"]

    production_subset = production_rolling_df[
        (production_rolling_df[team_col] == team) & (production_rolling_df[date_col] == as_of_date)
    ]

    stat_cols = [f"player_avg_{stat}_last_{window}" for stat in stats]
    result = expected.merge(
        production_subset[[player_col] + stat_cols],
        on=player_col,
        how="left"
    )

    assert len(result) == len(expected), (
        f"Merge changed row count: {len(expected)} -> {len(result)}. "
        f"player_rolling_df and production_rolling_df may be out of sync."
    )

    return result
    


def get_team_expected_totals(expected_roster_df: pd.DataFrame, stats: list = PRODUCTION_STATS,
                              window: int = EXPECTED_ROSTER_WINDOW) -> dict:
    """ sum each rolled stat across the full expected roster —
    "what this team normally produces" baseline for one team-game. """
    stat_sums = {}
    for stat in stats:
        stat_sums[f"expected_team_{stat}"] = expected_roster_df[f"player_avg_{stat}_last_{window}"].sum()
    return stat_sums


def get_team_missing_production(missing_player_ids: set, expected_roster_df: pd.DataFrame,
                                 stats: list = PRODUCTION_STATS, window: int = EXPECTED_ROSTER_WINDOW,
                                 player_col: str='PLAYER_ID') -> dict:
    """ same shape as injury_features.get_team_missing_impact(),
    generalized to every stat instead of just minutes — sum of trailing
    per-stat production for just the players who didn't show up. """
    mask = expected_roster_df[player_col].isin(missing_player_ids)

    missing_totals = {}
    for stat in stats:
        col = f"player_avg_{stat}_last_{window}"
        missing_totals[f"missing_team_{stat}"] = expected_roster_df.loc[mask, col].sum()

    return missing_totals


def get_production_ratios(expected_totals: dict, missing_totals: dict, stats: list = PRODUCTION_STATS) -> dict:
    production_ratios = {}

    for stat in stats:
        missing_team_stat = missing_totals[f"missing_team_{stat}"]
        expected_team_stat = expected_totals[f"expected_team_{stat}"]

        if expected_team_stat == 0:
            production_ratios[f"pct_{stat}_missing"] = 0.0
        else:
            production_ratios[f"pct_{stat}_missing"] = missing_team_stat / expected_team_stat

    return production_ratios


def build_player_production_features(player_long_df: pd.DataFrame, team_long_df: pd.DataFrame,
                                      player_col: str = "PLAYER_ID",
                                      team_col: str = "TEAM_ABBREVIATION",
                                      season_col: str = "season",
                                      date_col: str = "GAME_DATE",
                                      game_id_col: str = "GAME_ID",
                                      stats: list = PRODUCTION_STATS,
                                      window: int = EXPECTED_ROSTER_WINDOW,
                                      threshold: float = EXPECTED_ROSTER_THRESHOLD) -> pd.DataFrame:
    """ orchestrates 15a-15g across every (team, game) pair, same
    row-by-row-over-calendar pattern as injury_features.
    build_injury_features() (correctness first, vectorize later — see
    that function's docstring for the same reasoning).
 
    Output columns:
        GAME_ID, TEAM_ABBREVIATION,
        expected_team_PTS, missing_team_PTS, pct_PTS_missing,
        expected_team_REB, missing_team_REB, pct_REB_missing,
        expected_team_AST, missing_team_AST, pct_AST_missing,
        expected_team_usage_rate, missing_team_usage_rate, pct_usage_rate_missing """
    # --- Prep: raw columns Step 15's functions assume already exist ---
    player_long_df = add_reb_column(player_long_df)
    player_long_df = add_player_usage_rate(player_long_df, team_long_df,
                                            game_id_col=game_id_col, team_col=team_col)
    player_long_df = injury_features.add_player_appearance_flag(player_long_df)
 
    # --- Shared calendar, same one Step 14 uses ---
    calendar = injury_features.build_team_game_calendar(
        team_long_df, team_col=team_col, season_col=season_col,
        date_col=date_col, game_id_col=game_id_col
    )
 
    # --- Two separate rolling passes (per the earlier decision NOT to
    # fold them): Step 14's minutes/appearance rolling, and Step 15's
    # multi-stat production rolling. ---
    player_rolling = injury_features.add_player_rolling_appearance_and_minutes(
        player_long_df, calendar, player_col=player_col, team_col=team_col,
        season_col=season_col, date_col=date_col, game_id_col=game_id_col,
        window=window
    )
    production_rolling = add_player_rolling_production(
        player_long_df, calendar, player_col=player_col, team_col=team_col,
        season_col=season_col, date_col=date_col, game_id_col=game_id_col,
        stats=stats, window=window
    )
 
    minutes_col = f"player_avg_minutes_last_{window}"
 
    # --- Row-by-row over the calendar, same deferred-vectorization
    # approach as build_injury_features(): correctness first. ---
    rows = []
    for _, cal_row in calendar.iterrows():
        team = cal_row[team_col]
        game_id = cal_row[game_id_col]
        as_of_date = cal_row[date_col]
 
        # Step 14's roster/starters logic, reused as-is — this is a
        # second get_expected_roster() call beyond the one
        # get_expected_rotation_production() makes internally below;
        # kept separate on purpose since get_players_out()/get_starters()
        # need Step 14's minimal roster shape, not Step 15's
        # stats-attached one.
        expected_roster = injury_features.get_expected_roster(
            player_rolling, team, as_of_date, player_col=player_col,
            team_col=team_col, date_col=date_col, window=window, threshold=threshold
        )
        starters = injury_features.get_starters(expected_roster, n=STARTER_COUNT, minutes_col=minutes_col)
        actual_roster = injury_features.get_actual_game_roster(
            player_long_df, game_id, team, game_id_col=game_id_col,
            team_col=team_col, player_col=player_col
        )
        out_info = injury_features.get_players_out(expected_roster, actual_roster, starters, player_col=player_col)
 
        # Step 15's own roster, carrying the rolled production stats.
        expected_rotation = get_expected_rotation_production(
            player_rolling, production_rolling, team, as_of_date,
            player_col=player_col, team_col=team_col, date_col=date_col,
            stats=stats, window=window, threshold=threshold
        )
 
        expected_totals = get_team_expected_totals(expected_rotation, stats=stats, window=window)
        missing_totals = get_team_missing_production(
            out_info["missing_player_ids"], expected_rotation,
            player_col=player_col, stats=stats, window=window
        )
        ratios = get_production_ratios(expected_totals, missing_totals, stats=stats)
 
        row = {game_id_col: game_id, team_col: team}
        row.update(expected_totals)
        row.update(missing_totals)
        row.update(ratios)
        rows.append(row)
 
    return pd.DataFrame(rows)