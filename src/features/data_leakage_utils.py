"""
data_leakage_utils.py

Core utilities for preventing data leakage when building NBA ATS features.

RULE: When building a feature for a game played on `game_date`, you may only
use rows from other tables (team stats, injuries, betting lines, etc.) that
occurred STRICTLY BEFORE `game_date`. Same-day data is disallowed by default
because most of the stats you'd derive from a game (box scores, ratings)
are only finalized after that game ends -- using them would leak the game's
own outcome into its own features.

Usage pattern:

    from src.features.data_leakage_utils import as_of, assert_no_leakage

    # Get all games a team played before a cutoff date
    past_games = as_of(games_df, cutoff_date=game_date, date_col="date",
                        team_col="home_team", team_value="DEN")

    # After building your full feature table, verify it
    assert_no_leakage(features_df,
                       game_date_col="date",
                       source_date_col="source_max_date")
"""

from __future__ import annotations

import pandas as pd


def as_of(
    df: pd.DataFrame,
    cutoff_date,
    date_col: str = "date",
    inclusive: bool = False,
) -> pd.DataFrame:
    """
    Return only rows of `df` that occurred before `cutoff_date`.

    Parameters
    ----------
    df : DataFrame containing a date column (games, box scores, injuries, etc.)
    cutoff_date : the date of the game you are predicting (the "as of" point)
    date_col : name of the date column in df
    inclusive : if True, allows rows ON cutoff_date too. Default False, because
        for a game on date D you should not know D's own results yet.
        Only set True for things that are genuinely known before tip-off
        on game day itself (e.g. the starting lineup announced that morning,
        or the closing spread posted before the game).

    Returns
    -------
    Filtered DataFrame, safe to use as historical context for a game on
    cutoff_date.
    """
    d = pd.to_datetime(df[date_col])
    cutoff = pd.to_datetime(cutoff_date)
    mask = d <= cutoff if inclusive else d < cutoff
    return df.loc[mask].copy()


def team_as_of(
    df: pd.DataFrame,
    team: str,
    cutoff_date,
    date_col: str = "date",
    home_col: str = "home_team",
    away_col: str = "away_team",
    inclusive: bool = False,
) -> pd.DataFrame:
    """
    Return all of a specific team's games (home or away) strictly before
    cutoff_date. This is the building block for rolling stats, home/away
    splits, rest-day calculations, etc.
    """
    filtered = as_of(df, cutoff_date, date_col=date_col, inclusive=inclusive)
    team_mask = (filtered[home_col] == team) | (filtered[away_col] == team)
    return filtered.loc[team_mask].copy()


def build_feature_row(
    game_row: pd.Series,
    games_df: pd.DataFrame,
    stat_builder,
    date_col: str = "date",
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> dict:
    """
    Generic helper: for one row of the schedule (a game to predict),
    build features using ONLY data strictly before that game's date.

    `stat_builder(team, history_df) -> dict` is a function you supply that
    knows how to turn a team's historical games into feature values
    (e.g. rolling net rating). This function guarantees `history_df` never
    contains future information relative to game_row's date.
    """
    game_date = game_row[date_col]
    home_team = game_row[home_col]
    away_team = game_row[away_col]

    home_history = team_as_of(games_df, home_team, game_date, date_col, home_col, away_col)
    away_history = team_as_of(games_df, away_team, game_date, date_col, home_col, away_col)

    home_feats = {f"home_{k}": v for k, v in stat_builder(home_team, home_history).items()}
    away_feats = {f"away_{k}": v for k, v in stat_builder(away_team, away_history).items()}

    return {**home_feats, **away_feats}


def assert_no_leakage(
    features_df: pd.DataFrame,
    game_date_col: str = "date",
    source_date_col: str = "source_max_date",
) -> None:
    """
    Sanity check for a finished feature table.

    Expects the feature-building pipeline to have stamped each row with
    `source_date_col`: the MAX date of any historical data that went into
    that row's features. If any row's source_max_date >= its own game date,
    that row leaked future (or same-day) information.

    Raises AssertionError listing offending game_ids if leakage is found.
    """
    if source_date_col not in features_df.columns:
        raise ValueError(
            f"'{source_date_col}' not found. Your feature pipeline must "
            f"track the max source date used per row for this check to work."
        )

    game_dates = pd.to_datetime(features_df[game_date_col])
    source_dates = pd.to_datetime(features_df[source_date_col])

    leaked = features_df.loc[source_dates >= game_dates]

    if not leaked.empty:
        bad_ids = leaked.get("game_id", leaked.index).tolist()
        raise AssertionError(
            f"Data leakage detected in {len(leaked)} rows. "
            f"Offending game_ids/index: {bad_ids[:20]}"
            f"{' ...' if len(bad_ids) > 20 else ''}"
        )


def add_rest_days(
    games_df: pd.DataFrame,
    date_col: str = "date",
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pd.DataFrame:
    """
    Example of a leak-safe schedule feature: rest days before each game.
    Only ever looks BACKWARD at each team's own prior game date, so it
    cannot leak future information by construction.
    """
    df = games_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # long format: one row per team per game
    home = df[[date_col, home_col]].rename(columns={home_col: "team"})
    home["game_id"] = df.index
    home["is_home"] = True

    away = df[[date_col, away_col]].rename(columns={away_col: "team"})
    away["game_id"] = df.index
    away["is_home"] = False

    long_df = pd.concat([home, away]).sort_values(["team", date_col])
    long_df["prev_game_date"] = long_df.groupby("team")[date_col].shift(1)
    long_df["rest_days"] = (long_df[date_col] - long_df["prev_game_date"]).dt.days
    long_df["back_to_back"] = long_df["rest_days"] == 1

    home_rest = long_df[long_df["is_home"]].set_index("game_id")[["rest_days", "back_to_back"]]
    home_rest.columns = ["home_rest", "home_back_to_back"]

    away_rest = long_df[~long_df["is_home"]].set_index("game_id")[["rest_days", "back_to_back"]]
    away_rest.columns = ["away_rest", "away_back_to_back"]

    df = df.join(home_rest).join(away_rest)
    df["rest_difference"] = df["home_rest"] - df["away_rest"]
    return df