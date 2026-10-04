#historical lookup, needs second path for game that hasn't been played yet

import pandas as pd
from pathlib import Path

from src.features.feature_config import ROLLED, OUTCOME_COLUMNS, TARGETS

GAME_PATH = Path("data/processed/games_clean.parquet")

FORM_WINDOWS = (10, 5, 3)
FORM_STATS = ("net_rating", "offensive_rating", "defensive_rating", "pace")

def get_game_row(df, game_id):
    mask = (df["GAME_ID"] == game_id)
    matches = df[mask]
    if len(matches) == 0:
        raise ValueError(f"{game_id} not found")
    elif len(matches) > 1:
        raise ValueError(f"multiple games found for {game_id}")
    return matches.iloc[0]


def build_market_block(row):
    if pd.isna(row["home_implied_prob"]):
        raise ValueError("home implied probability is NaN")
    return {
        "home_decimal_odds": float(row["home_decimal_odds"]),
        "away_decimal_odds": float(row["away_decimal_odds"]),
        "home_implied_prob": float(row["home_implied_prob"]),
        "away_implied_prob": 1 - float(row["home_implied_prob"]),
        "source": "wippa-nba-data closing moneyline",
        "note": "implied probs are no-vig; decimal odds include the bookmaker margin."
    }

def _num(row, col, cast=float):
    """Read one column as a plain Python number; raise with the GAME_ID on NaN."""
    value = row[col]
    if pd.isna(value):
        raise ValueError(f"{row['GAME_ID']}: NaN in context column '{col}'")
    return cast(value)


def _pair(row, stem, cast=float):
    """{'home': ..., 'away': ...} for a stem that exists as home_<stem> and away_<stem>."""
    return {
        "home": _num(row, f"home_{stem}", cast),
        "away": _num(row, f"away_{stem}", cast),
    }

def build_context_block(row):
    # form: last-N rolling stats, with a difference where the table has one
    form = {}
    for n in FORM_WINDOWS:
        window = {}
        for stat in FORM_STATS:
            entry = _pair(row, f"last_{n}_{stat}")
            diff_col = f"last_{n}_{stat}_difference"
            if diff_col in row.index:  # no 3-game difference columns exist
                entry["difference"] = _num(row, diff_col)
            window[stat] = entry
        form[f"last_{n}"] = window

    # splits: home team at home, away team on the road (l10)
    splits = {
        "home_team_at_home": {
            "net_rating": _num(row, "home_team_home_net_rating_l10"),
            "win_pct": _num(row, "home_team_home_win_pct_l10"),
        },
        "away_team_on_road": {
            "net_rating": _num(row, "away_team_road_net_rating_l10"),
            "win_pct": _num(row, "away_team_road_win_pct_l10"),
        },
    }

    # schedule: rest and load
    schedule = {
        "rest_days": _pair(row, "rest_days", int),
        "is_back_to_back": _pair(row, "is_back_to_back", bool),
        "games_last_7_days": _pair(row, "games_last_7_days", int),
        "games_last_14_days": _pair(row, "games_last_14_days", int),
        "rest_difference": _num(row, "rest_difference", int),
    }

    return {"form": form, "splits": splits, "schedule": schedule}


def build_quality_block(row):
    quality = {}
    for n in (10, 5, 3):
        quality[f"home_last_{n}_games_available"] = _num(row, f"home_last_{n}_games_available", int)
        quality[f"away_last_{n}_games_available"] = _num(row, f"away_last_{n}_games_available", int)

    quality["early_season"] = bool(
        quality["home_last_10_games_available"] < 10
        or quality["away_last_10_games_available"] < 10
    )
    quality["injury_data"] = (
        "unavailable pre-game; absence cannot be assumed to be zero"
    )
    quality["splits_note"] = (
        "early in a season, home/road split values may be fallbacks "
        "(the team's overall last-10 net rating, or 0.5 for win pct) "
        "rather than true home/road splits"
    )
    return quality

def all_keys(d):
    for k, v in d.items():
        yield k
        if isinstance(v, dict):
            yield from all_keys(v)

def predict_game(game_id, df):
    row = get_game_row(df, game_id)
    output = {
        "game_id": str(row['GAME_ID']),
        "game_date": row['GAME_DATE'].date().isoformat(),
        "season": str(row['season']),
        "home_team": str(row['home_TEAM_ABBREVIATION']),
        "away_team": str(row['away_TEAM_ABBREVIATION']),
        "market": build_market_block(row),
        "context": build_context_block(row),
        "data_quality": build_quality_block(row)
    }
    leaked = set(all_keys(output)) & set(OUTCOME_COLUMNS + TARGETS)
    assert not leaked, f"outcome columns in prediction output: {leaked}"
    return output