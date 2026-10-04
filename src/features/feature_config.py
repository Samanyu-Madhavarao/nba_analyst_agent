SIDES = ["home", "away"]

def both(stems):
    return [f"{side}_{s}" for side in SIDES for s in stems]

IDENTIFIERS = ["GAME_ID", "GAME_DATE", "season"] + both([
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
    "MATCHUP", "opp_TEAM_ABBREVIATION",
])

ROLLED = both(
    [f"last_{n}_{stat}" for n in (3, 5, 10)
     for stat in ("net_rating", "offensive_rating", "defensive_rating", "pace")]
    + [f"last_{n}_games_available" for n in (3, 5, 10)]
)

SPLITS = [
    "home_team_home_net_rating_l10", "home_team_home_win_pct_l10",
    "away_team_road_net_rating_l10", "away_team_road_win_pct_l10",
]

SCHEDULE = both(["rest_days", "is_back_to_back",
                 "games_last_7_days", "games_last_14_days"])

EXPECTED_ROSTER_FEATURES = both([
    "expected_team_PTS", "expected_team_REB",
    "expected_team_AST", "expected_team_usage_rate",
])

ACTUAL_ABSENCE_FEATURES = both([
    "number_of_players_out", "number_of_starters_out", "team_missing_impact",
    "missing_team_PTS", "missing_team_REB", "missing_team_AST", "missing_team_usage_rate",
    "pct_PTS_missing", "pct_REB_missing", "pct_AST_missing", "pct_usage_rate_missing",
])

DIFFERENCES = [f"last_{n}_{stat}_difference"
               for n in (10, 5)
               for stat in ("net_rating", "offensive_rating",
                            "defensive_rating", "pace")] + ["rest_difference"]

PRE_GAME_FEATURES = (ROLLED + SPLITS + SCHEDULE
                     + EXPECTED_ROSTER_FEATURES + ACTUAL_ABSENCE_FEATURES + DIFFERENCES)

OUTCOME_COLUMNS = both([
    # raw box score (21)
    "WL", "MIN", "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK",
    "TOV", "PF", "PLUS_MINUS",
    # opponent raw box score (10)
    "opp_MIN", "opp_PTS", "opp_FGM", "opp_FGA", "opp_FG3M", "opp_FG3A",
    "opp_FTA", "opp_OREB", "opp_DREB", "opp_TOV",
    # possessions (2)
    "pos", "opp_pos",
    # un-rolled step 9 stats (14)
    "offensive_rating", "defensive_rating", "net_rating",
    "effective_fg_percentage", "true_shooting_percentage",
    "offensive_rebound_pct", "defensive_rebound_pct", "turnover_rate", "pace",
    "three_point_attempt_rate", "free_throw_rate",
    "opponent_effective_fg_percentage", "opponent_three_point_percentage",
    "opponent_turnover_rate",
    # result flag (1)
    "team_win",
])

MARKET_FEATURES = [
    "home_decimal_odds", "away_decimal_odds", "home_implied_prob"
]

ODDS_IDENTIFIERS = ["odds_date_shifted"]

TARGETS = ["actual_margin", "home_win", "total_points"]

def check_coverage(columns, include_odds=False):
    groups = {"identifiers": IDENTIFIERS, "pre_game": PRE_GAME_FEATURES,
              "outcome": OUTCOME_COLUMNS, "targets": TARGETS}
    if include_odds:
        groups["market"] = MARKET_FEATURES
        groups["odds_identifiers"] = ODDS_IDENTIFIERS
    all_named = [c for g in groups.values() for c in g]

    dupes = {c for c in all_named if all_named.count(c) > 1}
    assert not dupes, f"columns in more than one group: {dupes}"

    unclassified = set(columns) - set(all_named)
    assert not unclassified, f"unclassified columns: {unclassified}"

    phantom = set(all_named) - set(columns) - set(TARGETS)
    assert not phantom, f"named but not in table: {phantom}"