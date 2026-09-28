"""
test_injury_and_production_features.py

Verification script for Step 14 (injury_features.py) and Step 15
(player_production_features.py), run against REAL data.

Two scope modes, controlled by SCOPE_TEAM below:
- SCOPE_TEAM set to a team abbreviation (e.g. "DAL"): scopes to just that
  team's games in SCOPE_SEASON — fast, good for iterating on a fix.
- SCOPE_TEAM = None: scopes to EVERY team's games in SCOPE_SEASON (a full
  season, ~30 teams, ~1,230 games) — use this once single-team scopes
  have passed cleanly, to get real full-scale timing before committing
  to the full 10-season run.

Per-team diagnostics (spot check, first-game-of-season, roster
consistency, inspect_expected_roster) always need a specific team to
point at, regardless of scope mode — see SPOT_CHECK_TEAMS below.

Usage:
    python test_injury_and_production_features.py

Expects these to already exist on disk (adjust paths below to match your
project):
    data/raw/games.parquet
    data/raw/player_boxscores.parquet
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "features"))

import pandas as pd

from team_strength_features import build_team_stats_long, add_home_away
from rolling_stats import add_season_column
from injury_features import build_injury_features, EXPECTED_ROSTER_WINDOW, EXPECTED_ROSTER_THRESHOLD
from player_production_features import build_player_production_features, PRODUCTION_STATS

GAMES_PATH = Path("data/raw/games.parquet")
PLAYER_BOXSCORES_PATH = Path("data/raw/player_boxscores.parquet")

SCOPE_SEASON = "2024-25"

# Set to a team abbreviation (e.g. "DAL") to scope to one team's games.
# Set to None to scope to the FULL SEASON, every team — the next
# scale-up step before the full 10-season run.
SCOPE_TEAM = None

# Which team(s) the per-team diagnostics (spot check, first-game-of-
# season, roster consistency, inspect_expected_roster) run against.
# These need real teams regardless of scope mode. With SCOPE_TEAM=None,
# picking a few different teams here gets more coverage per run instead
# of needing a separate run per team.
SPOT_CHECK_TEAMS = ["DAL", "LAL", "BOS"]


# ----------------------------------------------------------------------
# 1. Data loading, scoped
# ----------------------------------------------------------------------

def load_scoped_data(team: str = SCOPE_TEAM, season: str = SCOPE_SEASON):
    """
    Loads real data and cuts it down to one season (all teams if team is
    None, or one team's games — plus every OTHER team they played, since
    team-level features need both sides of each game — if team is set).
    Keeps player_long_df matched to the same game set.
    """
    games_df = pd.read_parquet(GAMES_PATH)
    games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"])

    games_df = games_df[games_df["GAME_ID"].str[:3] == "002"].copy()

    team_long = build_team_stats_long(games_df)
    team_long = add_season_column(team_long, date_col="GAME_DATE")

    season_mask = team_long["season"] == season
    if team is not None:
        scope_game_ids = team_long.loc[
            (team_long["TEAM_ABBREVIATION"] == team) & season_mask, "GAME_ID"
        ].unique()
    else:
        # Full season, every team — every game_id played in this season.
        scope_game_ids = team_long.loc[season_mask, "GAME_ID"].unique()

    team_long_scoped = team_long[team_long["GAME_ID"].isin(scope_game_ids)].copy()

    player_long = pd.read_parquet(PLAYER_BOXSCORES_PATH)
    player_long["GAME_DATE"] = pd.to_datetime(player_long["GAME_DATE"])
    player_long_scoped = player_long[player_long["GAME_ID"].isin(scope_game_ids)].copy()
    player_long_scoped = add_season_column(player_long_scoped, date_col="GAME_DATE")

    scope_label = team if team is not None else "ALL TEAMS"
    print(f"Scoped to {scope_label} / {season}: {len(scope_game_ids)} games, "
          f"{len(team_long_scoped)} team-game rows, {len(player_long_scoped)} player-game rows")

    return team_long_scoped, player_long_scoped


# ----------------------------------------------------------------------
# 2. Run the pipelines, timed
# ----------------------------------------------------------------------

def run_pipelines(team_long_scoped: pd.DataFrame, player_long_scoped: pd.DataFrame):
    t0 = time.time()
    injury_out = build_injury_features(player_long_scoped, team_long_scoped)
    t1 = time.time()
    print(f"build_injury_features: {t1 - t0:.1f}s for {len(injury_out)} team-game rows "
          f"({(t1 - t0) / max(len(injury_out), 1):.3f}s/row)")

    t2 = time.time()
    production_out = build_player_production_features(player_long_scoped, team_long_scoped)
    t3 = time.time()
    print(f"build_player_production_features: {t3 - t2:.1f}s for {len(production_out)} team-game rows "
          f"({(t3 - t2) / max(len(production_out), 1):.3f}s/row)")

    return injury_out, production_out


# ----------------------------------------------------------------------
# 3. Structural sanity checks (already team-agnostic — operate on the
# whole output dataframe regardless of how many teams are in scope)
# ----------------------------------------------------------------------

def check_row_counts(team_long_scoped: pd.DataFrame, injury_out: pd.DataFrame, production_out: pd.DataFrame):
    expected_rows = team_long_scoped[["GAME_ID", "TEAM_ABBREVIATION"]].drop_duplicates().shape[0]
    print(f"\n--- Row counts ---")
    print(f"Expected (team, game) rows: {expected_rows}")
    print(f"injury_out rows: {len(injury_out)}  {'OK' if len(injury_out) == expected_rows else 'MISMATCH'}")
    print(f"production_out rows: {len(production_out)}  "
          f"{'OK' if len(production_out) == expected_rows else 'MISMATCH'}")

    for name, df in [("injury_out", injury_out), ("production_out", production_out)]:
        dup = df.duplicated(subset=["GAME_ID", "TEAM_ABBREVIATION"]).sum()
        print(f"{name} duplicate (game,team) rows: {dup}  {'OK' if dup == 0 else 'PROBLEM'}")


def check_nan_patterns(injury_out: pd.DataFrame, production_out: pd.DataFrame):
    print(f"\n--- NaN counts ---")
    print("injury_out:")
    print(injury_out.isna().sum())
    print("\nproduction_out:")
    print(production_out.isna().sum())
    print("\nExpect a small cluster of NaNs early in the season (no trailing "
          "history yet) — same pattern as Step 13's verification. With a full "
          "season in scope, expect roughly one NaN cluster PER TEAM at the "
          "season's start, not just one overall. Anything NaN mid-season or "
          "clustered oddly on one specific team/stat is worth investigating.")


def check_value_ranges(production_out: pd.DataFrame):
    print(f"\n--- Value range sanity ---")
    for stat in PRODUCTION_STATS:
        for prefix in ["expected_team", "missing_team"]:
            col = f"{prefix}_{stat}"
            if col not in production_out.columns:
                continue
            negatives = (production_out[col] < 0).sum()
            infs = production_out[col].isin([float("inf"), float("-inf")]).sum()
            print(f"{col}: min={production_out[col].min():.2f} max={production_out[col].max():.2f} "
                  f"negatives={negatives} infs={infs} "
                  f"{'PROBLEM' if negatives or infs else 'OK'}")

    for stat in PRODUCTION_STATS:
        col = f"pct_{stat}_missing"
        if col not in production_out.columns:
            continue
        out_of_range = ((production_out[col] < 0) | (production_out[col] > 1)).sum()
        print(f"{col}: min={production_out[col].min():.3f} max={production_out[col].max():.3f} "
              f"out_of_[0,1]_range={out_of_range} rows "
              f"{'check these rows' if out_of_range else 'OK'}")


# ----------------------------------------------------------------------
# 4. Hand-verified spot check — pick ONE real game, eyeball it
# ----------------------------------------------------------------------

def spot_check_one_game(injury_out: pd.DataFrame, production_out: pd.DataFrame,
                         player_long_scoped: pd.DataFrame, team: str, game_id: str):
    """
    Print everything the pipeline computed for one (team, game) so you can
    manually compare against what you know actually happened (box score,
    news, memory). This is the check nothing else here substitutes for.
    """
    print(f"\n--- Spot check: {team}, game {game_id} ---")

    injury_row = injury_out[(injury_out["GAME_ID"] == game_id) & (injury_out["TEAM_ABBREVIATION"] == team)]
    production_row = production_out[(production_out["GAME_ID"] == game_id) & (production_out["TEAM_ABBREVIATION"] == team)]

    print("injury_features row:")
    print(injury_row.to_string(index=False))
    print("\nplayer_production_features row:")
    print(production_row.to_string(index=False))

    actual_players = player_long_scoped[
        (player_long_scoped["GAME_ID"] == game_id) & (player_long_scoped["TEAM_ABBREVIATION"] == team)
    ]["PLAYER_ID"].tolist()
    print(f"\nPlayers who actually appeared in this game's box score for {team}: {actual_players}")
    print("Manually cross-check: does number_of_players_out look right given who you know "
          "was actually missing? Does missing_team_PTS look plausible for that player's "
          "normal scoring?")


# ----------------------------------------------------------------------
# 5. Leakage check — adapted from rolling_stats.test_no_leakage()
# ----------------------------------------------------------------------

def check_no_leakage_production(player_long_scoped: pd.DataFrame, team_game_calendar: pd.DataFrame,
                                 stat: str = "PTS", window: int = EXPECTED_ROSTER_WINDOW):
    """
    Same spirit as rolling_stats.test_no_leakage(): pick one player/date
    with enough prior games, hand-recompute the rolling average from
    strictly-earlier games, and assert it matches what the pipeline
    produced.
    """
    from player_production_features import add_reb_column, add_player_rolling_production
    from injury_features import add_player_appearance_flag

    df = add_reb_column(player_long_scoped)
    df = add_player_appearance_flag(df)
    df = add_player_rolling_production(df, team_game_calendar, stats=[stat], window=window)

    rolling_col = f"player_avg_{stat}_last_{window}"
    game_number = df.groupby(["PLAYER_ID", "season"]).cumcount() + 1
    candidates = df[(game_number > window) & df[rolling_col].notna()]

    if candidates.empty:
        print(f"\n--- Leakage check ({stat}) ---\nNo candidate row found with {window}+ prior "
              f"games — try a wider scope (more of the season) or a smaller window.")
        return

    target = candidates.iloc[0]
    target_player, target_team, target_season, target_date = (
        target["PLAYER_ID"], target["TEAM_ABBREVIATION"], target["season"], target["GAME_DATE"]
    )

    prior_games = df[
        (df["PLAYER_ID"] == target_player)
        & (df["season"] == target_season)
        & (df["GAME_DATE"] < target_date)
    ].sort_values("GAME_DATE")

    manual_value = prior_games[stat].tail(window).mean()
    computed_value = target[rolling_col]

    print(f"\n--- Leakage check ({stat}) ---")
    print(f"Player {target_player}, team {target_team}, date {target_date}")
    print(f"Manual: {manual_value:.3f}  Computed: {computed_value:.3f}")
    assert abs(manual_value - computed_value) < 1e-9, "LEAKAGE CHECK FAILED — do not trust this pipeline yet."
    print("PASSED")


# ----------------------------------------------------------------------
# 6. Edge cases — force them rather than waiting to hit them by accident
# ----------------------------------------------------------------------

def check_first_game_of_season(injury_out: pd.DataFrame, production_out: pd.DataFrame,
                                team_game_calendar: pd.DataFrame, team: str):
    print(f"\n--- Edge case: {team}'s first game of the season ---")
    first_game = team_game_calendar[team_game_calendar["TEAM_ABBREVIATION"] == team].sort_values("GAME_DATE").iloc[0]
    game_id = first_game["GAME_ID"]

    injury_row = injury_out[(injury_out["GAME_ID"] == game_id) & (injury_out["TEAM_ABBREVIATION"] == team)]
    production_row = production_out[(production_out["GAME_ID"] == game_id) & (production_out["TEAM_ABBREVIATION"] == team)]

    print(f"game_id={game_id}")
    print(injury_row.to_string(index=False))
    print(production_row.to_string(index=False))
    print("Confirm this ran without raising (divide-by-zero guards exercised) "
          "and check whether the values look like reasonable defaults rather than NaN/inf.")


def check_no_missing_players_case(injury_out: pd.DataFrame, production_out: pd.DataFrame):
    print(f"\n--- Edge case: games with nobody missing ---")
    clean_games = injury_out[injury_out["number_of_players_out"] == 0]
    print(f"{len(clean_games)} team-games with nobody missing")
    if clean_games.empty:
        print("None found in this scope — try widening SCOPE_SEASON.")
        return

    sample = clean_games.iloc[0]
    matching_production = production_out[
        (production_out["GAME_ID"] == sample["GAME_ID"]) & (production_out["TEAM_ABBREVIATION"] == sample["TEAM_ABBREVIATION"])
    ]
    missing_cols = [c for c in matching_production.columns if c.startswith("missing_team_")]
    all_zero = (matching_production[missing_cols] == 0).all(axis=1).all()
    print(f"All missing_team_* columns are 0.0 (not NaN) for this row: {all_zero} "
          f"{'OK' if all_zero else 'PROBLEM'}")


# ----------------------------------------------------------------------
# 7. Cross-file consistency — do Step 14 and Step 15's rosters agree?
# ----------------------------------------------------------------------

def check_roster_consistency(player_long_scoped: pd.DataFrame, team_game_calendar: pd.DataFrame,
                              team: str, n_samples: int = 5):
    from injury_features import add_player_appearance_flag, add_player_rolling_appearance_and_minutes
    from player_production_features import add_reb_column, add_player_rolling_production

    print(f"\n--- Cross-file roster consistency ({team}) ---")

    df14 = add_player_appearance_flag(player_long_scoped)
    player_rolling = add_player_rolling_appearance_and_minutes(df14, team_game_calendar)

    df15 = add_reb_column(player_long_scoped)
    df15 = add_player_appearance_flag(df15)
    production_rolling = add_player_rolling_production(df15, team_game_calendar, stats=["PTS", "REB", "AST"])

    dates = team_game_calendar[team_game_calendar["TEAM_ABBREVIATION"] == team]["GAME_DATE"].sort_values().unique()
    sample_dates = dates[-n_samples:] if len(dates) >= n_samples else dates

    for d in sample_dates:
        players_14 = set(player_rolling[(player_rolling["TEAM_ABBREVIATION"] == team) & (player_rolling["GAME_DATE"] == d)]["PLAYER_ID"])
        players_15 = set(production_rolling[(production_rolling["TEAM_ABBREVIATION"] == team) & (production_rolling["GAME_DATE"] == d)]["PLAYER_ID"])
        match = players_14 == players_15
        print(f"{d}: Step14 players={len(players_14)} Step15 players={len(players_15)} "
              f"{'MATCH' if match else 'MISMATCH — investigate before trusting merged output'}")


def inspect_expected_roster(player_rolling_df, team, as_of_date, player_long_df,
                             window: int = EXPECTED_ROSTER_WINDOW, threshold: float = EXPECTED_ROSTER_THRESHOLD):
    from injury_features import get_expected_roster

    roster = get_expected_roster(player_rolling_df, team, as_of_date, window=window, threshold=threshold)

    names = player_long_df[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates()
    roster = roster.merge(names, on="PLAYER_ID", how="left")

    minutes_col = f"player_avg_minutes_last_{window}"
    roster = roster.sort_values(minutes_col, ascending=False)

    print(f"\nExpected roster for {team} on {as_of_date} ({len(roster)} players):")
    print(roster[["PLAYER_NAME", "appearance_rate", minutes_col]].to_string(index=False))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    team_long_scoped, player_long_scoped = load_scoped_data()

    print(team_long_scoped["GAME_ID"].str[:3].value_counts())

    injury_out, production_out = run_pipelines(team_long_scoped, player_long_scoped)

    check_row_counts(team_long_scoped, injury_out, production_out)
    check_nan_patterns(injury_out, production_out)
    check_value_ranges(production_out)

    from injury_features import build_team_game_calendar, add_player_appearance_flag, add_player_rolling_appearance_and_minutes
    calendar = build_team_game_calendar(team_long_scoped)

    player_long_flagged = add_player_appearance_flag(player_long_scoped)
    player_rolling = add_player_rolling_appearance_and_minutes(player_long_flagged, calendar)

    check_no_leakage_production(player_long_scoped, calendar)
    check_no_missing_players_case(injury_out, production_out)

    # Per-team diagnostics need a real team regardless of scope mode —
    # loop over SPOT_CHECK_TEAMS to get coverage across several teams
    # in one run, rather than needing a separate run per team.
    for check_team in SPOT_CHECK_TEAMS:
        if check_team not in team_long_scoped["TEAM_ABBREVIATION"].unique():
            print(f"\n{check_team} not in current scope — skipping its per-team diagnostics.")
            continue

        sample_game_id = calendar[calendar["TEAM_ABBREVIATION"] == check_team].sort_values("GAME_DATE").iloc[-1]["GAME_ID"]
        spot_check_one_game(injury_out, production_out, player_long_scoped, check_team, sample_game_id)

        check_first_game_of_season(injury_out, production_out, calendar, check_team)
        check_roster_consistency(player_long_scoped, calendar, check_team)

    print("\n--- Done. Review every 'PROBLEM'/'MISMATCH'/failed assert above before "
          "trusting Step 14 or Step 15 against the full dataset. ---")