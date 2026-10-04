# src/features/add_betting_lines.py
"""
Joins closing moneyline odds onto features.parquet and (if a spread column
is configured) adds ATS columns.

Input:   data/processed/features.parquet          (unchanged, all games)
         data/raw/betting_lines/nba_<season>_results_odds.parquet
Output:  data/processed/features_with_odds.parquet (only games with odds)

Run:
    python -m src.features.add_betting_lines
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.features.feature_config import check_coverage

FEATURES_PATH = Path("data/processed/features.parquet")
ODDS_DIR = Path("data/raw/betting_lines")
OUTPUT_PATH = Path("data/processed/features_with_odds.parquet")

# --- Raw odds column names (confirmed from the 2023-24 file) ---
ODDS_DATE_COL = "date"
ODDS_HOME_COL = "home_team"
ODDS_AWAY_COL = "away_team"
ODDS_HOME_ODDS_COL = "home_odds"   # decimal odds, e.g. 1.90
ODDS_AWAY_ODDS_COL = "away_odds"
ODDS_ROUND_COL = "round"
ODDS_ALLSTAR_COL = "is_allstar"

# Rows with these `round` labels are dropped. Check the printed label counts
# on first run and add any other non-regular-season labels you see.
EXCLUDED_ROUNDS = {"Pre-season"}

# Spread is optional. This file has no spread column, so it's off (None).
# If you find a source with one, set the column name here.
ODDS_SPREAD_COL = None
SPREAD_REFERS_TO = "home"  # "home": used as-is (negative = home favored); "away": sign flipped

TEAM_NAME_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

# Seasons believed absent from the source (used only to label the report).
EXPECTED_MISSING_SEASONS = {"2015-16", "2019-20", "2020-21"}

KEY = ["GAME_DATE", "home_TEAM_ABBREVIATION", "away_TEAM_ABBREVIATION"]
MARKET_COLS = ["home_decimal_odds", "away_decimal_odds", "home_implied_prob"]

ODDS_HOME_SCORE_COL = "home_score"
ODDS_AWAY_SCORE_COL = "away_score"
SCORE_COLS = ["odds_home_score", "odds_away_score"]
MAX_SCORE_MISMATCH_RATE = 0.005

# Validation thresholds
MIN_MEAN_OVERROUND = 1.0
MAX_MEAN_OVERROUND = 1.15
MAX_PROB_VS_WINRATE_GAP = 0.05
MIN_PROB_MARGIN_CORR = 0.30     # net rating alone was 0.338; the market should beat it
MAX_ABS_MEAN_ATS_MARGIN = 1.0   # spread checks (only if a spread column exists)
MIN_SPREAD_CORR = 0.35
MAX_ABS_SPREAD = 30
MAX_PUSH_RATE = 0.03

MAX_NON_NBA_ROWS = 40

def load_raw_odds(odds_dir: Path = ODDS_DIR) -> pd.DataFrame:
    """Read every per-season odds file and concatenate. Prints rows per season."""
    files = sorted(odds_dir.glob("nba_*_results_odds.parquet"))
    if not files:
        raise FileNotFoundError(f"No odds files found in {odds_dir}")

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        # e.g. nba_2023-2024_results_odds -> "2023-2024" (label for printing only)
        df["odds_season"] = f.stem.split("_")[1]
        frames.append(df)

    odds = pd.concat(frames, ignore_index=True)
    print("Odds rows per source season:")
    print(odds.groupby("odds_season").size().to_string())
    return odds


def filter_regular_season(odds: pd.DataFrame) -> pd.DataFrame:
    """Drop preseason (and all-star) rows so they don't pollute the match report."""
    df = odds.copy()

    if ODDS_ROUND_COL in df.columns:
        print("\nOdds rows by round label:")
        print(df[ODDS_ROUND_COL].value_counts().to_string())
        df = df[~df[ODDS_ROUND_COL].isin(EXCLUDED_ROUNDS)]

    if ODDS_ALLSTAR_COL in df.columns:
        df = df[~df[ODDS_ALLSTAR_COL].fillna(False).astype(bool)]

    df = df.reset_index(drop=True)
    print(f"\nOdds rows after filtering: {len(df)}")
    return df


def standardize_odds(odds: pd.DataFrame) -> pd.DataFrame:
    """
    Parse dates, map team names, validate decimal odds, and add the no-vig
    home implied probability. Returns KEY + market columns + odds_row_id.
    """
    df = odds.copy()

    df["GAME_DATE"] = pd.to_datetime(df[ODDS_DATE_COL]).dt.normalize()

    known = set(TEAM_NAME_MAP)
    home_names = df[ODDS_HOME_COL].astype(str).str.strip()
    away_names = df[ODDS_AWAY_COL].astype(str).str.strip()
    non_nba = ~(home_names.isin(known) & away_names.isin(known))

    if non_nba.any():
        print(f"\nDropping {non_nba.sum()} rows with non-NBA team names:")
        print(sorted(set(home_names[non_nba]) | set(away_names[non_nba])))
        assert non_nba.sum() <= MAX_NON_NBA_ROWS, (
            f"{non_nba.sum()} rows dropped for unknown team names; "
            f"that's too many to be exhibitions. Likely a naming variant; "
            f"add it to TEAM_NAME_MAP."
        )
        df = df[~non_nba].copy()

    # Team names -> abbreviations; fail loudly and list anything unmapped.
    for src_col, out_col in [(ODDS_HOME_COL, "home_TEAM_ABBREVIATION"),
                             (ODDS_AWAY_COL, "away_TEAM_ABBREVIATION")]:
        names = df[src_col].astype(str).str.strip()
        mapped = names.map(TEAM_NAME_MAP)
        unmapped = sorted(names[mapped.isna()].unique())
        if unmapped:
            raise ValueError(
                f"Unmapped team names in '{src_col}': {unmapped}. "
                f"Add them to TEAM_NAME_MAP."
            )
        df[out_col] = mapped

    # Decimal odds must be > 1.0 (a payout multiplier including the stake).
    home_odds = pd.to_numeric(df[ODDS_HOME_ODDS_COL], errors="coerce")
    away_odds = pd.to_numeric(df[ODDS_AWAY_ODDS_COL], errors="coerce")
    for label, s in [("home", home_odds), ("away", away_odds)]:
        bad = s.notna() & (s <= 1.0)
        assert not bad.any(), f"{bad.sum()} {label} odds values <= 1.0; not decimal odds?"

    df["home_decimal_odds"] = home_odds
    df["away_decimal_odds"] = away_odds

    # No-vig implied home win probability: 1/odds for each side, then
    # normalize so the two probabilities sum to 1 (removes the bookmaker margin).
    raw_home = 1.0 / home_odds
    raw_away = 1.0 / away_odds
    df["home_implied_prob"] = raw_home / (raw_home + raw_away)

    df["odds_home_score"] = pd.to_numeric(df[ODDS_HOME_SCORE_COL], errors="coerce")
    df["odds_away_score"] = pd.to_numeric(df[ODDS_AWAY_SCORE_COL], errors="coerce")
    keep = KEY + MARKET_COLS + SCORE_COLS

    if ODDS_SPREAD_COL is not None:
        spread = pd.to_numeric(df[ODDS_SPREAD_COL], errors="coerce")
        if SPREAD_REFERS_TO == "home":
            df["home_spread"] = spread
        elif SPREAD_REFERS_TO == "away":
            df["home_spread"] = -spread
        else:
            raise ValueError(f"SPREAD_REFERS_TO must be 'home' or 'away', got {SPREAD_REFERS_TO!r}")
        keep = keep + ["home_spread"]

    df["odds_row_id"] = range(len(df))
    df = df[keep + ["odds_row_id"]]

    dupes = df.duplicated(subset=KEY, keep=False)
    if dupes.any():
        raise ValueError(
            f"{dupes.sum()} rows share (date, home, away) in the odds data. "
            f"Inspect before joining."
        )
    return df


def join_to_features(features: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join odds onto features on (date, home, away). Unmatched games get a
    second attempt with odds dates shifted back one day (some sources date late
    games the next calendar day). Returns ALL feature rows, with NaN odds where
    nothing matched, so the match report can see the misses.
    """
    features = features.reset_index(drop=True)
    feature_cols = list(features.columns)

    first = features.merge(odds, on=KEY, how="left", validate="one_to_one")
    first["odds_date_shifted"] = False

    missing = first["home_decimal_odds"].isna()
    if missing.any():
        # Odds rows already claimed by an exact-date match can't be reused.
        used_ids = first.loc[~missing, "odds_row_id"]
        shifted = odds[~odds["odds_row_id"].isin(used_ids)].copy()
        shifted["GAME_DATE"] = shifted["GAME_DATE"] - pd.Timedelta(days=1)

        second = first.loc[missing, feature_cols].merge(
            shifted, on=KEY, how="left", validate="one_to_one"
        )
        second["odds_date_shifted"] = second["home_decimal_odds"].notna()

        first = pd.concat([first.loc[~missing], second], ignore_index=True)

    assert first["odds_row_id"].dropna().is_unique, "An odds row was matched to two games"
    first = first.drop(columns="odds_row_id")

    first = first.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    assert first["GAME_ID"].is_unique, "Join duplicated GAME_IDs"
    assert len(first) == len(features), "Join changed the number of feature rows"
    return first


def report_match_rates(joined: pd.DataFrame) -> None:
    """Per-season matched vs. total. Separates known gaps from partial scrapes."""
    joined = joined.assign(has_odds=joined["home_decimal_odds"].notna())
    summary = joined.groupby("season")["has_odds"].agg(total="size", matched="sum")
    summary["rate"] = summary["matched"] / summary["total"]
    summary["note"] = ""
    for season in summary.index:
        rate = summary.loc[season, "rate"]
        if season in EXPECTED_MISSING_SEASONS and rate == 0:
            summary.loc[season, "note"] = "known gap"
        elif rate < 0.98:
            summary.loc[season, "note"] = "CHECK: partial or unexpected gap"

    print("\nMatch rate by season:")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nRows matched via 1-day date shift: {int(joined['odds_date_shifted'].sum())}")


def validate_moneyline(df: pd.DataFrame) -> None:
    """Sanity checks on the market probabilities. Asserts so bad data fails loudly."""
    overround = 1.0 / df["home_decimal_odds"] + 1.0 / df["away_decimal_odds"]
    mean_prob = df["home_implied_prob"].mean()
    home_win_rate = df["home_win"].astype(int).mean()
    corr = df["home_implied_prob"].corr(df["actual_margin"])

    print("\nMoneyline validation:")
    print(f"  overround min/mean/max: {overround.min():.3f} / {overround.mean():.3f} / {overround.max():.3f}  (expect ~1.03-1.08)")
    print(f"  mean home implied prob: {mean_prob:.3f}   actual home win rate: {home_win_rate:.3f}")
    print(f"  corr(home_implied_prob, actual_margin): {corr:.3f}")

    assert MIN_MEAN_OVERROUND <= overround.mean() <= MAX_MEAN_OVERROUND, (
        f"Mean overround {overround.mean():.3f} is implausible; are these decimal odds?"
    )
    assert abs(mean_prob - home_win_rate) < MAX_PROB_VS_WINRATE_GAP, (
        "Mean implied home probability is far from the actual home win rate; "
        "home/away odds may be swapped or the join is wrong."
    )
    assert corr > MIN_PROB_MARGIN_CORR, (
        f"Implied probability barely correlates with margin ({corr:.3f}); "
        f"check the join and the home/away columns."
    )


def market_log_loss_report(df: pd.DataFrame) -> None:
    """
    The market's own log loss / Brier per season, to compare against the
    model's numbers (val 2023-24 and test 2024-25 onward).
    """
    y_all = df["home_win"].astype(int)
    print("\nMarket (no-vig moneyline) scores by season:")
    print(f"{'season':<10}{'games':>7}{'log loss':>10}{'brier':>8}")
    for season, g in df.groupby("season"):
        y = g["home_win"].astype(int)
        p = g["home_implied_prob"]
        print(f"{season:<10}{len(g):>7}{log_loss(y, p, labels=[0, 1]):>10.4f}{brier_score_loss(y, p):>8.4f}")
    p_all = df["home_implied_prob"]
    print(f"{'all':<10}{len(df):>7}{log_loss(y_all, p_all, labels=[0, 1]):>10.4f}{brier_score_loss(y_all, p_all):>8.4f}")


def add_ats_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Only used when a spread column is configured.
    ats_margin = actual_margin + home_spread (positive = home covered).
    home_cover: 1/0, and NaN on pushes so they can't be silently used as 0.
    """
    df = df.copy()
    df["ats_margin"] = df["actual_margin"] + df["home_spread"]
    df["is_push"] = df["ats_margin"] == 0
    df["home_cover"] = (df["ats_margin"] > 0).astype(float).where(~df["is_push"])
    return df


def validate_spread(df: pd.DataFrame) -> None:
    """Only used when a spread column is configured."""
    mean_ats = df["ats_margin"].mean()
    corr = (-df["home_spread"]).corr(df["actual_margin"])
    max_abs = df["home_spread"].abs().max()
    push_rate = df["is_push"].mean()

    print("\nSpread validation:")
    print(f"  mean(actual_margin + home_spread): {mean_ats:.3f}  (expect ~0)")
    print(f"  corr(-home_spread, actual_margin): {corr:.3f}  (expect > net rating's 0.338)")
    print(f"  max |home_spread|: {max_abs:.1f}")
    print(f"  push rate: {push_rate:.4f}")

    assert abs(mean_ats) < MAX_ABS_MEAN_ATS_MARGIN, "Mean ATS margin far from 0; sign probably flipped."
    assert corr > MIN_SPREAD_CORR, f"Spread/margin correlation {corr:.3f} too low; check sign or join."
    assert max_abs <= MAX_ABS_SPREAD, f"Implausible spread magnitude: {max_abs}"
    assert push_rate < MAX_PUSH_RATE, f"Push rate {push_rate:.3f} is suspiciously high"


def save_features_with_odds(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    """Write the parquet, then reload and re-assert shape and GAME_ID uniqueness."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    reloaded = pd.read_parquet(path)
    assert reloaded.shape == df.shape, f"Saved shape {reloaded.shape} != built shape {df.shape}"
    assert reloaded["GAME_ID"].is_unique
    print(f"\nsaved {reloaded.shape} -> {path}")

def check_scores(matched: pd.DataFrame) -> None:
    """Independent join check: odds-file scores must equal box-score points."""
    home_ok = matched["odds_home_score"] == matched["home_PTS"]
    away_ok = matched["odds_away_score"] == matched["away_PTS"]
    ok = home_ok & away_ok
    bad = matched[~ok]

    print("\nScore check:")
    print(f"  matched games: {len(matched)}   score mismatches: {len(bad)}")
    print("  mismatches by date-shift flag:")
    print(matched.assign(mismatch=~ok).groupby("odds_date_shifted")["mismatch"]
          .agg(games="size", mismatches="sum").to_string())

    if len(bad):
        swapped = ((bad["odds_home_score"] == bad["away_PTS"])
                   & (bad["odds_away_score"] == bad["home_PTS"]))
        print(f"  of the mismatches, {int(swapped.sum())} look like home/away swapped")
        cols = ["GAME_DATE", "home_TEAM_ABBREVIATION", "away_TEAM_ABBREVIATION",
                "home_PTS", "away_PTS", "odds_home_score", "odds_away_score"]
        print(bad[cols].head(20).to_string(index=False))

    rate = len(bad) / len(matched)
    assert rate < MAX_SCORE_MISMATCH_RATE, (
        f"{rate:.3%} of matched games have different scores; the join (or the "
        f"date shift) is attaching odds to the wrong games."
    )
    return ok


if __name__ == "__main__":
    features = pd.read_parquet(FEATURES_PATH)

    odds = filter_regular_season(load_raw_odds())
    odds = standardize_odds(odds)
    joined = join_to_features(features, odds)
    report_match_rates(joined)

    # Inner-join behavior: keep only games that have odds.
    matched = joined.dropna(subset=["home_decimal_odds"]).reset_index(drop=True)
    print(f"\nGames with odds: {len(matched)} of {len(features)}")

    ok = check_scores(matched)
    matched = matched[ok].drop(columns=SCORE_COLS).reset_index(drop=True)

    validate_moneyline(matched)
    market_log_loss_report(matched)

    if ODDS_SPREAD_COL is not None:
        matched = add_ats_columns(matched)
        validate_spread(matched)

    check_coverage(matched.columns.tolist(), include_odds=True)
    save_features_with_odds(matched)

    # TODO: register the new columns in feature_config.py, then run
    # check_coverage() against `matched`:
    #   MARKET_FEATURES: home_decimal_odds, away_decimal_odds, home_implied_prob
    #                    (+ home_spread if a spread column is configured)
    #   TARGETS:         ats_margin, home_cover, is_push  (only with a spread)
    #   IDENTIFIERS:     odds_date_shifted