from pathlib import Path
import pandas as pd

from team_strength_features import build_team_stats_long, add_home_away, to_wide
from rolling_stats import add_rolling_stats, add_games_available, handle_missing_rolling_stats
from home_away_features import add_win_column, add_home_road_split_stats, add_win_percentages
from schedule_features import add_rest_days, add_back_to_back, add_rest_difference, add_games_in_window, handle_missing_rest_days
from feature_config import check_coverage, PRE_GAME_FEATURES


GAMES_PATH = Path("data/processed/games_clean.parquet")
INJURY_PATH = Path("data/processed/injury_features.parquet")
PROD_PATH = Path("data/processed/player_production_features.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")

def load_and_filter(path=GAMES_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["GAME_ID"].str.startswith("002")].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    assert df["GAME_ID"].nunique() == 13_209, f"expected 13209 games, got {df['GAME_ID'].nunique()}"
    assert len(df) == 26_418, f"expected 26418 rows, got {len(df)}"
    return df

def build_long_features(df: pd.DataFrame) -> pd.DataFrame:
    df = build_team_stats_long(df)
    df = add_home_away(df)
    df = add_rolling_stats(df)
    df = add_games_available(df)
    df = add_win_column(df)
    df = add_home_road_split_stats(df)
    df = add_win_percentages(df)
    df = add_rest_days(df)
    df = add_back_to_back(df)
    df = add_games_in_window(df)
    assert len(df) == 26_418, f"row count changed: {len(df)}"
    return df

def merge_injury_production(df: pd.DataFrame, injury_path=INJURY_PATH, prod_path=PROD_PATH) -> pd.DataFrame:
    injury = pd.read_parquet(injury_path)
    prod = pd.read_parquet(prod_path)
    key_cols = ["GAME_ID", "TEAM_ABBREVIATION"]

    for name, frame in [("injury", injury), ("prod", prod)]:
        missing = [c for c in key_cols if c not in frame.columns]
        assert not missing, f"{name} is missing key columns: {missing}"

        collisions = (set(frame.columns) - set(key_cols)) & set(df.columns)
        assert not collisions, f"{name} columns already exist in df: {collisions}"

    df = df.merge(injury, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left", validate="one_to_one")
    df = df.merge(prod, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left", validate="one_to_one")

    new_cols = [c for frame in (injury, prod) for c in frame.columns if c not in key_cols]
    assert len(df) == 26_418, f"row count changed: {len(df)}"
    assert len(new_cols) == 15, f"expected 15 new columns, got {len(new_cols)}"
    nan_total = df[new_cols].isna().sum().sum()
    assert nan_total == 0, f"{nan_total} NaNs in injury/prod columns, likely a key mismatch"

    return df

def clean_split_columns(wide: pd.DataFrame) -> pd.DataFrame:
    to_drop = []
    renames = {}

    for stat in ["net_rating", "win_pct"]:
        home_real  = f"home_home_last_10_{stat}"   # home team's home split (real)
        home_empty = f"home_road_last_10_{stat}"   # home team's road split (all NaN)
        away_empty = f"away_home_last_10_{stat}"   # away team's home split (all NaN)
        away_real  = f"away_road_last_10_{stat}"   # away team's road split (real)
        assert wide[home_empty].isna().all(), f"{home_empty} is not all NaN"
        assert wide[away_empty].isna().all(), f"{away_empty} is not all NaN"
        to_drop.extend([home_empty, away_empty])
        renames[home_real] = f"home_team_home_{stat}_l10"
        renames[away_real] = f"away_team_road_{stat}_l10"

    wide = wide.rename(columns=renames)
    wide = wide.drop(columns=to_drop)

    assert (wide["home_GAME_DATE"] == wide["away_GAME_DATE"]).all()
    wide["GAME_DATE"] = wide["home_GAME_DATE"]
    wide["season"] = wide["home_season"]
    wide = wide.drop(columns=["home_GAME_DATE", "away_GAME_DATE", "home_season", "away_season"])

    return wide

def fill_split_fallbacks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    home_mask = df["is_home"]
    road_mask = ~df["is_home"]

    # net rating: fall back to the team's overall rolled value
    df.loc[home_mask, "home_last_10_net_rating"] = (
        df.loc[home_mask, "home_last_10_net_rating"]
        .fillna(df.loc[home_mask, "last_10_net_rating"])
    )
    df.loc[road_mask, "road_last_10_net_rating"] = (
        df.loc[road_mask, "road_last_10_net_rating"]
        .fillna(df.loc[road_mask, "last_10_net_rating"])
    )

    # win pct: no overall rolled win pct exists, so use a neutral 0.5
    df.loc[home_mask, "home_last_10_win_pct"] = df.loc[home_mask, "home_last_10_win_pct"].fillna(0.5)
    df.loc[road_mask, "road_last_10_win_pct"] = df.loc[road_mask, "road_last_10_win_pct"].fillna(0.5)

    home_cols = ["home_last_10_net_rating", "home_last_10_win_pct"]
    road_cols = ["road_last_10_net_rating", "road_last_10_win_pct"]

    assert df.loc[home_mask, home_cols].isna().sum().sum() == 0, "NaNs left in home split cols on home rows"
    assert df.loc[road_mask, road_cols].isna().sum().sum() == 0, "NaNs left in road split cols on road rows"
    return df


def build_wide_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    games_before = df["GAME_ID"].nunique()

    df = handle_missing_rest_days(df, strategy="fill_default")
    df = handle_missing_rolling_stats(df, strategy="drop")
    df = fill_split_fallbacks(df)

    games_after = df["GAME_ID"].nunique()
    print(f"games before: {games_before}, after: {games_after}, rows: {len(df)}")

    wide = to_wide(df)
    assert wide["GAME_ID"].is_unique
    assert 12_879 <= len(wide) <= 13_044, f"unexpected wide row count: {len(wide)}"
    print(games_before - len(wide))

    wide = clean_split_columns(wide)
    split_cols = [
        "home_team_home_net_rating_l10",
        "home_team_home_win_pct_l10",
        "away_team_road_net_rating_l10",
        "away_team_road_win_pct_l10",
    ]
    n_nan = int(wide[split_cols].isna().sum().sum())
    assert n_nan == 0, f"{n_nan} NaNs remain in split columns after fill"
    
    return wide

def add_targets(wide: pd.DataFrame) -> pd.DataFrame:
    wide = wide.copy()
    wide["actual_margin"] = wide["home_PTS"] - wide["away_PTS"]
    wide["home_win"] = (wide["actual_margin"] > 0).astype(int)
    wide["total_points"] = wide["home_PTS"] + wide["away_PTS"]
    home_won = wide["home_WL"] == "W"
    assert (wide["actual_margin"] > 0).eq(home_won).all(), "margin sign disagrees with home_WL"
    assert (wide["home_WL"] != wide["away_WL"]).all(), "both teams have the same WL"
    assert wide["actual_margin"].ne(0).all(), "found a tie"

    return wide

def add_pregame_differences(wide, stats, windows):
    wide = wide.copy()
    new_cols = []
    for n in windows:
        for stat in stats:
            stem = f"last_{n}_{stat}"
            name = f"{stem}_difference"
            wide[name] = wide[f"home_{stem}"] - wide[f"away_{stem}"]
            new_cols.append(name)
    return wide

def correlation_check(wide, features, target="actual_margin", top_n=10):
    X = wide[features].astype(float)
    corr = X.corrwith(wide[target])
    ranked = corr.reindex(corr.abs().sort_values(ascending=False).index)
    print(ranked.head(top_n).to_string())
    return ranked

def save_features(wide: pd.DataFrame, path=FEATURES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(path, index=False)

    saved = pd.read_parquet(path)
    assert saved.shape == wide.shape, f"shape mismatch: {saved.shape} vs {wide.shape}"
    assert saved["GAME_ID"].is_unique, "duplicate GAME_IDs in saved file"
    assert saved["GAME_DATE"].dtype == wide["GAME_DATE"].dtype, "GAME_DATE dtype changed"
    print(f"saved {saved.shape} -> {path}")

if __name__ == "__main__":
    df = load_and_filter()
    df = build_long_features(df)
    df = merge_injury_production(df)
    wide = build_wide_table(df)
    wide = add_targets(wide)
    wide = add_pregame_differences(
        wide,
        stats=["net_rating", "offensive_rating", "defensive_rating", "pace"],
        windows=[10, 5],
    )
    wide = add_rest_difference(wide)
    check_coverage(wide.columns.tolist())
    ranked = correlation_check(wide, PRE_GAME_FEATURES)
    save_features(wide)
