from pathlib import Path
import pandas as pd

GAMES_PATH = Path("data/processed/games_clean.parquet")
INJURY_PATH = Path("data/processed/injury_features.parquet")
EXPECTED_GAMES = 13_209  # from the Step 14/15 full run


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


df = pd.read_parquet(GAMES_PATH)
inj = pd.read_parquet(INJURY_PATH)

# 1. Columns and dtypes
section("1. COLUMNS AND DTYPES")
print(df.dtypes.to_string())

# 2. Sizes and GAME_ID
section("2. SIZES AND GAME_ID")
print(f"rows:              {len(df)}")
print(f"unique GAME_IDs:   {df['GAME_ID'].nunique()}")
print(f"GAME_ID dtype:     {df['GAME_ID'].dtype}")
print(f"sample GAME_IDs:   {df['GAME_ID'].head(3).tolist()}")
print(f"rows per game_id (value counts):\n{df.groupby('GAME_ID').size().value_counts().to_string()}")

# 3. Season type: dedicated column if present, plus GAME_ID prefix
section("3. SEASON TYPE")
candidate_cols = [c for c in df.columns if "season" in c.lower() or "type" in c.lower()]
print(f"columns that look season/type related: {candidate_cols}")

gid = df["GAME_ID"].astype(str).str.zfill(10)
prefix = gid.str[:3]
print("\nGAME_ID prefix counts (team-game rows):")
print(prefix.value_counts().to_string())
print("\nUnique games per prefix (002=regular, 003=all-star, 004=playoffs, 005=play-in, 001=preseason):")
print(df.assign(prefix=prefix).groupby("prefix")["GAME_ID"].nunique().to_string())

# 4. Injury parquet comparison
section("4. INJURY_FEATURES.PARQUET COMPARISON")
print(f"injury GAME_ID dtype: {inj['GAME_ID'].dtype}")
print(f"injury sample GAME_IDs: {inj['GAME_ID'].head(3).tolist()}")
print(f"injury rows: {len(inj)}, unique games: {inj['GAME_ID'].nunique()}")

games_ids = set(gid)
inj_ids = set(inj["GAME_ID"].astype(str).str.zfill(10))
print(f"\nGAME_IDs in games_clean but not in injury: {len(games_ids - inj_ids)}")
print(f"GAME_IDs in injury but not in games_clean: {len(inj_ids - games_ids)}")

reg_ids = set(gid[prefix == "002"])
print(f"\nRegular-season (002) games in games_clean: {len(reg_ids)}  (target: {EXPECTED_GAMES})")
print(f"Regular-season IDs missing from injury:     {len(reg_ids - inj_ids)}")
print(f"Injury IDs not in regular-season set:       {len(inj_ids - reg_ids)}")
print(f"Filter matches Step 14/15 exactly: {reg_ids == inj_ids}")

# Extras useful for the next step
section("5. EXTRAS")
if "GAME_DATE" in df.columns:
    print(f"GAME_DATE dtype: {df['GAME_DATE'].dtype}")
    d = pd.to_datetime(df["GAME_DATE"])
    print(f"date range: {d.min().date()} -> {d.max().date()}")
if "MATCHUP" in df.columns:
    print(f"MATCHUP sample: {df['MATCHUP'].head(2).tolist()}")