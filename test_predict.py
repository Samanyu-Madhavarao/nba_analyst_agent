import pandas as pd
import json

from src.prediction.predict import build_context_block
from src.features.feature_config import OUTCOME_COLUMNS, TARGETS
from src.prediction.logistic_regression import FEATURE_PATH
from src.prediction.predict import get_game_row, build_market_block, build_quality_block
from src.prediction.predict import predict_game


df = pd.read_parquet(FEATURE_PATH)
print("GAME_ID dtype:", df["GAME_ID"].dtype, "| example:", df["GAME_ID"].iloc[0])

# Pick a game from a recent season
game_id = df.loc[df["season"] == "2024-25", "GAME_ID"].iloc[0]
row = get_game_row(df, game_id)
print("game:", row["GAME_DATE"], row["home_TEAM_ABBREVIATION"], "vs", row["away_TEAM_ABBREVIATION"])

block = build_market_block(row)
print(block)

# 1. Probabilities sum to 1
assert abs(block["home_implied_prob"] + block["away_implied_prob"] - 1) < 1e-9

# 2. Odds still carry the margin: overround should be around 1.04
overround = 1 / block["home_decimal_odds"] + 1 / block["away_decimal_odds"]
print(f"overround: {overround:.4f}")
assert 1.0 < overround < 1.12

# 3. Plain Python types only (JSON-safe)
for k, v in block.items():
    assert type(v) in (float, str), f"{k} is {type(v)}"

# 4. get_game_row fails loudly on a missing game (2019-20 has no odds)
try:
    get_game_row(df, "0021900001")
    raise SystemExit("FAIL: expected ValueError for missing game")
except ValueError as e:
    print("missing game raised:", e)

# 5. build_market_block fails loudly on a NaN probability
bad = row.copy()
bad["home_implied_prob"] = float("nan")
try:
    build_market_block(bad)
    raise SystemExit("FAIL: expected ValueError for NaN prob")
except ValueError as e:
    print("NaN prob raised:", e)

# 6. Sweep: every game in the table builds a valid block
for _, r in df.iterrows():
    b = build_market_block(r)
    assert 0 < b["home_implied_prob"] < 1
print(f"all {len(df)} games produced a valid market block")

ctx = build_context_block(row)
print(json.dumps(ctx, indent=2)[:1500])

# no outcome/target names leaked as keys anywhere
def all_keys(d):
    for k, v in d.items():
        yield k
        if isinstance(v, dict):
            yield from all_keys(v)
assert not (set(all_keys(ctx)) & set(OUTCOME_COLUMNS + TARGETS))

# sweep: every game builds, and json.dumps works (proves plain types)
for _, r in df.iterrows():
    json.dumps(build_context_block(r))
print("context sweep passed for", len(df), "games")

q = build_quality_block(row)
print(q)

# sweep: builds for every game, JSON-safe, and early_season matches the counts
n_early = 0
for _, r in df.iterrows():
    qb = build_quality_block(r)
    json.dumps(qb)
    expected = qb["home_last_10_games_available"] < 10 or qb["away_last_10_games_available"] < 10
    assert qb["early_season"] == expected
    n_early += qb["early_season"]
print(f"quality sweep passed; {n_early} of {len(df)} games flagged early_season")

out = predict_game(game_id, df)
dumped = json.dumps(out)
print(dumped[:300])

# outcome names must not appear anywhere, keys or values
for banned in ("actual_margin", "home_win", "total_points"):
    assert banned not in dumped, f"{banned} found in output"

# sweep: every game builds and serializes
for gid in df["GAME_ID"]:
    json.dumps(predict_game(gid, df))
print(f"predict_game sweep passed for {len(df)} games")

print("\nALL CHECKS PASSED")