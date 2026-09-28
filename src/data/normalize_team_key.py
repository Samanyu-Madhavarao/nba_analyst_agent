import pandas as pd

games = pd.read_parquet("data/processed/games_one_row.parquet")
lines = pd.read_parquet("data/raw/lines_v2.parquet")

games["home_team_code"] = games["HOME_TEAM_ABBREVIATION"]
games["away_team_code"] = games["AWAY_TEAM_ABBREVIATION"]
games["game_date"] = pd.to_datetime(games["GAME_DATE"]).dt.date

print("games date range:", games["game_date"].min(), "to", games["game_date"].max())
print("lines date range:", lines["game_date"].min(), "to", lines["game_date"].max())

merged = games.merge(
    lines,
    on=["game_date", "home_team_code", "away_team_code"],
    how="left",
    validate="1:1"
)

missing = merged["home_spread"].isna().sum()
print(f"{missing} / {len(merged)} games missing a matched spread")

overlap_season = merged[merged["game_date"].between(pd.Timestamp("2022-10-01").date(), pd.Timestamp("2023-06-30").date())]
still_missing = overlap_season["home_spread"].isna()
print(f"{still_missing.sum()} / {len(overlap_season)} missing within the overlapping season")
print(overlap_season[still_missing])

missing_games = merged[merged["home_spread"].isna()]
print(missing_games[["game_date", "home_team_code", "away_team_code"]])

merged["actual_margin"] = merged["HOME_PTS"] - merged["AWAY_PTS"]
merged["ats_margin"] = merged["actual_margin"] + merged["home_spread"]
merged["home_covered"] = merged["ats_margin"] > 0
merged["push"] = merged["ats_margin"] == 0

final = merged[merged["home_spread"].notna()].copy()  # drop the 3 unmatched
final.to_parquet("data/processed/games_with_lines.parquet", index=False)

final = pd.read_parquet("data/processed/games_with_lines.parquet")
sample = final.sample(5, random_state=1)
print(sample[["game_date", "home_team_code", "away_team_code", "HOME_PTS", "AWAY_PTS", "home_spread", "actual_margin", "ats_margin", "home_covered", "push"]])

print(final["home_covered"].mean())   # should be roughly ~0.5, not wildly skewed
print(final["push"].sum())            # should be small but nonzero (pushes are rare but real)