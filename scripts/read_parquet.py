import pandas as pd
odds = pd.read_parquet("data/raw/betting_lines/nba_2022-2023_results_odds.parquet")  # adjust filename
print(odds.columns.tolist())
print(odds.head(3).T)
names = sorted(set(odds["home_team"]) | set(odds["away_team"]))
print(len(names), names)
features = pd.read_parquet("data/processed/features.parquet")
abbrevs = sorted(set(features["home_TEAM_ABBREVIATION"]) | set(features["away_TEAM_ABBREVIATION"]))
print(len(abbrevs), abbrevs)