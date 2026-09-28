import pandas as pd

TEAM_CODE_MAP = {
    "atl": "ATL", "bkn": "BKN", "bos": "BOS", "cha": "CHA", "chi": "CHI",
    "cle": "CLE", "dal": "DAL", "den": "DEN", "det": "DET", "gs": "GSW",
    "hou": "HOU", "ind": "IND", "lac": "LAC", "lal": "LAL", "mem": "MEM",
    "mia": "MIA", "mil": "MIL", "min": "MIN", "no": "NOP", "ny": "NYK",
    "okc": "OKC", "orl": "ORL", "phi": "PHI", "phx": "PHX", "por": "POR",
    "sa": "SAS", "sac": "SAC", "tor": "TOR", "utah": "UTA", "wsh": "WAS",
}

df = pd.read_csv("data/raw/nba_2008-2026.csv")

df["home_spread"] = df.apply(
    lambda r: -abs(r["spread"]) if r["whos_favored"] == "home" else abs(r["spread"]),
    axis=1
)

df["game_date"] = pd.to_datetime(df["date"]).dt.date
df["home_team_code"] = df["home"].map(TEAM_CODE_MAP)
df["away_team_code"] = df["away"].map(TEAM_CODE_MAP)

assert df["home_team_code"].isna().sum() == 0, "unmapped home codes found"
assert df["away_team_code"].isna().sum() == 0, "unmapped away codes found"

df.to_parquet("data/raw/lines_v2.parquet", index=False)
print("saved", df.shape)
print(df.columns.tolist())