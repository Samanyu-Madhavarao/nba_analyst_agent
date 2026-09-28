TEAM_NAME_MAP = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BKN", "Charlotte": "CHA",
    "Chicago": "CHI", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
    "Detroit": "DET", "GoldenState": "GSW", "Houston": "HOU", "Indiana": "IND",
    "LAClippers": "LAC", "LALakers": "LAL", "Memphis": "MEM", "Miami": "MIA",
    "Milwaukee": "MIL", "Minnesota": "MIN", "NewOrleans": "NOP", "NewYork": "NYK",
    "OklahomaCity": "OKC", "Orlando": "ORL", "Philadelphia": "PHI", "Phoenix": "PHX",
    "Portland": "POR", "Sacramento": "SAC", "SanAntonio": "SAS", "Toronto": "TOR",
    "Utah": "UTA", "Washington": "WAS", "LA Clippers": "LAC", "Oklahoma City": "OKC",
    "Golden State": "GSW"
}

def normalize_team(name: str) -> str:
    name = name.strip()
    if name not in TEAM_NAME_MAP:
        raise ValueError(f"Unmapped team name: '{name}'")
    return TEAM_NAME_MAP[name]