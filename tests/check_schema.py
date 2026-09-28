"""
check_schema.py

Quick inspector: prints row counts, columns, and a sample row for each
data file in your project. Run this from your project root
(C:\\Projects\\nba_analyst_agent) with the venv active:

    python3 tests/check_schema.py

Edit the `files` list below if your filenames differ from what's shown
in your folder tree.
"""

import pandas as pd
from pathlib import Path

files = [
    "data/raw/games.parquet",
    "data/raw/boxscores.parquet",
    "data/raw/boxscores_advanced.parquet",
    "data/processed/games_one_row.parquet",
    "data/processed/games_with_lines.parquet",
    "data/raw/lines.parquet",
    "data/raw/lines_v2.parquet",
]

for f in files:
    p = Path(f)
    try:
        if p.is_dir():
            sub = list(p.glob("*.parquet"))
            if not sub:
                print(f"{f}: directory, no .parquet files found directly inside "
                      f"(check for a nested folder structure)")
                continue
            df = pd.read_parquet(sub[0])
            print(f"{f} (sample file: {sub[0].name}): {len(df):,} rows")
        else:
            df = pd.read_parquet(p)
            print(f"{f}: {len(df):,} rows")
        print(f"   columns: {list(df.columns)}")
        print(f"   sample row:\n{df.head(1).to_string()}\n")
    except FileNotFoundError:
        print(f"{f}: NOT FOUND\n")
    except Exception as e:
        print(f"{f}: ERROR - {e}\n")