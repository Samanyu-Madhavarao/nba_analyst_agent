import pandas as pd
import requests
import os
import re
import io
from playwright.sync_api import sync_playwright

RAW_DIR = "data/raw/lines"
os.makedirs(RAW_DIR, exist_ok=True)

# Only seasons actually available on the site (verify current list yourself —
# archive currently runs 2007-08 through 2022-23)
SEASON_URLS = {
    "2022-23": "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-2022-23",
    "2021-22": "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-2021-22",
    "2020-21": "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-2020-21",
    "2019-20": "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-2019-20",
    "2018-19": "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-2018-19",
    "2017-18": "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-2017-18",
    "2016-17": "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-2016-17",
    "2015-16": "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-2015-16"
}

COLUMN_NAMES = ["Date", "Rot", "VH", "Team", "1st", "2nd", "3rd", "4th",
                "Final", "Open", "Close", "ML", "2H"]

def fetch_full_season_html(url: str, debug_dump_path: str = None) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",  # hide the most obvious headless tell
            ],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 900},
        )
        # Remove the navigator.webdriver flag that gives headless browsers away
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        page = context.new_page()

        response = page.goto(url, timeout=60000, wait_until="networkidle")
        print(f"  HTTP status: {response.status if response else 'no response'}")

        # Scroll repeatedly until row count stops growing
        prev_row_count = 0
        for _ in range(50):
            page.mouse.wheel(0, 5000)
            page.wait_for_timeout(500)
            row_count = page.locator("tr").count()
            if row_count == prev_row_count:
                break
            prev_row_count = row_count

        html = page.content()

        # --- diagnostics ---
        print(f"  HTML length: {len(html)} chars")
        print(f"  'Rot' present: {'Rot' in html}")
        print(f"  page title: {page.title()!r}")
        if any(marker in html for marker in ("Just a moment", "Attention Required", "captcha", "cf-browser-verification")):
            print("  WARNING: response looks like a bot-check / interstitial page, not real content")

        if debug_dump_path:
            with open(debug_dump_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"  Dumped raw HTML to {debug_dump_path} for inspection")

        browser.close()
        return html

def html_to_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html), match="Rot", header=None)  # "Rot" column is unique to the odds table
    odds_table = tables[0]

    odds_table = odds_table.iloc[1:].reset_index(drop=True)
    odds_table.columns = COLUMN_NAMES

    if len(odds_table) < 2000:  # full season ≈ 2460 rows; playoffs/lockout years vary, but this catches partial loads
        print(f"  WARNING: only {len(odds_table)} rows — season may not have fully loaded")

    return odds_table

def fetch_season_table(url: str) -> pd.DataFrame:
    html = fetch_full_season_html(url)
    return html_to_table(html)

def parse_date(raw_date: str, season: str) -> str:
    """raw_date is like '101', '1018', '1231' — MDD or MMDD, no leading zero."""
    raw_date = str(int(raw_date)).zfill(4)  # normalize '101' -> '0101'
    month = int(raw_date[:2])
    day = int(raw_date[2:])
    start_year, end_year = season.split("-")
    year = start_year if month >= 8 else "20" + end_year
    return f"{year}-{month:02d}-{day:02d}"

def parse_spread_value(raw) -> float:
    """Handle 'pk' (pick 'em = spread of 0) and other non-numeric odds values."""
    if isinstance(raw, str):
        raw = raw.strip().lower()
        if raw in ("pk", "pick", "pick'em", "even"):
            return 0.0
        if raw in ("", "nl", "n/a", "-"):
            return None  # no line available — handle downstream (drop row, or flag as missing)
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None

def parse_season_table(df: pd.DataFrame, season: str) -> pd.DataFrame:
    games = []
    for i in range(0, len(df) - 1, 2):
        row_a = df.iloc[i]      # first row of the pair
        row_b = df.iloc[i + 1]  # second row of the pair

        # Use the VH column directly rather than assuming order
        away = row_a if row_a["VH"] == "V" else row_b
        home = row_a if row_a["VH"] == "H" else row_b

        # Determine favorite by moneyline sign (negative ML = favorite)
        home_ml = parse_spread_value(home["ML"])
        away_ml = parse_spread_value(away["ML"])
        if home_ml is None or away_ml is None:
            continue  # or however you want to handle missing moneylines
        home_is_favorite = home_ml < away_ml

        favorite_row = home if home_is_favorite else away
        underdog_row = away if home_is_favorite else home

        close_spread = favorite_row["Close"]   # spread lives on the favorite's row
        close_total = underdog_row["Close"]    # total lives on the underdog's row

        # Standardize to home_spread convention (negative = home favored)
        home_close_spread = -abs(parse_spread_value(close_spread)) if home_is_favorite else abs(parse_spread_value(close_spread))

        games.append({
            "season": season,
            "date": parse_date(home["Date"], season),
            "away_team_raw": away["Team"],
            "home_team_raw": home["Team"],
            "away_score": away["Final"],
            "home_score": home["Final"],
            "home_close_spread": home_close_spread,
            "close_total": close_total,
            "home_ml": home["ML"],
            "away_ml": away["ML"],
        })

    return pd.DataFrame(games)


for season, url in SEASON_URLS.items():
    out_path = f"{RAW_DIR}/{season}.parquet"
    if os.path.exists(out_path):
        print(f"Skipping {season} (already downloaded)")
        continue

    print(f"Fetching {season}")
    raw_html = fetch_full_season_html(url, debug_dump_path=f"debug_{season}.html")
    raw_table = html_to_table(raw_html)

    parsed = parse_season_table(raw_table, season)
    parsed.to_parquet(out_path)
    print(f"  Saved {len(parsed)} games")