"""
Football Match Data Collector
Downloads historical Premier League data from Football-Data.co.uk
"""

import pandas as pd
import os

# Seasons to download (format: YYMM where YY is start year)
SEASONS = [
    "2122",  # 2021-22
    "2223",  # 2022-23
    "2324",  # 2023-24
    "2425",  # 2024-25
    "2526",  # 2025-26 (current)
]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
DATA_DIR = "data"

# 2025-26 Premier League teams
PREMIER_LEAGUE_TEAMS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Burnley",        # Promoted 2025-26
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Leeds",          # Promoted 2025-26
    "Liverpool",
    "Man City",
    "Man United",
    "Newcastle",
    "Nott'm Forest",
    "Sunderland",     # Promoted 2025-26
    "Tottenham",
    "West Ham",
    "Wolves",
]


def download_season(season: str) -> pd.DataFrame:
    """Download data for a single season."""
    url = BASE_URL.format(season=season)
    print(f"Downloading {season}...")

    try:
        df = pd.read_csv(url)
        df["Season"] = season
        return df
    except Exception as e:
        print(f"Error downloading {season}: {e}")
        return pd.DataFrame()


def collect_all_data() -> pd.DataFrame:
    """Download and combine all seasons."""
    os.makedirs(DATA_DIR, exist_ok=True)

    all_data = []
    for season in SEASONS:
        df = download_season(season)
        if not df.empty:
            all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    # Keep only the columns we need
    columns_to_keep = [
        "Season", "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",  # Full time: Home Goals, Away Goals, Result
        "HTHG", "HTAG", "HTR",  # Half time stats
        "HS", "AS",             # Shots
        "HST", "AST",           # Shots on target
        "HF", "AF",             # Fouls
        "HC", "AC",             # Corners
        "HY", "AY",             # Yellow cards
        "HR", "AR",             # Red cards
        "B365H", "B365D", "B365A",  # Bet365 odds
    ]

    # Only keep columns that exist in the data
    columns_to_keep = [c for c in columns_to_keep if c in combined.columns]
    combined = combined[columns_to_keep]

    # Filter to current Premier League teams
    combined = combined[
        (combined["HomeTeam"].isin(PREMIER_LEAGUE_TEAMS)) &
        (combined["AwayTeam"].isin(PREMIER_LEAGUE_TEAMS))
    ]
    print(f"Filtered to {len(combined)} matches between Premier League teams")

    # Save to CSV
    output_path = os.path.join(DATA_DIR, "premier_league.csv")
    combined.to_csv(output_path, index=False)
    print(f"\nSaved {len(combined)} matches to {output_path}")

    return combined


def show_data_summary(df: pd.DataFrame):
    """Display summary of the collected data."""
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    print(f"Total matches: {len(df)}")
    print(f"Seasons: {df['Season'].unique()}")
    print(f"Teams: {df['HomeTeam'].nunique()}")
    print(f"\nResult distribution:")
    print(df["FTR"].value_counts())
    print(f"\nSample of recent matches:")
    print(df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].tail(10))


if __name__ == "__main__":
    df = collect_all_data()
    show_data_summary(df)
