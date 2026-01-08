"""
Feature Engineering for Football Match Prediction
Transforms raw match data into features the model can learn from.
"""

import pandas as pd
import numpy as np


def calculate_team_form(df: pd.DataFrame, team: str, date: str, n_matches: int = 5) -> dict:
    """
    Calculate a team's recent form (last n matches before the given date).
    Returns points, goals scored, goals conceded from recent games.
    """
    # Get matches involving this team before this date
    team_home = df[(df["HomeTeam"] == team) & (df["Date"] < date)]
    team_away = df[(df["AwayTeam"] == team) & (df["Date"] < date)]

    # Create unified view of team's matches
    home_matches = team_home[["Date", "FTHG", "FTAG", "FTR"]].copy()
    home_matches.columns = ["Date", "GF", "GA", "Result"]
    home_matches["Points"] = home_matches["Result"].map({"H": 3, "D": 1, "A": 0})

    away_matches = team_away[["Date", "FTAG", "FTHG", "FTR"]].copy()
    away_matches.columns = ["Date", "GF", "GA", "Result"]
    away_matches["Points"] = away_matches["Result"].map({"A": 3, "D": 1, "H": 0})

    all_matches = pd.concat([home_matches, away_matches]).sort_values("Date", ascending=False)
    recent = all_matches.head(n_matches)

    if len(recent) == 0:
        return {"form_points": 0, "form_gf": 0, "form_ga": 0, "matches_played": 0}

    return {
        "form_points": recent["Points"].sum(),
        "form_gf": recent["GF"].mean(),
        "form_ga": recent["GA"].mean(),
        "matches_played": len(recent),
    }


def calculate_home_away_strength(df: pd.DataFrame, team: str, date: str) -> dict:
    """Calculate team's home and away performance."""
    home_matches = df[(df["HomeTeam"] == team) & (df["Date"] < date)]
    away_matches = df[(df["AwayTeam"] == team) & (df["Date"] < date)]

    # Home strength
    if len(home_matches) > 0:
        home_wins = (home_matches["FTR"] == "H").sum()
        home_win_rate = home_wins / len(home_matches)
        home_goals_avg = home_matches["FTHG"].mean()
    else:
        home_win_rate = 0.33
        home_goals_avg = 1.5

    # Away strength
    if len(away_matches) > 0:
        away_wins = (away_matches["FTR"] == "A").sum()
        away_win_rate = away_wins / len(away_matches)
        away_goals_avg = away_matches["FTAG"].mean()
    else:
        away_win_rate = 0.33
        away_goals_avg = 1.0

    return {
        "home_win_rate": home_win_rate,
        "home_goals_avg": home_goals_avg,
        "away_win_rate": away_win_rate,
        "away_goals_avg": away_goals_avg,
    }


def calculate_rest_days(df: pd.DataFrame, team: str, date) -> int:
    """
    Calculate days since the team's last match.
    More rest = fresher players, potential advantage.
    """
    # Get all matches involving this team before this date
    team_matches = df[
        ((df["HomeTeam"] == team) | (df["AwayTeam"] == team)) &
        (df["Date"] < date)
    ].sort_values("Date", ascending=False)

    if len(team_matches) == 0:
        return 7  # Default to a week if no prior matches

    last_match_date = team_matches.iloc[0]["Date"]
    days_rest = (date - last_match_date).days
    return min(days_rest, 14)  # Cap at 14 to avoid outliers (international breaks)


def calculate_head_to_head(df: pd.DataFrame, home_team: str, away_team: str, date: str) -> dict:
    """Calculate head-to-head record between two teams."""
    h2h = df[
        ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team) |
         (df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team)) &
        (df["Date"] < date)
    ]

    if len(h2h) == 0:
        return {"h2h_home_wins": 0, "h2h_away_wins": 0, "h2h_draws": 0}

    home_team_wins = 0
    away_team_wins = 0
    draws = 0

    for _, match in h2h.iterrows():
        if match["FTR"] == "D":
            draws += 1
        elif match["HomeTeam"] == home_team:
            if match["FTR"] == "H":
                home_team_wins += 1
            else:
                away_team_wins += 1
        else:  # home_team was playing away
            if match["FTR"] == "A":
                home_team_wins += 1
            else:
                away_team_wins += 1

    total = len(h2h)
    return {
        "h2h_home_wins": home_team_wins / total,
        "h2h_away_wins": away_team_wins / total,
        "h2h_draws": draws / total,
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all features for each match.
    This is the main function that transforms raw data into model-ready features.
    """
    print("Engineering features... (this may take a minute)")

    # Ensure date is properly formatted and sorted
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)

    features_list = []

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing match {idx}/{len(df)}")

        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        date = row["Date"]

        # Get form for both teams
        home_form = calculate_team_form(df, home_team, date)
        away_form = calculate_team_form(df, away_team, date)

        # Get home/away strength
        home_strength = calculate_home_away_strength(df, home_team, date)
        away_strength = calculate_home_away_strength(df, away_team, date)

        # Get head to head
        h2h = calculate_head_to_head(df, home_team, away_team, date)

        # Get rest days
        home_rest = calculate_rest_days(df, home_team, date)
        away_rest = calculate_rest_days(df, away_team, date)

        features = {
            "match_id": idx,
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
            # Home team features
            "home_form_points": home_form["form_points"],
            "home_form_gf": home_form["form_gf"],
            "home_form_ga": home_form["form_ga"],
            "home_home_win_rate": home_strength["home_win_rate"],
            "home_goals_avg": home_strength["home_goals_avg"],
            # Away team features
            "away_form_points": away_form["form_points"],
            "away_form_gf": away_form["form_gf"],
            "away_form_ga": away_form["form_ga"],
            "away_away_win_rate": away_strength["away_win_rate"],
            "away_goals_avg": away_strength["away_goals_avg"],
            # Head to head
            "h2h_home_wins": h2h["h2h_home_wins"],
            "h2h_away_wins": h2h["h2h_away_wins"],
            "h2h_draws": h2h["h2h_draws"],
            # Rest days
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_diff": home_rest - away_rest,
            # Derived features
            "form_diff": home_form["form_points"] - away_form["form_points"],
            "attack_diff": home_form["form_gf"] - away_form["form_gf"],
            "defense_diff": away_form["form_ga"] - home_form["form_ga"],
            # Target variable
            "result": row["FTR"],
            # Betting odds (for comparison)
            "odds_home": row.get("B365H", np.nan),
            "odds_draw": row.get("B365D", np.nan),
            "odds_away": row.get("B365A", np.nan),
        }

        features_list.append(features)

    features_df = pd.DataFrame(features_list)

    # Save features
    features_df.to_csv("data/features.csv", index=False)
    print(f"\nSaved {len(features_df)} matches with features to data/features.csv")

    return features_df


if __name__ == "__main__":
    # Load raw data and engineer features
    df = pd.read_csv("data/premier_league.csv")
    features_df = engineer_features(df)

    print("\nFeature columns:")
    print(features_df.columns.tolist())
    print("\nSample features:")
    print(features_df.head())
