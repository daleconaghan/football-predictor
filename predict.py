"""
Match Prediction Script
Use the trained model to predict upcoming matches.
"""

import pandas as pd
import pickle
from features import calculate_team_form, calculate_home_away_strength, calculate_head_to_head, calculate_rest_days
from datetime import datetime


def load_model(filepath: str = "model/predictor.pkl"):
    """Load the trained model."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_historical_data(filepath: str = "data/premier_league.csv"):
    """Load historical data for feature calculation."""
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df


def predict_match(model, df: pd.DataFrame, home_team: str, away_team: str) -> dict:
    """
    Predict the outcome of a match between two teams.
    Uses historical data up to today to calculate features.
    """
    today = pd.Timestamp.now()

    # Calculate features
    home_form = calculate_team_form(df, home_team, today)
    away_form = calculate_team_form(df, away_team, today)
    home_strength = calculate_home_away_strength(df, home_team, today)
    away_strength = calculate_home_away_strength(df, away_team, today)
    h2h = calculate_head_to_head(df, home_team, away_team, today)
    home_rest = calculate_rest_days(df, home_team, today)
    away_rest = calculate_rest_days(df, away_team, today)

    features = pd.DataFrame([{
        "home_form_points": home_form["form_points"],
        "home_form_gf": home_form["form_gf"],
        "home_form_ga": home_form["form_ga"],
        "home_home_win_rate": home_strength["home_win_rate"],
        "home_goals_avg": home_strength["home_goals_avg"],
        "away_form_points": away_form["form_points"],
        "away_form_gf": away_form["form_gf"],
        "away_form_ga": away_form["form_ga"],
        "away_away_win_rate": away_strength["away_win_rate"],
        "away_goals_avg": away_strength["away_goals_avg"],
        "h2h_home_wins": h2h["h2h_home_wins"],
        "h2h_away_wins": h2h["h2h_away_wins"],
        "h2h_draws": h2h["h2h_draws"],
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_diff": home_rest - away_rest,
        "form_diff": home_form["form_points"] - away_form["form_points"],
        "attack_diff": home_form["form_gf"] - away_form["form_gf"],
        "defense_diff": away_form["form_ga"] - home_form["form_ga"],
    }])

    # Get prediction and probabilities
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}

    return {
        "home_team": home_team,
        "away_team": away_team,
        "prediction": prediction,
        "probabilities": prob_dict,
        "confidence": max(probabilities),
        "home_form": home_form,
        "away_form": away_form,
    }


def display_prediction(result: dict):
    """Display prediction in a readable format."""
    print("\n" + "="*60)
    print(f"  {result['home_team']} vs {result['away_team']}")
    print("="*60)

    outcome_map = {"H": f"{result['home_team']} Win", "D": "Draw", "A": f"{result['away_team']} Win"}
    print(f"\n  PREDICTION: {outcome_map[result['prediction']]}")
    print(f"  Confidence: {result['confidence']:.1%}")

    print(f"\n  Probabilities:")
    print(f"    {result['home_team']} Win: {result['probabilities'].get('H', 0):.1%}")
    print(f"    Draw:          {result['probabilities'].get('D', 0):.1%}")
    print(f"    {result['away_team']} Win: {result['probabilities'].get('A', 0):.1%}")

    print(f"\n  Form (last 5 games):")
    print(f"    {result['home_team']}: {result['home_form']['form_points']} pts, {result['home_form']['form_gf']:.1f} goals/game")
    print(f"    {result['away_team']}: {result['away_form']['form_points']} pts, {result['away_form']['form_gf']:.1f} goals/game")

    print("="*60)


def get_all_teams(df: pd.DataFrame) -> list:
    """Get list of all teams in the dataset."""
    teams = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())
    return sorted(teams)


def interactive_mode():
    """Run interactive prediction mode."""
    print("Loading model and data...")
    model = load_model()
    df = load_historical_data()

    teams = get_all_teams(df)
    print(f"\nAvailable teams ({len(teams)}):")
    for i, team in enumerate(teams, 1):
        print(f"  {i:2}. {team}")

    while True:
        print("\n" + "-"*40)
        home = input("Enter home team (or 'quit' to exit): ").strip()
        if home.lower() == "quit":
            break

        away = input("Enter away team: ").strip()

        # Validate teams
        if home not in teams:
            print(f"Team '{home}' not found. Check spelling.")
            continue
        if away not in teams:
            print(f"Team '{away}' not found. Check spelling.")
            continue

        result = predict_match(model, df, home, away)
        display_prediction(result)


def predict_upcoming_fixtures():
    """Example: Predict some upcoming fixtures."""
    model = load_model()
    df = load_historical_data()

    # Example fixtures - update these with actual upcoming games
    fixtures = [
        ("Arsenal", "Liverpool"),
        ("Man City", "Chelsea"),
        ("Man United", "Tottenham"),
        ("Newcastle", "Everton"),
    ]

    print("\n" + "="*60)
    print("  UPCOMING FIXTURE PREDICTIONS")
    print("="*60)

    for home, away in fixtures:
        try:
            result = predict_match(model, df, home, away)
            display_prediction(result)
        except Exception as e:
            print(f"Could not predict {home} vs {away}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--fixtures":
        predict_upcoming_fixtures()
    else:
        interactive_mode()
