"""
Backtest Model Performance
Simulates predicting each gameweek of the 2025-26 season using only data available at that time.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from features import (
    calculate_team_form,
    calculate_home_away_strength,
    calculate_head_to_head,
    calculate_rest_days,
)
from train_model import FEATURE_COLUMNS


def load_all_data():
    """Load all historical data."""
    df = pd.read_csv("data/premier_league.csv")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def get_features_for_match(df, home_team, away_team, match_date):
    """Calculate features for a single match using only historical data."""
    home_form = calculate_team_form(df, home_team, match_date)
    away_form = calculate_team_form(df, away_team, match_date)
    home_strength = calculate_home_away_strength(df, home_team, match_date)
    away_strength = calculate_home_away_strength(df, away_team, match_date)
    h2h = calculate_head_to_head(df, home_team, away_team, match_date)
    home_rest = calculate_rest_days(df, home_team, match_date)
    away_rest = calculate_rest_days(df, away_team, match_date)

    return {
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
    }


def train_model_on_historical(df_train):
    """Train model on historical data only."""
    # Engineer features for training data
    features_list = []

    for idx, row in df_train.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        date = row["Date"]

        features = get_features_for_match(df_train, home_team, away_team, date)
        features["result"] = row["FTR"]
        features_list.append(features)

    features_df = pd.DataFrame(features_list)

    # Remove rows with no form data
    features_df = features_df[features_df["home_form_points"] > 0]

    if len(features_df) < 50:
        return None  # Not enough training data

    X = features_df[FEATURE_COLUMNS]
    y = features_df["result"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def run_backtest():
    """Run backtest on 2025-26 season."""
    print("Loading data...")
    df = load_all_data()

    # Split into training (pre-2025-26) and test (2025-26 season)
    season_start = pd.Timestamp("2025-08-01")
    df_historical = df[df["Date"] < season_start].copy()
    df_2526 = df[df["Date"] >= season_start].copy()

    print(f"Historical matches (training): {len(df_historical)}")
    print(f"2025-26 matches (testing): {len(df_2526)}")

    if len(df_2526) == 0:
        print("No 2025-26 season data found.")
        return

    # Assign gameweeks based on match order (10 matches per gameweek)
    # This is more accurate than date-based detection during congested periods
    df_2526 = df_2526.sort_values("Date").reset_index(drop=True)
    df_2526["Gameweek"] = (df_2526.index // 10) + 1

    max_gw = df_2526["Gameweek"].max()
    print(f"Detected {max_gw} gameweeks in the data")

    results = []
    all_predictions = []

    print("\n" + "="*70)
    print("BACKTEST: 2025-26 SEASON")
    print("="*70)

    for gw in sorted(df_2526["Gameweek"].unique()):
        gw_matches = df_2526[df_2526["Gameweek"] == gw]

        if len(gw_matches) == 0:
            continue

        # Get the earliest date in this gameweek
        gw_date = gw_matches["Date"].min()

        # Train model on all data before this gameweek
        training_data = pd.concat([
            df_historical,
            df_2526[df_2526["Date"] < gw_date]
        ])

        model = train_model_on_historical(training_data)

        if model is None:
            print(f"Gameweek {gw}: Insufficient training data")
            continue

        gw_correct = 0
        gw_total = 0

        print(f"\n--- GAMEWEEK {gw} ---")

        for _, match in gw_matches.iterrows():
            home = match["HomeTeam"]
            away = match["AwayTeam"]
            actual = match["FTR"]

            # Get features using only data available before the match
            features = get_features_for_match(training_data, home, away, match["Date"])
            X_pred = pd.DataFrame([features])[FEATURE_COLUMNS]

            # Predict
            pred = model.predict(X_pred)[0]
            proba = model.predict_proba(X_pred)[0]
            confidence = max(proba)

            correct = pred == actual
            gw_correct += int(correct)
            gw_total += 1

            symbol = "✓" if correct else "✗"
            print(f"  {symbol} {home} vs {away}: predicted {pred} ({confidence:.0%}), actual {actual}")

            all_predictions.append({
                "gameweek": gw,
                "date": match["Date"],
                "home": home,
                "away": away,
                "predicted": pred,
                "confidence": confidence,
                "actual": actual,
                "correct": correct,
            })

        gw_accuracy = gw_correct / gw_total if gw_total > 0 else 0
        results.append({
            "gameweek": gw,
            "correct": gw_correct,
            "total": gw_total,
            "accuracy": gw_accuracy,
        })
        print(f"  Gameweek {gw} accuracy: {gw_correct}/{gw_total} ({gw_accuracy:.0%})")

    # Summary
    print("\n" + "="*70)
    print("BACKTEST SUMMARY")
    print("="*70)

    results_df = pd.DataFrame(results)
    predictions_df = pd.DataFrame(all_predictions)

    total_correct = results_df["correct"].sum()
    total_matches = results_df["total"].sum()
    overall_accuracy = total_correct / total_matches if total_matches > 0 else 0

    print(f"\nOverall: {total_correct}/{total_matches} correct ({overall_accuracy:.1%})")

    print("\nBy gameweek:")
    for _, row in results_df.iterrows():
        bar = "█" * int(row["accuracy"] * 20)
        print(f"  GW{row['gameweek']:2}: {bar} {row['accuracy']:.0%} ({row['correct']}/{row['total']})")

    # Breakdown by prediction type
    print("\nBy prediction type:")
    for pred_type in ["H", "D", "A"]:
        subset = predictions_df[predictions_df["predicted"] == pred_type]
        if len(subset) > 0:
            acc = subset["correct"].mean()
            label = {"H": "Home wins", "D": "Draws", "A": "Away wins"}[pred_type]
            print(f"  {label}: {acc:.1%} ({subset['correct'].sum()}/{len(subset)})")

    # Compare to baseline (always predict home win)
    home_baseline = (predictions_df["actual"] == "H").mean()
    print(f"\nBaseline (always predict home): {home_baseline:.1%}")
    print(f"Model improvement over baseline: {(overall_accuracy - home_baseline)*100:+.1f} percentage points")

    # Save detailed results
    predictions_df.to_csv("data/backtest_results.csv", index=False)
    print(f"\nDetailed results saved to data/backtest_results.csv")

    return predictions_df


if __name__ == "__main__":
    run_backtest()
