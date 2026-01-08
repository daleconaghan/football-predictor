"""
Prediction Tracker
Log predictions and track accuracy over time.
"""

import pandas as pd
import os
from datetime import datetime
from predict import load_model, load_historical_data, predict_match

TRACKER_FILE = "data/predictions.csv"


def load_tracker() -> pd.DataFrame:
    """Load existing predictions or create empty tracker."""
    if os.path.exists(TRACKER_FILE):
        return pd.read_csv(TRACKER_FILE)
    return pd.DataFrame(columns=[
        "date_predicted", "match_date", "home_team", "away_team",
        "predicted_result", "confidence",
        "prob_home", "prob_draw", "prob_away",
        "actual_result", "correct"
    ])


def save_tracker(df: pd.DataFrame):
    """Save tracker to CSV."""
    os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True)
    df.to_csv(TRACKER_FILE, index=False)
    print(f"Saved {len(df)} predictions to {TRACKER_FILE}")


def log_prediction(home_team: str, away_team: str, match_date: str = None):
    """Log a new prediction."""
    model = load_model()
    df = load_historical_data()

    result = predict_match(model, df, home_team, away_team)
    tracker = load_tracker()

    new_row = {
        "date_predicted": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "match_date": match_date or "TBD",
        "home_team": home_team,
        "away_team": away_team,
        "predicted_result": result["prediction"],
        "confidence": round(result["confidence"], 3),
        "prob_home": round(result["probabilities"].get("H", 0), 3),
        "prob_draw": round(result["probabilities"].get("D", 0), 3),
        "prob_away": round(result["probabilities"].get("A", 0), 3),
        "actual_result": None,
        "correct": None,
    }

    tracker = pd.concat([tracker, pd.DataFrame([new_row])], ignore_index=True)
    save_tracker(tracker)

    print(f"\nLogged prediction: {home_team} vs {away_team}")
    print(f"Prediction: {result['prediction']} ({result['confidence']:.1%})")
    return result


def update_result(home_team: str, away_team: str, actual_result: str):
    """Update a prediction with the actual result."""
    tracker = load_tracker()

    # Find the most recent prediction for this fixture without a result
    mask = (
        (tracker["home_team"] == home_team) &
        (tracker["away_team"] == away_team) &
        (tracker["actual_result"].isna())
    )

    if not mask.any():
        print(f"No pending prediction found for {home_team} vs {away_team}")
        return

    idx = tracker[mask].index[-1]  # Get most recent
    tracker.loc[idx, "actual_result"] = actual_result
    tracker.loc[idx, "correct"] = tracker.loc[idx, "predicted_result"] == actual_result

    save_tracker(tracker)

    was_correct = tracker.loc[idx, "correct"]
    predicted = tracker.loc[idx, "predicted_result"]
    print(f"\nUpdated: {home_team} vs {away_team}")
    print(f"Predicted: {predicted}, Actual: {actual_result}")
    print(f"Result: {'✓ Correct' if was_correct else '✗ Wrong'}")


def show_stats():
    """Show prediction accuracy statistics."""
    tracker = load_tracker()

    if len(tracker) == 0:
        print("No predictions logged yet.")
        return

    completed = tracker[tracker["actual_result"].notna()]
    pending = tracker[tracker["actual_result"].isna()]

    print("\n" + "="*50)
    print("PREDICTION TRACKER STATS")
    print("="*50)

    print(f"\nTotal predictions: {len(tracker)}")
    print(f"Completed: {len(completed)}")
    print(f"Pending: {len(pending)}")

    if len(completed) > 0:
        accuracy = completed["correct"].mean()
        print(f"\nAccuracy: {accuracy:.1%} ({completed['correct'].sum()}/{len(completed)})")

        # Breakdown by predicted result
        print("\nBy prediction type:")
        for result in ["H", "D", "A"]:
            subset = completed[completed["predicted_result"] == result]
            if len(subset) > 0:
                acc = subset["correct"].mean()
                label = {"H": "Home wins", "D": "Draws", "A": "Away wins"}[result]
                print(f"  {label}: {acc:.1%} ({subset['correct'].sum()}/{len(subset)})")

        # Recent predictions
        print("\nRecent completed predictions:")
        recent = completed.tail(10)[["home_team", "away_team", "predicted_result", "actual_result", "correct"]]
        for _, row in recent.iterrows():
            symbol = "✓" if row["correct"] else "✗"
            print(f"  {symbol} {row['home_team']} vs {row['away_team']}: predicted {row['predicted_result']}, actual {row['actual_result']}")

    if len(pending) > 0:
        print("\nPending predictions:")
        for _, row in pending.iterrows():
            print(f"  {row['home_team']} vs {row['away_team']}: {row['predicted_result']} ({row['confidence']:.1%})")


def show_pending():
    """Show predictions awaiting results."""
    tracker = load_tracker()
    pending = tracker[tracker["actual_result"].isna()]

    if len(pending) == 0:
        print("No pending predictions.")
        return

    print("\n" + "="*50)
    print("PENDING PREDICTIONS")
    print("="*50)

    for _, row in pending.iterrows():
        print(f"\n{row['home_team']} vs {row['away_team']}")
        print(f"  Prediction: {row['predicted_result']} ({row['confidence']:.1%})")
        print(f"  Probabilities: H={row['prob_home']:.1%} D={row['prob_draw']:.1%} A={row['prob_away']:.1%}")
        print(f"  Match date: {row['match_date']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python tracker.py log <home_team> <away_team> [match_date]")
        print("  python tracker.py result <home_team> <away_team> <H/D/A>")
        print("  python tracker.py stats")
        print("  python tracker.py pending")
        print("\nExamples:")
        print("  python tracker.py log Arsenal Liverpool 2025-12-28")
        print("  python tracker.py result Arsenal Liverpool H")
        sys.exit(0)

    command = sys.argv[1]

    if command == "log" and len(sys.argv) >= 4:
        home = sys.argv[2]
        away = sys.argv[3]
        match_date = sys.argv[4] if len(sys.argv) > 4 else None
        log_prediction(home, away, match_date)

    elif command == "result" and len(sys.argv) == 5:
        home = sys.argv[2]
        away = sys.argv[3]
        result = sys.argv[4].upper()
        if result not in ["H", "D", "A"]:
            print("Result must be H (home win), D (draw), or A (away win)")
        else:
            update_result(home, away, result)

    elif command == "stats":
        show_stats()

    elif command == "pending":
        show_pending()

    else:
        print("Invalid command. Run without arguments for usage.")
