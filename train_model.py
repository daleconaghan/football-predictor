"""
Model Training for Football Match Prediction
Trains a Random Forest classifier to predict match outcomes.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os


# Features to use for prediction
FEATURE_COLUMNS = [
    "home_form_points",
    "home_form_gf",
    "home_form_ga",
    "home_home_win_rate",
    "home_goals_avg",
    "away_form_points",
    "away_form_gf",
    "away_form_ga",
    "away_away_win_rate",
    "away_goals_avg",
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "form_diff",
    "attack_diff",
    "defense_diff",
]


def load_and_prepare_data(filepath: str = "data/features.csv"):
    """Load features and prepare for training."""
    df = pd.read_csv(filepath)

    # Remove rows with missing values in features
    df = df.dropna(subset=FEATURE_COLUMNS + ["result"])

    # Skip early matches where teams don't have enough history
    df = df[df["home_form_points"] > 0]

    X = df[FEATURE_COLUMNS]
    y = df["result"]

    return X, y, df


def train_model(X, y):
    """Train the Random Forest model."""
    # Split data - use recent matches for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Don't shuffle to maintain time order
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, df):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)

    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.1%}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Compare to betting odds baseline
    print("\n" + "="*50)
    print("COMPARISON TO BETTING ODDS")
    print("="*50)

    test_indices = X_test.index
    test_df = df.loc[test_indices].copy()
    test_df["predicted"] = y_pred

    # Bookmaker's implied prediction (lowest odds = most likely)
    def get_bookmaker_pred(row):
        if pd.isna(row["odds_home"]):
            return None
        odds = {"H": row["odds_home"], "D": row["odds_draw"], "A": row["odds_away"]}
        return min(odds, key=odds.get)

    test_df["bookmaker_pred"] = test_df.apply(get_bookmaker_pred, axis=1)
    test_df = test_df.dropna(subset=["bookmaker_pred"])

    if len(test_df) > 0:
        bookmaker_accuracy = (test_df["bookmaker_pred"] == test_df["result"]).mean()
        model_accuracy = (test_df["predicted"] == test_df["result"]).mean()

        print(f"Bookmaker accuracy: {bookmaker_accuracy:.1%}")
        print(f"Our model accuracy: {model_accuracy:.1%}")
        print(f"Difference: {(model_accuracy - bookmaker_accuracy)*100:+.1f} percentage points")

    return accuracy


def show_feature_importance(model, feature_names):
    """Display which features matter most."""
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    print(importance.to_string(index=False))


def save_model(model, filepath: str = "model/predictor.pkl"):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {filepath}")


if __name__ == "__main__":
    # Load data
    print("Loading features...")
    X, y, df = load_and_prepare_data()

    print(f"\nTotal samples: {len(X)}")
    print(f"Result distribution:\n{y.value_counts()}")

    # Train
    print("\nTraining model...")
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    # Evaluate
    evaluate_model(model, X_test, y_test, df)

    # Feature importance
    show_feature_importance(model, FEATURE_COLUMNS)

    # Cross-validation for robustness check
    print("\n" + "="*50)
    print("CROSS-VALIDATION")
    print("="*50)
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")

    # Save model
    save_model(model)
