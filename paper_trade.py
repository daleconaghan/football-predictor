"""
Paper Trading Tracker
Track hypothetical bets to measure model edge over time.
"""

import pandas as pd
import os
from datetime import datetime
from predict import load_model, load_historical_data, predict_match

TRADES_FILE = "data/paper_trades.csv"
STARTING_BANKROLL = 1000  # Virtual euros


def load_trades() -> pd.DataFrame:
    """Load existing trades or create empty tracker."""
    if os.path.exists(TRADES_FILE):
        return pd.read_csv(TRADES_FILE)
    return pd.DataFrame(columns=[
        "date", "match_date", "home_team", "away_team",
        "prediction", "confidence", "odds", "implied_prob",
        "stake", "value", "actual_result", "profit", "bankroll"
    ])


def save_trades(df: pd.DataFrame):
    """Save trades to CSV."""
    os.makedirs(os.path.dirname(TRADES_FILE), exist_ok=True)
    df.to_csv(TRADES_FILE, index=False)


def get_current_bankroll(df: pd.DataFrame) -> float:
    """Get current bankroll from last completed trade."""
    completed = df[df["actual_result"].notna()]
    if len(completed) == 0:
        return STARTING_BANKROLL
    return completed.iloc[-1]["bankroll"]


def calculate_value(model_prob: float, odds: float) -> float:
    """Calculate expected value of a bet."""
    return (model_prob * odds) - 1


def calculate_stake(bankroll: float, value: float, odds: float, max_stake_pct: float = 0.05) -> float:
    """
    Calculate stake using fractional Kelly criterion.
    Uses 25% Kelly for safety.
    """
    if value <= 0:
        return 0

    edge = value
    kelly_fraction = edge / (odds - 1) if odds > 1 else 0
    stake_pct = kelly_fraction * 0.25  # Quarter Kelly
    stake_pct = min(stake_pct, max_stake_pct)  # Cap at 5% of bankroll

    return round(bankroll * stake_pct, 2)


def place_trade(home_team: str, away_team: str, odds: float, match_date: str = None):
    """Place a paper trade."""
    model = load_model()
    df = load_historical_data()
    trades = load_trades()

    result = predict_match(model, df, home_team, away_team)
    prediction = result["prediction"]
    confidence = result["confidence"]

    # Get probability for our predicted outcome
    model_prob = result["probabilities"].get(prediction, 0)
    implied_prob = 1 / odds
    value = calculate_value(model_prob, odds)

    bankroll = get_current_bankroll(trades)
    stake = calculate_stake(bankroll, value, odds)

    if stake == 0:
        print(f"\nNO VALUE BET")
        print(f"Model: {model_prob:.1%} vs Implied: {implied_prob:.1%}")
        print(f"Value: {value:.1%} (need > 0% for value)")
        print("Skipping this bet.")
        return

    new_trade = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "match_date": match_date or "TBD",
        "home_team": home_team,
        "away_team": away_team,
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "odds": odds,
        "implied_prob": round(implied_prob, 3),
        "stake": stake,
        "value": round(value, 3),
        "actual_result": None,
        "profit": None,
        "bankroll": None,
    }

    trades = pd.concat([trades, pd.DataFrame([new_trade])], ignore_index=True)
    save_trades(trades)

    outcome_map = {"H": f"{home_team} Win", "D": "Draw", "A": f"{away_team} Win"}

    print(f"\n{'='*50}")
    print(f"PAPER TRADE PLACED")
    print(f"{'='*50}")
    print(f"Match: {home_team} vs {away_team}")
    print(f"Prediction: {outcome_map[prediction]}")
    print(f"Model probability: {model_prob:.1%}")
    print(f"Bookmaker implied: {implied_prob:.1%}")
    print(f"Value: {value:.1%}")
    print(f"Odds: {odds}")
    print(f"Stake: €{stake:.2f} ({stake/bankroll*100:.1f}% of bankroll)")
    print(f"Current bankroll: €{bankroll:.2f}")
    print(f"{'='*50}")


def settle_trade(home_team: str, away_team: str, actual_result: str):
    """Settle a paper trade with the actual result."""
    trades = load_trades()

    # Find the most recent unsettled trade for this fixture
    mask = (
        (trades["home_team"] == home_team) &
        (trades["away_team"] == away_team) &
        (trades["actual_result"].isna())
    )

    if not mask.any():
        print(f"No pending trade found for {home_team} vs {away_team}")
        return

    idx = trades[mask].index[-1]
    trade = trades.loc[idx]

    prediction = trade["prediction"]
    odds = trade["odds"]
    stake = trade["stake"]

    # Get previous bankroll
    prev_completed = trades[(trades["actual_result"].notna()) & (trades.index < idx)]
    if len(prev_completed) == 0:
        prev_bankroll = STARTING_BANKROLL
    else:
        prev_bankroll = prev_completed.iloc[-1]["bankroll"]

    # Calculate profit
    if prediction == actual_result:
        profit = stake * (odds - 1)
        new_bankroll = prev_bankroll + profit
        result_str = "WIN"
    else:
        profit = -stake
        new_bankroll = prev_bankroll + profit
        result_str = "LOSS"

    trades.loc[idx, "actual_result"] = actual_result
    trades.loc[idx, "profit"] = round(profit, 2)
    trades.loc[idx, "bankroll"] = round(new_bankroll, 2)

    save_trades(trades)

    print(f"\n{'='*50}")
    print(f"TRADE SETTLED: {result_str}")
    print(f"{'='*50}")
    print(f"Match: {home_team} vs {away_team}")
    print(f"Predicted: {prediction}, Actual: {actual_result}")
    print(f"Stake: €{stake:.2f} @ {odds}")
    print(f"Profit: €{profit:+.2f}")
    print(f"New bankroll: €{new_bankroll:.2f}")
    print(f"{'='*50}")


def show_stats():
    """Show paper trading statistics."""
    trades = load_trades()

    if len(trades) == 0:
        print("No trades yet.")
        return

    completed = trades[trades["actual_result"].notna()]
    pending = trades[trades["actual_result"].isna()]

    print(f"\n{'='*60}")
    print(f"PAPER TRADING STATS")
    print(f"{'='*60}")

    print(f"\nStarting bankroll: €{STARTING_BANKROLL:.2f}")

    if len(completed) > 0:
        current_bankroll = completed.iloc[-1]["bankroll"]
        total_profit = current_bankroll - STARTING_BANKROLL
        roi = (total_profit / STARTING_BANKROLL) * 100

        wins = (completed["profit"] > 0).sum()
        losses = (completed["profit"] < 0).sum()
        win_rate = wins / len(completed) * 100

        avg_odds = completed["odds"].mean()
        avg_value = completed["value"].mean() * 100

        print(f"Current bankroll: €{current_bankroll:.2f}")
        print(f"Total profit: €{total_profit:+.2f}")
        print(f"ROI: {roi:+.1f}%")
        print(f"\nTotal trades: {len(completed)}")
        print(f"Wins: {wins} | Losses: {losses}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Average odds: {avg_odds:.2f}")
        print(f"Average value: {avg_value:+.1f}%")

        # Monthly breakdown
        completed["month"] = pd.to_datetime(completed["date"]).dt.to_period("M")
        monthly = completed.groupby("month").agg({
            "profit": "sum",
            "stake": "sum",
            "actual_result": "count"
        }).rename(columns={"actual_result": "trades"})

        if len(monthly) > 1:
            print(f"\nMonthly breakdown:")
            for month, row in monthly.iterrows():
                month_roi = (row["profit"] / row["stake"]) * 100 if row["stake"] > 0 else 0
                print(f"  {month}: €{row['profit']:+.2f} ({row['trades']} trades, {month_roi:+.1f}% ROI)")

    if len(pending) > 0:
        print(f"\nPending trades: {len(pending)}")
        total_at_risk = pending["stake"].sum()
        print(f"Total at risk: €{total_at_risk:.2f}")
        for _, trade in pending.iterrows():
            print(f"  {trade['home_team']} vs {trade['away_team']}: {trade['prediction']} @ {trade['odds']} (€{trade['stake']:.2f})")

    # Break-even analysis
    if len(completed) >= 10:
        print(f"\n{'='*60}")
        print("EDGE ANALYSIS")
        print(f"{'='*60}")

        # Expected vs actual
        expected_profit = (completed["value"] * completed["stake"]).sum()
        actual_profit = completed["profit"].sum()

        print(f"Expected profit (based on value): €{expected_profit:+.2f}")
        print(f"Actual profit: €{actual_profit:+.2f}")
        print(f"Variance: €{actual_profit - expected_profit:+.2f}")

        if len(completed) >= 50:
            if roi > 5:
                print("\n✓ Looking good! ROI > 5% after 50+ bets suggests possible edge.")
            elif roi > 0:
                print("\n~ Marginally profitable. Need more data to confirm edge.")
            else:
                print("\n✗ Losing money. Model may not have a real edge.")


def show_pending():
    """Show pending trades."""
    trades = load_trades()
    pending = trades[trades["actual_result"].isna()]

    if len(pending) == 0:
        print("No pending trades.")
        return

    print(f"\n{'='*50}")
    print("PENDING TRADES")
    print(f"{'='*50}")

    for _, trade in pending.iterrows():
        print(f"\n{trade['home_team']} vs {trade['away_team']}")
        print(f"  Prediction: {trade['prediction']} @ {trade['odds']}")
        print(f"  Stake: €{trade['stake']:.2f}")
        print(f"  Value: {trade['value']*100:+.1f}%")
        print(f"  Match date: {trade['match_date']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python paper_trade.py bet <home> <away> <odds> [match_date]")
        print("  python paper_trade.py settle <home> <away> <H/D/A>")
        print("  python paper_trade.py stats")
        print("  python paper_trade.py pending")
        print("\nExamples:")
        print("  python paper_trade.py bet Sunderland Leeds 2.625 2025-12-28")
        print("  python paper_trade.py settle Sunderland Leeds H")
        sys.exit(0)

    command = sys.argv[1]

    if command == "bet" and len(sys.argv) >= 5:
        home = sys.argv[2]
        away = sys.argv[3]
        odds = float(sys.argv[4])
        match_date = sys.argv[5] if len(sys.argv) > 5 else None
        place_trade(home, away, odds, match_date)

    elif command == "settle" and len(sys.argv) == 5:
        home = sys.argv[2]
        away = sys.argv[3]
        result = sys.argv[4].upper()
        if result not in ["H", "D", "A"]:
            print("Result must be H, D, or A")
        else:
            settle_trade(home, away, result)

    elif command == "stats":
        show_stats()

    elif command == "pending":
        show_pending()

    else:
        print("Invalid command. Run without arguments for usage.")
