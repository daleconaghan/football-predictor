# Premier League Match Predictor & Betting Strategy

A machine learning system that predicts English Premier League match outcomes and tests betting strategies against market odds using statistical edge detection and Kelly Criterion bankroll management.

## Current Performance

**Prediction Accuracy:** 46.7% (14/30 matches)
**Paper Trading Win Rate:** 33.3% (4/12 settled bets)
**Current Bankroll:** €897.79 (starting: €1,000)
**Total P&L:** -€102.21 (-10.2% ROI)
**Best Trade:** Fulham @ 3.30 odds (+€108.70 profit)

*Last Updated: January 8, 2026 (Gameweek 21)*

## What This Does

This project combines:
- **Match outcome prediction** using XGBoost machine learning model
- **Market efficiency testing** by comparing model probabilities to bookmaker odds
- **Betting strategy simulation** with proper bankroll management (Kelly Criterion)
- **Performance tracking** across multiple gameweeks with full P&L analysis

Unlike typical prediction projects that just report accuracy, this tests whether statistical predictions can find value in real betting markets.

## Methodology

### 1. Data Collection (`collect_data.py`)
- Scrapes historical Premier League match data
- Stores results, team stats, and match outcomes

### 2. Feature Engineering (`features.py`)
- Creates rolling team statistics (form, goals scored/conceded, home/away performance)
- Generates head-to-head metrics
- Builds temporal features (days since last match, etc.)

### 3. Model Training (`train_model.py`)
- XGBoost classifier for multi-class prediction (Home/Draw/Away)
- Outputs probability estimates for each outcome
- Cross-validation for model evaluation

### 4. Prediction (`predict.py`)
- Generates match predictions with confidence scores
- Outputs probabilities for H/D/A outcomes
- Saves predictions to CSV for tracking

### 5. Paper Trading (`paper_trade.py`)
- Compares model probabilities to bookmaker odds
- Identifies value bets (model probability > implied odds probability)
- Uses Kelly Criterion for stake sizing
- Tracks bankroll and P&L over time

### 6. Backtesting (`backtest.py`)
- Tests strategy on historical data
- Evaluates performance metrics

## Key Findings

### What's Working
- **High-confidence predictions** (>65%) are more reliable
- **January performance** (+€61.35, 40.9% ROI) vs December (-€41.03)
- **Value detection** on longer odds (Fulham @ 3.30 was a profitable outlier)

### Challenges
- **Draw-heavy weeks** reduce prediction accuracy (4 draws in GW21)
- **Lower-confidence bets** (<50%) are unprofitable
- **Market efficiency** makes consistent edge difficult
- **Variance** in small sample sizes (only 30 matches tracked so far)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/football-predictor.git
cd football-predictor

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Full Pipeline
```bash
# Collect data, train model, generate predictions
python run.py
```

### Make Predictions for Upcoming Matches
```bash
# Predict specific fixtures
python predict.py --fixtures
```

### Track Paper Trading Results
```bash
# Log bet outcomes and update P&L
python paper_trade.py
```

## Project Structure

```
football-predictor/
├── collect_data.py      # Data collection pipeline
├── features.py          # Feature engineering
├── train_model.py       # Model training
├── predict.py           # Generate predictions
├── paper_trade.py       # Betting strategy simulation
├── backtest.py          # Historical performance testing
├── tracker.py           # Results tracking utilities
├── run.py               # Main pipeline runner
├── requirements.txt     # Dependencies
├── data/
│   ├── premier_league.csv      # Historical match data
│   ├── features.csv            # Engineered features
│   ├── predictions.csv         # Match predictions & results
│   ├── paper_trades.csv        # Bet tracking with P&L
│   └── backtest_results.csv    # Backtesting data
└── model/                      # Trained model files
```

## Tech Stack

- **Python 3.x**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - ML utilities
- **xgboost** - Gradient boosting classifier

## Lessons Learned

1. **Prediction accuracy ≠ profitability** - 46% accuracy can be profitable if you bet on the right matches
2. **Market efficiency is real** - bookmakers are very good at pricing matches
3. **Bankroll management matters** - Kelly Criterion prevents catastrophic losses
4. **Small edges compound** - Finding 2-3% edge per bet can be profitable long-term
5. **Variance is brutal** - Short-term results don't indicate long-term performance

## Next Steps

- [ ] Expand to 50+ matches for statistical significance
- [ ] Incorporate more advanced features (player injuries, weather, etc.)
- [ ] Test different staking strategies (flat stakes, fractional Kelly)
- [ ] Build confidence thresholds (only bet on >60% confidence predictions)
- [ ] Compare XGBoost to other models (Random Forest, Neural Networks)

## Data Sources

- Historical match data scraped from public sources
- Odds data manually collected from bookmaker websites for paper trading simulation

## Disclaimer

This project is for **educational purposes only**. It simulates betting strategies with paper trading (no real money). Gambling involves risk, and past performance does not guarantee future results. The model's performance may not be representative of real-world betting success.

---

**Status:** Active development | Updated weekly with new gameweek results
