# Stock Prediction Project - Quick Start Guide

## What This Project Does

Predicts daily stock price movements using:
- **Technical indicators** (SMA, RSI, MACD, Bollinger Bands)
- **News sentiment** (VADER sentiment analysis on 442K+ financial articles)
- **Politician trading signals** (Congressional STOCK Act disclosures)

**Key Finding:** Achieved 60-70% accuracy on financial sector stocks (WFC, PFE) with superior risk-adjusted returns (Sharpe ratio 2.22).

---

## Project Structure

```
StockPrediction/
├── src/                    # Core source code
│   ├── data_loader.py      # Fetch stocks, news, politician trades
│   ├── feature_engineering.py  # Create 61 features
│   ├── model_xgboost.py    # XGBoost model with regularization
│   └── config.py           # Configuration & API keys
│
├── scripts/                # Analysis scripts
│   ├── validate_multiyear.py   # Multi-year validation (8 stocks, 2 years)
│   ├── economic_backtest.py    # Trading simulation with costs
│   └── live_prediction_demo.py # Real-time predictions
│
├── data/                   # Datasets
│   ├── archive/            # 442K news articles (CSV)
│   └── SentimentAnalysis/  # FinancialPhraseBank
│
├── results/                # Experiment outputs (CSV)
├── models/                 # Trained sentiment model
├── visualizations/         # 14 result charts (PNG)
└── tests/                  # Unit tests
```

---

## Getting Started

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
QUIVER_API_KEY=your_quiver_api_key_here
NEWS_API_KEY=your_newsapi_key_here  # Optional, for live demo
```

Get keys from:
- [Quiver Quantitative](https://www.quiverquant.com/) - Politician trading data
- [NewsAPI](https://newsapi.org/) - Real-time news (optional)

### 3. Verify Setup

```bash
python -c "from src.data_loader import *; from src.feature_engineering import *; print('Setup complete!')"
```

---

## Run the Project

### Option 1: Full Validation (Recommended)

Run multi-year validation across 8 stocks:

```bash
python scripts/validate_multiyear.py
```

**Output:** Accuracy results for each stock-year combination, saved to `results/`.

### Option 2: Economic Backtest

Test trading performance with transaction costs:

```bash
python scripts/economic_backtest.py
```

**Output:** Sharpe ratio, returns vs buy-and-hold, win rates.

### Option 3: Live Prediction Demo

Generate real-time BUY/SELL signals:

```bash
python scripts/live_prediction_demo.py
```

**Output:** Predictions for 6 stocks with confidence levels.

---

## Key Results

| Metric | Value |
|--------|-------|
| Best Accuracy | 70.0% (WFC 2018) |
| Avg Accuracy | 51.8% (above 50% random) |
| Sharpe Ratio | 2.22 (excellent risk-adjusted) |
| Win Rate | 61.7% |

**Sector Performance:**
- Financials (WFC): 60-70% 
- Healthcare (PFE): 58-61% 
- Technology (NVDA, TSLA): 35-55% 

---

## What to Explore

### 1. Check Results

```bash
# View validation results
cat results/multiyear_validation_results.csv

# View economic backtest
cat results/economic_backtest_results.csv
```

### 2. View Visualizations

All charts are in `visualizations/`:
- `sector_performance.png` - Performance by sector
- `overfitting_analysis.png` - Train vs test accuracy
- `economic_performance.png` - Trading simulation results

### 3. Modify Stocks

Edit the stock list in any validation script:

```python
STOCKS = ['WFC', 'PFE', 'BABA']  # Add your own tickers
```

---

## Troubleshooting

### Import errors
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### API key errors
```bash
# Check .env file exists and has keys
cat .env
```

### yfinance not working
- Try recent date ranges (last 2 years)
- Use major tickers (AAPL, MSFT, WFC)

---

## Documentation

- **GRADER_README.md** - Grading guide and full execution
- **scripts/README.md** - Script descriptions

---

## Disclaimer

This project is for **educational/research purposes only**. Not financial advice. Past performance does not equal future results.

---

**Ready to go! Start with:**

```bash
python scripts/validate_multiyear.py
```
