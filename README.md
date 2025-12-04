# Stock Movement Prediction with News Sentiment and Politician Trading Signals

A machine learning project investigating whether congressional trading signals combined with news sentiment and technical indicators can improve daily stock price prediction.

## Research Question

> Does incorporating politician-trade signals into a prediction pipeline (alongside technical indicators and news sentiment) improve daily stock direction prediction and yield incremental economic value?

---

## Key Results

### Prediction Accuracy

| Sector | Stock | 2018 | 2019 | Average |
|--------|-------|------|------|---------|
| **Financials** | WFC | **70.0%** | 62.5% | 66.3% |
| **Healthcare** | PFE | 57.9% | 61.0% | 59.5% |
| **International** | BABA | 52.6% | **67.7%** | 60.2% |
| Technology | NFLX | 43.8% | 47.6% | 45.7% |
| Technology | GOOGL | 52.6% | 47.4% | 50.0% |
| Technology | NVDA | 42.1% | 35.0% | 38.6% |
| **Overall** | | 47.1% | 53.6% | **51.8%** |

### Economic Performance

| Metric | Value |
|--------|-------|
| Average Sharpe Ratio | **2.22** (excellent) |
| Best Excess Return | +9.5% (WFC 2018 vs buy-and-hold) |
| Average Win Rate | 61.7% |

**Key Finding:** Politician trading signals provide value for **financial and healthcare sectors**, but limited utility for technology stocks.

---

## Features

### Data Sources
- **Stock Prices**: Yahoo Finance (OHLCV data)
- **News Sentiment**: 442,000 financial articles with VADER scoring
- **Politician Trades**: Congressional STOCK Act disclosures via Quiver Quantitative

### Feature Engineering (61 features)
- **Technical Indicators** (20): SMA, RSI, MACD, Bollinger Bands, volatility
- **Sentiment Features** (4): Compound, positive, negative scores, article count
- **Politician Signals** (23): Net trade index, conviction score, temporal patterns
- **Market Context** (14): SPY, QQQ, VIX relative strength

### Model
- **XGBoost** with aggressive regularization
- Walk-forward validation (80/20 train/test split)
- Multi-year testing (2018-2019)

---

## Project Structure

```
StockPrediction/
├── src/                        # Core source code
│   ├── data_loader.py          # Data fetching (stocks, news, trades)
│   ├── feature_engineering.py  # 61 feature creation
│   ├── model_xgboost.py        # XGBoost with regularization
│   ├── advanced_politician_features.py  # Politician signal features
│   └── config.py               # Configuration settings
│
├── scripts/                    # Analysis scripts
│   ├── validate_multiyear.py   # Main validation (8 stocks, 2 years)
│   ├── economic_backtest.py    # Trading simulation
│   ├── live_prediction_demo.py # Real-time predictions
│   ├── analyze_feature_importance.py  # Feature rankings
│   └── README.md               # Script documentation
│
├── data/                       # Datasets
│   ├── archive/                # News articles (CSV)
│   └── SentimentAnalysis/      # FinancialPhraseBank
│
├── results/                    # Experiment outputs (CSV)
├── models/                     # Trained sentiment classifier
├── visualizations/             # Charts and figures (PNG)
├── tests/                      # Unit tests
│
├── README.md                   # This file
├── QUICKSTART.md              # Quick start guide
├── GRADER_README.md           # Grading guide
└── requirements.txt           # Python dependencies
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

```bash
# Clone repository
git clone <repository-url>
cd StockPrediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"
```

### API Keys

Create a `.env` file:
```
QUIVER_API_KEY=your_key_here
NEWS_API_KEY=your_key_here  # Optional
```

Get keys from:
- [Quiver Quantitative](https://www.quiverquant.com/) (politician trades)
- [NewsAPI](https://newsapi.org/) (real-time news, optional)

### Run Validation

```bash
# Full multi-year validation
python scripts/validate_multiyear.py

# Economic backtest
python scripts/economic_backtest.py

# Live predictions
python scripts/live_prediction_demo.py
```

---

## Visualizations

All figures are in `visualizations/`:

| File | Description |
|------|-------------|
| `sector_performance.png` | Accuracy by sector |
| `overfitting_analysis.png` | Train vs test gaps |
| `economic_performance.png` | Trading simulation |
| `feature_importance_visualization.png` | Top features |

---

## Results Files

All results are in `results/`:

| File | Description |
|------|-------------|
| `multiyear_validation_results.csv` | Accuracy by stock-year |
| `economic_backtest_results.csv` | Trading performance |
| `feature_importance_rankings.csv` | Top 25 features |
| `overfitting_experiments.csv` | Regularization tests |

---

## Testing

```bash
pytest tests/ -v
```

---

## Documentation

- **QUICKSTART.md** - Get running in 5 minutes
- **GRADER_README.md** - Full execution guide for grading
- **scripts/README.md** - Script descriptions

---

## Research Findings

### What Works
1. **Financial sector stocks** respond well to politician signals (WFC: 70%)
2. **Risk-adjusted returns** are strong (Sharpe 2.22)
3. **Downside protection** during market downturns

### What Doesn't
1. **Technology stocks** show limited predictability
2. **Overfitting** persists despite regularization (38% avg gap)
3. **Absolute returns** often underperform buy-and-hold

### Implications
- Politician trading signals are **sector-specific**, not universal
- Best used for **risk management** rather than return maximization
- Supports hybrid strategies combining alternative data with fundamentals

---

## Disclaimer

This project is for **educational and research purposes only**. It should not be used for actual trading without professional financial advice. Past performance does not guarantee future results.

---

## Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) - Stock data
- [NLTK VADER](https://www.nltk.org/) - Sentiment analysis
- [XGBoost](https://xgboost.readthedocs.io/) - Machine learning
- [Quiver Quantitative](https://www.quiverquant.com/) - Politician trading data

---

## License

MIT License - See LICENSE file for details.

---

**Authors:** Conner Murphy and William Coleman
