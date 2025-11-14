# 📁 Project Structure

Clean, organized codebase for academic paper and live trading validation.

---

## 📂 Directory Layout

```
StockPrediction/
├── src/                          # Core source code
│   ├── config.py                 # Configuration & API keys
│   ├── data_loader.py            # Data fetching & loading
│   ├── feature_engineering.py    # Feature creation & preprocessing
│   ├── model_xgboost.py          # XGBoost implementation
│   ├── model_lstm.py             # LSTM implementation
│   └── model_gru.py              # GRU implementation
│
├── scripts/                      # Executable scripts
│   ├── validate_multiyear.py     # Main validation (16 experiments)
│   ├── validate_with_feature_selection.py  # Top-20 features validation
│   ├── compare_sequence_models.py          # XGBoost vs LSTM vs GRU
│   ├── economic_backtest.py      # Trading strategy backtest
│   ├── live_prediction_demo.py   # Live BUY/SELL demo (presentation)
│   └── daily_prediction_tracker.py         # Daily prediction logging
│
├── paper/                        # Academic paper (LaTeX)
│   ├── stock_prediction_paper.tex          # Main paper
│   ├── references.bib            # Bibliography
│   └── README.md                 # Compilation instructions
│
├── docs/                         # Documentation
│   ├── FINAL_VALIDATION_SUMMARY.md         # Complete validation results
│   ├── VALIDATION_RESULTS_ANALYSIS.md      # In-depth analysis
│   ├── PAPER_GOALS_EFFICACY_ANALYSIS.md   # Project vs paper goals
│   ├── EXECUTIVE_SUMMARY_PAPER_EFFICACY.md
│   ├── PAPER_ALIGNMENT_MATRIX.md
│   ├── ACTION_PLAN_PAPER_COMPLETION.md
│   └── QUICK_REFERENCE_PAPER_STATUS.md
│
├── data/                         # Data files (not in repo)
│   ├── archive/                  # News datasets (442K articles)
│   └── [Downloaded via APIs]     # Stock & politician data
│
├── results/                      # Output files
│   ├── multiyear_validation_results.csv
│   ├── economic_backtest_results.csv
│   ├── stock_coverage_analysis.csv
│   ├── feature_importance_rankings.csv
│   └── daily_predictions_log.csv
│
├── README.md                     # Main project documentation
├── PAPER_COMPLETE.md             # Paper summary & compilation
├── ECONOMIC_BACKTEST_RESULTS.md  # Backtest details & analysis
├── PRESENTATION_DEMO_GUIDE.md    # Presentation walkthrough
├── LIVE_TRADING_LOG.md           # Real trading tracker
├── LIVE_TRADING_GUIDE.md         # Live trading workflow
├── requirements.txt              # Python dependencies
└── .env                          # API keys (not in repo)
```

---

## 🎯 Key Files by Purpose

### For Reproducing Paper Results:
1. `scripts/validate_multiyear.py` - Run all 16 validation experiments
2. `scripts/economic_backtest.py` - Economic validation
3. `docs/FINAL_VALIDATION_SUMMARY.md` - Complete results

### For Understanding the System:
1. `README.md` - Project overview
2. `PAPER_COMPLETE.md` - Paper summary
3. `src/feature_engineering.py` - 61 features explained
4. `docs/VALIDATION_RESULTS_ANALYSIS.md` - Deep dive into results

### For Presentations:
1. `scripts/live_prediction_demo.py` - Live demo script
2. `PRESENTATION_DEMO_GUIDE.md` - Complete guide
3. `ECONOMIC_BACKTEST_RESULTS.md` - Talking points

### For Live Trading:
1. `scripts/daily_prediction_tracker.py` - Daily predictions
2. `LIVE_TRADING_GUIDE.md` - Workflow
3. `LIVE_TRADING_LOG.md` - Trade tracker

---

## 📊 Data Files

### Input Data (data/):
- `archive/analyst_ratings_processed.csv` (1.4M rows)
- `archive/raw_analyst_ratings.csv` (1.4M rows)
- `archive/raw_partner_headlines.csv` (1.8M rows)
- **Total:** 442K unique news articles (2014-2020)

### Results (results/):
- `multiyear_validation_results.csv` - All validation results
- `economic_backtest_results.csv` - Backtest outcomes
- `daily_predictions_log.csv` - Live predictions log

---

## 🔧 Core Modules

### src/config.py
- API endpoints & keys
- File paths
- Model hyperparameters
- Validation stocks list

### src/data_loader.py
- `fetch_stock_data()` - Yahoo Finance integration
- `fetch_politician_trades()` - Quiver API integration
- `fetch_news_sentiment()` - NewsAPI integration
- `fetch_historical_news_kaggle()` - Load CSV news
- `aggregate_daily_sentiment()` - VADER sentiment analysis

### src/feature_engineering.py
- `create_features()` - Generate 61 features
- Technical indicators (SMA, RSI, MACD, volatility)
- Sentiment features (compound, positive, negative, count)
- Politician trading features (23 advanced metrics)
- Market context (SPY, QQQ, VIX)

### src/model_xgboost.py
- `train_xgboost_model()` - Train with regularization
- `evaluate_xgboost_model()` - Calculate metrics
- Aggressive regularization to combat overfitting

---

## 🎓 Paper Components

### paper/stock_prediction_paper.tex
- 12-page LaTeX document
- Abstract, Introduction, Literature Review
- Data, Methodology, Results, Discussion
- Complete with tables and citations

### Key Results:
- WFC 2018: 70.0% accuracy
- BABA 2019: 67.7% accuracy
- PFE 2019: 61.0% accuracy
- Sharpe Ratio: 2.22
- Average: 51.8% (honest reporting)

---

## 🚀 Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Add API keys to .env
echo "QUIVER_API_KEY=your_key" >> .env
echo "NEWS_API_KEY=your_key" >> .env

# Run validation
python scripts/validate_multiyear.py

# Run economic backtest
python scripts/economic_backtest.py

# Live demo
python scripts/live_prediction_demo.py
```

---

## 📝 Notes

- **Data not included:** News CSVs are 4GB+ (use provided data loader scripts)
- **API keys required:** Quiver Quantitative, NewsAPI (optional)
- **Python 3.8+** required
- **Training time:** ~10-15 min for full validation

---

This structure supports:
✅ Academic reproducibility  
✅ Live trading validation  
✅ Presentation demos  
✅ Future extensions

