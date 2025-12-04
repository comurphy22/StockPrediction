# Grader README - Course Project Execution Guide

**Project:** Stock Movement Prediction with News Sentiment and Politician Trading Signals  
**Authors:** Conner Murphy and William Coleman  
**Date:** December 2025

---

## One-Command Execution

To run **all experiments** with a single command:

```bash
python RUN_ALL.py
```

**Estimated Runtime:** 30-40 minutes  
**What it runs:**
1. Data validation (2-3 min)
2. Multi-year validation - 16 experiments (10-15 min)
3. Feature selection validation (8-12 min)
4. Economic backtesting (10-15 min)
5. Live prediction demo (2-3 min)

---

## Prerequisites

### System Requirements
- **Python:** 3.8 or higher
- **RAM:** 8GB minimum
- **OS:** macOS, Linux, or Windows

### Installation

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"
```

### API Keys (Required)

Create a `.env` file in the project root:

```bash
QUIVER_API_KEY=your_quiver_api_key_here
NEWS_API_KEY=your_news_api_key_here  # Optional
```

**Getting API Keys:**
- **Quiver Quantitative** (required): [https://www.quiverquant.com/](https://www.quiverquant.com/)
- **NewsAPI** (optional): [https://newsapi.org/](https://newsapi.org/)

**Note:** Historical analysis works without NewsAPI using the included Kaggle datasets.

---

## Verification Before Running

```bash
python -c "
from src.data_loader import *
from src.feature_engineering import *
from src.model_xgboost import *
print('All dependencies loaded successfully')
"
```

---

## Running the Complete Pipeline

```bash
python RUN_ALL.py
```

**Expected Output:**
```
================================================================================
 STOCK MOVEMENT PREDICTION - COMPLETE EXPERIMENTAL PIPELINE
================================================================================

STEP 1/5: DATA VALIDATION
STEP 2/5: MULTI-YEAR VALIDATION (16 experiments)
STEP 3/5: FEATURE SELECTION
STEP 4/5: ECONOMIC BACKTESTING
STEP 5/5: LIVE PREDICTION DEMO

================================================================================
 ALL EXPERIMENTS COMPLETE!
================================================================================

RESULTS:
   - results/multiyear_validation_results.csv
   - results/economic_backtest_results.csv
   - results/feature_importance_rankings.csv

KEY FINDINGS:
   - Best Accuracy: WFC 70.0% (2018), BABA 67.7% (2019)
   - Sharpe Ratio: 2.22 (excellent risk-adjusted returns)
   - Sector Performance: Financials (66%) > Healthcare (60%) > Tech (39%)
```

---

## Results Files

After running, check `results/` for:

| File | Description |
|------|-------------|
| `multiyear_validation_results.csv` | 16 experiments (8 stocks x 2 years) |
| `economic_backtest_results.csv` | Trading simulation with costs |
| `feature_importance_rankings.csv` | Top 25 features |

---

## Running Individual Experiments

```bash
# Multi-Year Validation (10-15 min)
python scripts/validate_multiyear.py

# Economic Backtesting (10-15 min)
python scripts/economic_backtest.py

# Live Prediction Demo (2-3 min)
python scripts/live_prediction_demo.py
```

---

## Project Structure

```
StockPrediction/
├── src/                    # Core source code
│   ├── data_loader.py      # Data fetching
│   ├── feature_engineering.py  # 61 features
│   └── model_xgboost.py    # XGBoost model
│
├── scripts/                # Runnable experiments
│   ├── validate_multiyear.py
│   ├── economic_backtest.py
│   └── live_prediction_demo.py
│
├── data/                   # News datasets
├── results/                # Output CSVs
├── models/                 # Trained models
├── visualizations/         # Charts (PNG)
│
├── README.md              # Project overview
└── QUICKSTART.md          # Quick start guide
```

---

## Expected Results

### Validation Accuracy

| Stock | Year | Expected Accuracy |
|-------|------|-------------------|
| WFC | 2018 | ~70% |
| BABA | 2019 | ~68% |
| PFE | 2019 | ~61% |
| Average | All | ~52% |

### Economic Performance

| Metric | Expected Value |
|--------|----------------|
| Sharpe Ratio | ~2.0-2.5 |
| Win Rate | ~60-65% |
| WFC Excess Return | ~+8-11% |

---

## Troubleshooting

### "Module not found" errors
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "API key not found" errors
```bash
cat .env  # Verify keys are set
```

### Data file errors
```bash
ls data/archive/  # Should show CSV files
```

---

## Runtime Estimates

| Script | Runtime |
|--------|---------|
| RUN_ALL.py | 30-40 min |
| validate_multiyear.py | 10-15 min |
| economic_backtest.py | 10-15 min |
| live_prediction_demo.py | 2-3 min |

---

## Grading Checklist

Before running:
- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] `.env` file with API keys
- [ ] Data files in `data/archive/`

Run:
- [ ] `python RUN_ALL.py`
- [ ] Wait 30-40 minutes

Verify:
- [ ] `results/multiyear_validation_results.csv` exists (16 rows)
- [ ] `results/economic_backtest_results.csv` exists
- [ ] Metrics match expected ranges

---

**Authors:** Conner Murphy and William Coleman  
**Contact:** See paper for contact information
