# ⚙️ Setup Instructions

Quick guide to get the project running on your machine.

---

## 📋 Prerequisites

- **Python 3.8+** (tested on 3.8, 3.9, 3.10)
- **pip** (package installer)
- **10GB disk space** (for data files)
- **API keys** (see below)

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourrepo/StockPrediction.git
cd StockPrediction
```

### 2. Create Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- pandas, numpy (data processing)
- scikit-learn (ML models)
- xgboost (gradient boosting)
- tensorflow (deep learning)
- yfinance (stock data)
- vaderSentiment (sentiment analysis)
- requests (API calls)

---

## 🔑 API Keys Setup

### Required: Quiver Quantitative

**For politician trading data:**

1. Go to [https://www.quiverquant.com/](https://www.quiverquant.com/)
2. Sign up for an account
3. Copy your API key
4. Create `.env` file in project root:

```bash
# Create .env file
touch .env

# Add to .env:
QUIVER_API_KEY=your_actual_key_here
```

### Optional: NewsAPI

**For real-time news (live demo only):**

1. Go to [https://newsapi.org/](https://newsapi.org/)
2. Sign up for free tier
3. Copy your API key
4. Add to `.env`:

```bash
NEWS_API_KEY=your_actual_key_here
```

**Note:** Historical news works without this key using Kaggle data.

---

## 📊 Data Setup

### Option A: Use Existing Kaggle Data (Recommended)

The project uses historical news data from Kaggle CSVs in `data/archive/`:
- `analyst_ratings_processed.csv` (1.4M rows)
- `raw_analyst_ratings.csv` (1.4M rows)
- `raw_partner_headlines.csv` (1.8M rows)

✅ **This data is already included if you cloned with data files**

### Option B: Download Fresh Data

If data folder is missing:

1. Download from Kaggle: [Financial News Dataset](https://www.kaggle.com/)
2. Extract CSVs to `data/archive/`
3. Verify structure:
   ```bash
   ls data/archive/
   # Should show: analyst_ratings_processed.csv, raw_analyst_ratings.csv, raw_partner_headlines.csv
   ```

---

## ✅ Verify Setup

Run this test to ensure everything works:

```bash
python -c "
from src.data_loader import *
from src.feature_engineering import *
from src.model_xgboost import *
print('✅ All imports successful!')
"
```

If you see `✅ All imports successful!`, you're ready to go!

---

## 🎯 First Run

Try a quick validation on one stock:

```bash
# This will:
# 1. Load WFC stock data (2018)
# 2. Fetch politician trades
# 3. Load news sentiment
# 4. Train XGBoost model
# 5. Show accuracy results

python scripts/validate_multiyear.py
```

**Expected output:**
```
════════════════════════════════════════════════════════════
     MULTI-YEAR VALIDATION: Stock Movement Prediction
════════════════════════════════════════════════════════════

✅ Processing: WFC (2018)
   [1/7] Fetching stock data...        ✅ 251 days
   [2/7] Loading news sentiment...     ✅ 823 articles
   [3/7] Fetching politician trades... ✅ 153 trades
   ...
   Final Accuracy: 70.0%
```

---

## 🐛 Troubleshooting

### "Module not found" errors
```bash
# Make sure you activated the virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### "API key not found" errors
```bash
# Check .env file exists
ls -a | grep .env

# Check .env contents (should show your keys)
cat .env

# Make sure src/config.py loads properly
python -c "from src.config import QUIVER_API_KEY; print(QUIVER_API_KEY[:10])"
```

### "File not found" errors
```bash
# Check data folder structure
ls data/archive/

# Should contain:
# - analyst_ratings_processed.csv
# - raw_analyst_ratings.csv
# - raw_partner_headlines.csv
```

### Out of memory errors
```bash
# Reduce number of stocks being processed
# Edit src/config.py:
VALIDATION_STOCKS = ['WFC', 'BABA']  # Just 2 stocks instead of 8
```

---

## 📁 Project Structure After Setup

```
StockPrediction/
├── venv/                    # Virtual environment (created by you)
├── .env                     # API keys (created by you)
├── data/
│   └── archive/             # News CSV files
├── src/                     # Source code
├── scripts/                 # Runnable scripts
├── results/                 # Output files (created when you run scripts)
├── paper/                   # LaTeX paper
├── docs/                    # Documentation
├── requirements.txt         # Dependencies
└── README.md               # Main documentation
```

---

## 🚀 Next Steps

Once setup is complete:

1. **Run full validation:**
   ```bash
   python scripts/validate_multiyear.py
   ```

2. **Run economic backtest:**
   ```bash
   python scripts/economic_backtest.py
   ```

3. **Try live demo:**
   ```bash
   python scripts/live_prediction_demo.py
   ```

4. **Read the paper:**
   - See `paper/stock_prediction_paper.tex`
   - Or read `PAPER_COMPLETE.md` for summary

5. **Check results:**
   - `docs/FINAL_VALIDATION_SUMMARY.md` - Complete validation
   - `ECONOMIC_BACKTEST_RESULTS.md` - Trading performance

---

## 💡 Tips

- **First time?** Start with `python scripts/live_prediction_demo.py` - it's fast and impressive
- **For research?** Run `python scripts/validate_multiyear.py` for full reproducibility
- **For presentations?** See `PRESENTATION_DEMO_GUIDE.md` for walkthrough
- **For live trading?** Read `LIVE_TRADING_GUIDE.md` for workflow

---

## ⏱️ Expected Runtimes

On a typical laptop (4-core, 16GB RAM):
- **Live demo:** 2-3 minutes
- **Single stock validation:** 1-2 minutes
- **Full validation (8 stocks, 3 years):** 10-15 minutes
- **Economic backtest:** 10-15 minutes

---

## 🆘 Getting Help

If you encounter issues:

1. Check `docs/` folder for detailed documentation
2. Read `CONTRIBUTING.md` for development guidelines
3. Open an issue with:
   - Error message
   - Steps to reproduce
   - Your Python version: `python --version`
   - Your OS: `uname -a` (Mac/Linux) or `ver` (Windows)

---

**Setup complete! Happy predicting! 📈**

