# Stock Prediction Project - Quick Start Guide

## 🎯 What You've Got

A complete data science project structure for predicting stock movements using:
- **Technical indicators** (SMA, RSI, MACD)
- **News sentiment** (VADER sentiment analysis)
- **Politician trading signals** (Congressional trades)

## 📁 Project Files

### Core Python Modules (`src/`)
1. **data_loader.py** - Fetch stock data, news, and politician trades
   - `fetch_stock_data()` - Uses yfinance for OHLCV data
   - `fetch_news_sentiment()` - Placeholder for news + VADER sentiment
   - `fetch_politician_trades()` - Placeholder for API integration

2. **feature_engineering.py** - Create features from raw data
   - `create_features()` - Generates technical indicators + target variable
   - `handle_missing_values()` - Clean data strategies

3. **model.py** - Train and evaluate models
   - `train_model()` - Random Forest & Logistic Regression
   - `evaluate_model()` - Comprehensive metrics
   - `backtest_strategy()` - Simple trading simulation

### Jupyter Notebook (`notebooks/`)
- **01_baseline_model.ipynb** - Complete workflow with technical indicators only

### Configuration
- **requirements.txt** - All Python dependencies
- **setup.sh** - Automated setup script
- **.gitignore** - Keeps data and secrets out of git

## 🚀 Getting Started

### Option 1: Automated Setup (Recommended)
```bash
cd /Users/connermurphy/dev/StockPrediction
./setup.sh
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"
```

### Option 3: Just Install Packages
If you already have a Python environment:
```bash
pip install -r requirements.txt
```

## 📊 Running the Baseline Model

### In Jupyter Notebook (Recommended)
```bash
jupyter notebook notebooks/01_baseline_model.ipynb
```

Then run all cells to:
1. Load Apple (AAPL) stock data from 2022-2023
2. Create technical indicator features
3. Train Random Forest and Logistic Regression models
4. Evaluate with time-series cross-validation
5. See performance metrics and visualizations

### As Python Script
```python
from src import fetch_stock_data, create_features, train_model, evaluate_model
from sklearn.model_selection import train_test_split

# Fetch data
stock_data = fetch_stock_data('AAPL', '2023-01-01', '2023-12-31')

# Create features
X, y, dates = create_features(stock_data)

# Remove NaNs
X = X.dropna()
y = y.loc[X.index]

# Train-test split (time-series aware)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train and evaluate
model = train_model(X_train, y_train, model_type='random_forest')
metrics = evaluate_model(model, X_test, y_test)
```

## 🔍 What to Explore First

### 1. Try Different Tickers
In the notebook, change:
```python
TICKER = "AAPL"  # Try: "MSFT", "GOOGL", "TSLA", etc.
```

### 2. Adjust Date Ranges
```python
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"
```

### 3. Tune Model Parameters
```python
# In model.py or notebook
model = train_model(X_train, y_train, 
                   model_type='random_forest',
                   n_estimators=200,  # More trees
                   max_depth=15)      # Deeper trees
```

### 4. Check Feature Importance
After training a Random Forest, the notebook shows which features matter most.

## 📈 Expected Performance

### Baseline (Technical Indicators Only)
- **Accuracy**: ~52-58% (vs ~50% random)
- **Precision**: ~53-60%
- **F1 Score**: ~53-58%

These are starting points. The goal is to improve by adding sentiment and politician signals!

## 🛠️ Next Steps

### Phase 1: News Sentiment Integration
1. Get a free API key from [NewsAPI.org](https://newsapi.org/)
2. Update `fetch_news_sentiment()` in `data_loader.py`
3. Create notebook `02_sentiment_integration.ipynb`
4. Compare performance with baseline

### Phase 2: Politician Trading Signals
1. Get API access to [Quiver Quantitative](https://www.quiverquant.com/) or [Finnhub](https://finnhub.io/)
2. Update `fetch_politician_trades()` in `data_loader.py`
3. Create notebook `03_politician_signals.ipynb`
4. Add politician features to model

### Phase 3: Combined Model
1. Merge all three data sources
2. Train full model in `04_combined_model.ipynb`
3. Analyze feature importance
4. Answer: Do politician trades help prediction?

### Phase 4: Economic Backtest
1. Create `05_economic_backtest.ipynb`
2. Add transaction costs and slippage
3. Calculate Sharpe ratio, max drawdown
4. Answer: Can we make money with this?

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Add your own tests in `tests/` directory.

## 📝 Key Research Questions

The project is designed to answer:

1. **Do technical indicators predict stock direction?** ✅ (Baseline establishes this)
2. **Does news sentiment add value?** ⏳ (Next step)
3. **Do politician trades signal future moves?** ⏳ (Phase 2)
4. **What's the economic value?** ⏳ (Phase 4)

## 🐛 Troubleshooting

### Import errors
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Reinstall packages
pip install -r requirements.txt
```

### yfinance not working
Some tickers may not have data for all periods. Try:
- Recent date ranges (last 2 years)
- Major stocks (AAPL, MSFT, GOOGL)
- Check if ticker symbol is correct

### NLTK errors
```python
import nltk
nltk.download('vader_lexicon')
```

## 📚 Learning Resources

### Understanding the Code
- **Technical Analysis**: [ta library docs](https://technical-analysis-library-in-python.readthedocs.io/)
- **VADER Sentiment**: [NLTK VADER guide](https://www.nltk.org/howto/sentiment.html)
- **Time Series CV**: [sklearn docs](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

### Financial ML
- Book: "Advances in Financial Machine Learning" by Marcos López de Prado
- Course: Coursera Machine Learning for Trading

## 💡 Tips

1. **Start small**: Run baseline on 1-2 years of data first
2. **Visualize everything**: The notebook includes plotting code
3. **Don't overfit**: Use time-series split, not random split
4. **Test incrementally**: Add one feature set at a time
5. **Document findings**: Keep notes in markdown cells

## ⚠️ Important Notes

- **Not financial advice**: This is educational/research only
- **Past performance ≠ future results**: Markets change
- **Transaction costs matter**: Real trading has fees and slippage
- **Data quality**: Garbage in = garbage out
- **Test thoroughly**: Before using real money (which you shouldn't!)

## 🎓 Educational Goals

By completing this project, you'll learn:
- ✅ Feature engineering for time series
- ✅ Technical analysis implementation
- ✅ Sentiment analysis with NLP
- ✅ Time-series cross-validation
- ✅ Model evaluation for classification
- ✅ Backtesting trading strategies
- ✅ API integration for financial data
- ✅ End-to-end ML project structure

## 🤝 Need Help?

- Check the comprehensive docstrings in each `.py` file
- Run example code at bottom of each module: `python src/data_loader.py`
- Read the notebook markdown cells for context
- Check the main README.md

---

**You're all set! Open the notebook and start exploring.** 🚀

```bash
jupyter notebook notebooks/01_baseline_model.ipynb
```
