# Stock Movement Prediction with News Sentiment and Politician Position Signals

A machine learning project to predict daily stock price movements using technical indicators, news sentiment analysis, and politician trading signals.

## 📊 Project Overview

**Goal:** Build a predictive model for daily stock direction that answers the research question:

> Does incorporating politician-trade signals into a prediction pipeline (alongside technical indicators and news sentiment) improve daily stock direction prediction and yield incremental economic value in a simple backtest?

### 🎉 Latest Results (November 2025)

#### Current Best Model: V1 Baseline
**Model:** XGBoost with 25 optimal features  
**Test Scope:** 10 diverse stocks (2019 data)  
**Average Test Accuracy:** **56.24%** (12% edge over random)  
**Status:** ✅ **Realistic ceiling reached**

#### Investigation Complete
After testing sentiment improvements (V3, V4) and regularization approaches (V5), we determined:
- **56% accuracy is the realistic ceiling** for daily predictions with current features
- **Overfitting (44% gap)** is fundamental to the problem, not fixable by hyperparameters
- **Stock-specific variance** is high (40% to 84% accuracy across stocks)
- **Economic value exists** IF properly risk-managed

| Performance Tier (V1) | Count | Stocks |
|-----------------|-------|--------|
| Excellent (>70%) | 1 | BABA (83.87%) |
| Good (60-70%) | 2 | QCOM (66.67%), AAPL (63.41%) |
| Moderate (50-60%) | 4 | NFLX, NVDA, MSFT, GOOGL |
| Poor (<50%) | 3 | AMZN, MU, TSLA |

**Key Findings:**
- **56% daily accuracy is realistic ceiling** - daily stock movements are inherently noisy
- **More data doesn't help** - V3 with 3154 articles performed worse than V1 with 0 articles for AAPL
- **Better sentiment doesn't help much** - V4 with expert-labeled classifier improved <1%
- **Regularization provides minimal benefit** - V5 with alpha=1.0 improved only +1.4% on average
- **Stock-specific behavior matters** - BABA (84%), AAPL (63%), MU (42%) show huge variance
- **Politician trading signals work** - AAPL achieved 63% with NO news data

📄 **[Full Investigation](docs/PROJECT_FINAL_SUMMARY.md)** | 📊 **[Option A Results](docs/OPTION_A_INVESTIGATION_RESULTS.md)** | � **[MVP Analysis](docs/MVP_VALIDATION_SUMMARY.md)**

## 🎯 Core Features

1. **Technical Indicators**
   - Simple Moving Averages (SMA): 10, 20, 50-day
   - Relative Strength Index (RSI): 14-day
   - Moving Average Convergence Divergence (MACD)
   - Price and volume momentum features

2. **News Sentiment Analysis**
   - VADER sentiment scoring
   - Daily aggregated sentiment scores
   - Integration with news APIs

3. **Politician Trading Signals**
   - Congressional trading data
   - Buy/sell frequency and volume
   - Trading activity indicators

## 📁 Project Structure

```
StockPrediction/
├── src/                         # Core source code
│   ├── data_loader.py           # Data fetching (stocks, news, politician trades)
│   ├── feature_engineering.py   # Feature creation and processing
│   ├── model_xgboost.py         # XGBoost model with regularization
│   ├── advanced_politician_features.py  # Advanced politician signal features
│   └── config.py                # Configuration settings
│
├── scripts/                     # Analysis and validation scripts
│   ├── validate_mvp.py          # V1: Baseline validation (10 stocks)
│   ├── validate_mvp_v5_optimized.py  # V5: Optimized with L1 regularization
│   ├── fix_overfitting_experiments.py  # Hyperparameter optimization
│   ├── summarize_overfitting_results.py  # Results analysis
│   ├── analyze_feature_importance.py  # Feature importance analysis
│   ├── train_financial_sentiment.py   # Sentiment model training
│   └── visualize_mvp_results.py       # Results visualization
│
├── results/                     # Experiment results (CSV)
│   ├── mvp_validation_results.csv     # V1 baseline results
│   ├── overfitting_experiments.csv    # Regularization experiments
│   └── feature_importance_rankings.csv # Top 25 features
│
├── logs/                        # Execution logs
├── docs/                        # Documentation
│   ├── MVP_RESULTS.md           # Detailed validation results
│   └── MVP_VALIDATION_SUMMARY.md # V1-V4 comparison analysis
│
├── models/                      # Trained models
├── visualizations/              # Charts and graphs
├── archive/                     # Old experiments (V2-V4, etc.)
└── notebooks/                   # Jupyter notebooks
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd StockPrediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (for sentiment analysis):
```python
python -c "import nltk; nltk.download('vader_lexicon')"
```

### Quick Start

Run the baseline model notebook:
```bash
jupyter notebook notebooks/01_baseline_model.ipynb
```

Or use the Python scripts directly:

```python
from src import fetch_stock_data, create_features, train_model, evaluate_model

# Fetch data
stock_data = fetch_stock_data('AAPL', '2023-01-01', '2023-12-31')

# Create features
X, y, dates = create_features(stock_data)

# Train model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = train_model(X_train, y_train, model_type='random_forest')
metrics = evaluate_model(model, X_test, y_test)
```

## 📚 Notebooks

1. **01_baseline_model.ipynb** - Baseline model using only technical indicators
   - Data loading and exploration
   - Feature engineering
   - Model training (Random Forest & Logistic Regression)
   - Cross-validation and evaluation

2. **02_sentiment_integration.ipynb** _(To be created)_
   - News sentiment data integration
   - Sentiment feature engineering
   - Performance comparison

3. **03_politician_signals.ipynb** _(To be created)_
   - Politician trading data integration
   - Signal feature engineering
   - Performance comparison

4. **04_combined_model.ipynb** _(To be created)_
   - Full model with all features
   - Feature importance analysis
   - Final performance evaluation

5. **05_economic_backtest.ipynb** _(To be created)_
   - Realistic trading simulation
   - Transaction costs and slippage
   - Risk-adjusted returns analysis

## 🔧 Configuration

### API Keys

To use real news and politician trading data, you'll need API keys:

1. **News API**: Sign up at [newsapi.org](https://newsapi.org/)
2. **Finnhub**: Sign up at [finnhub.io](https://finnhub.io/)
3. **Quiver Quantitative**: Sign up at [quiverquant.com](https://www.quiverquant.com/)

Create a `.env` file in the project root:
```
NEWS_API_KEY=your_news_api_key
FINNHUB_API_KEY=your_finnhub_key
QUIVER_API_KEY=your_quiver_key
```

## 📊 Data Sources

- **Stock Data**: [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`
- **News Sentiment**: NewsAPI, Alpha Vantage, or Finnhub
- **Politician Trades**: Finnhub or Quiver Quantitative

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

## 📈 Model Performance

### Baseline Model (Technical Indicators Only)

Metrics will be displayed after running the baseline notebook. Expected performance:
- Accuracy: ~52-58%
- Precision: ~53-60%
- F1 Score: ~53-58%

## 🛣️ Roadmap

- [x] Project structure setup
- [x] Technical indicator features
- [x] Baseline model (Random Forest & Logistic Regression)
- [ ] News sentiment integration
- [ ] Politician trading signals
- [ ] Combined feature model
- [ ] Economic backtesting framework
- [ ] Model deployment pipeline

## 📝 Research Questions

1. Do sentiment scores improve prediction accuracy over technical indicators alone?
2. Do politician trades have statistically significant predictive power?
3. What is the marginal contribution of each data source?
4. Can the model generate positive risk-adjusted returns after transaction costs?

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It should not be used for actual trading without:
- Extensive additional testing and validation
- Professional financial advice
- Understanding of market risks

Past performance does not guarantee future results.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Technical analysis library: [ta](https://github.com/bukosabino/ta)
- Financial data: [yfinance](https://github.com/ranaroussi/yfinance)
- Sentiment analysis: [NLTK VADER](https://www.nltk.org/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Predicting! 📈**
