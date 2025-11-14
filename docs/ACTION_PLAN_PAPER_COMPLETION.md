# Action Plan: Complete Paper Implementation

**Goal:** Address critical gaps to enable top-tier publication  
**Timeline:** 2-5 days depending on scope  
**Status:** Ready to execute

---

## Quick Start: Choose Your Path

### Path A: Fast Track to Conference Paper (2-3 days)
**Implements:** LSTM/GRU + Multi-year testing + Ablation studies  
**Result:** 87.5% alignment, suitable for ACM ICAIF, IEEE conferences  
**Sections:** 1-3 below

### Path B: Strong Top-Tier Paper (3-5 days)
**Implements:** Path A + Backtesting + ROC-AUC + 3-year dataset  
**Result:** 95% alignment, suitable for KDD, ICML/NeurIPS workshops  
**Sections:** 1-6 below

### Path C: Maximum Impact (5-8 days)
**Implements:** Path B + Committee weighting + Walk-forward validation  
**Result:** 100% alignment, competitive for top venues  
**Sections:** 1-8 below

**Recommended: Path B** (best ROI for time invested)

---

## Section 1: Implement LSTM/GRU Models (1-2 days)

### Objective
Add sequence models promised in paper methodology

### Implementation Steps

#### Step 1.1: Create LSTM Model Class (2 hours)

**File:** `src/model_lstm.py`

```python
"""
LSTM model for stock prediction with temporal dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class StockLSTM(nn.Module):
    """
    LSTM model for binary stock movement prediction.
    
    Architecture:
    - Input: [batch_size, sequence_length, n_features]
    - LSTM layers with dropout
    - Fully connected output layer
    - Sigmoid activation for binary classification
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(StockLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)
        
        # Take last time step
        last_output = lstm_out[:, -1, :]
        
        # Fully connected + sigmoid
        out = self.fc(last_output)
        out = self.sigmoid(out)
        
        return out

def prepare_sequences(X, y, sequence_length=10):
    """
    Convert tabular data to sequences for LSTM.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)
    sequence_length : int
        Number of time steps in each sequence
    
    Returns:
    --------
    X_seq : np.ndarray
        Sequences (n_sequences, sequence_length, n_features)
    y_seq : np.ndarray
        Targets for sequences (n_sequences,)
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    
    return np.array(X_seq), np.array(y_seq)

def train_lstm_model(X_train, y_train, X_test, y_test, 
                     sequence_length=10, hidden_size=64, 
                     num_layers=2, epochs=50, learning_rate=0.001):
    """
    Train LSTM model for stock prediction.
    
    Returns:
    --------
    model : StockLSTM
        Trained model
    train_metrics : dict
        Training accuracy
    test_metrics : dict
        Test metrics (accuracy, precision, recall, f1)
    """
    # Prepare sequences
    X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = prepare_sequences(X_test, y_test, sequence_length)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_seq)
    y_test_tensor = torch.FloatTensor(y_test_seq).unsqueeze(1)
    
    # Initialize model
    input_size = X_train_seq.shape[2]
    model = StockLSTM(input_size, hidden_size, num_layers)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        # Training accuracy
        train_pred = (model(X_train_tensor) > 0.5).float()
        train_acc = (train_pred == y_train_tensor).float().mean().item()
        
        # Test predictions
        test_pred = (model(X_test_tensor) > 0.5).float()
        test_acc = (test_pred == y_test_tensor).float().mean().item()
        
        # Additional metrics
        y_pred = test_pred.squeeze().numpy()
        y_true = y_test_tensor.squeeze().numpy()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
    
    train_metrics = {'accuracy': train_acc}
    test_metrics = {
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return model, train_metrics, test_metrics
```

**Time estimate:** 2 hours

#### Step 1.2: Create GRU Model Class (1 hour)

**File:** `src/model_gru.py`

- Copy LSTM code and replace `nn.LSTM` with `nn.GRU`
- Adjust docstrings
- Keep same architecture otherwise

**Time estimate:** 1 hour

#### Step 1.3: Create Comparison Script (3 hours)

**File:** `scripts/compare_sequence_models.py`

```python
"""
Compare XGBoost, LSTM, and GRU on 10 stocks.
Tests if sequence models provide value over tabular approach.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_loader import fetch_stock_data, fetch_politician_trades, fetch_historical_news_kaggle, aggregate_daily_sentiment
from feature_engineering import create_features, handle_missing_values
from model_xgboost import train_xgboost_model, evaluate_xgboost_model
from model_lstm import train_lstm_model
from model_gru import train_gru_model

STOCKS = ['NFLX', 'NVDA', 'BABA', 'QCOM', 'MU', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
YEAR = 2019
SEQUENCE_LENGTH = 10  # Use 10 days of history

results = []

for ticker in STOCKS:
    print(f"\n{'='*70}")
    print(f"Testing {ticker}")
    print(f"{'='*70}")
    
    # Fetch data
    stock_data = fetch_stock_data(ticker, f'{YEAR}-01-01', f'{YEAR}-12-31')
    news_data = fetch_historical_news_kaggle(ticker, f'{YEAR}-01-01', f'{YEAR}-12-31')
    politician_data = fetch_politician_trades(ticker, f'{YEAR}-01-01', f'{YEAR}-12-31')
    
    news_sentiment = aggregate_daily_sentiment(news_data)
    
    # Create features
    X, y, dates = create_features(stock_data, news_sentiment, politician_data, ticker)
    X_clean = handle_missing_values(X, strategy='drop')
    y_clean = y.loc[X_clean.index]
    
    # Train/test split (80/20)
    split_idx = int(len(X_clean) * 0.8)
    X_train, X_test = X_clean.values[:split_idx], X_clean.values[split_idx:]
    y_train, y_test = y_clean.values[:split_idx], y_clean.values[split_idx:]
    
    # Normalize for neural networks
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test XGBoost (baseline)
    print("\n[1/3] Testing XGBoost...")
    xgb_model, xgb_train_metrics, xgb_test_metrics = train_xgboost_model(X_train, y_train)
    xgb_test_metrics = evaluate_xgboost_model(xgb_model, X_test, y_test)
    
    # Test LSTM
    print("\n[2/3] Testing LSTM...")
    lstm_model, lstm_train_metrics, lstm_test_metrics = train_lstm_model(
        X_train_scaled, y_train, X_test_scaled, y_test,
        sequence_length=SEQUENCE_LENGTH, epochs=50
    )
    
    # Test GRU
    print("\n[3/3] Testing GRU...")
    gru_model, gru_train_metrics, gru_test_metrics = train_gru_model(
        X_train_scaled, y_train, X_test_scaled, y_test,
        sequence_length=SEQUENCE_LENGTH, epochs=50
    )
    
    # Store results
    results.append({
        'ticker': ticker,
        'xgb_train_acc': xgb_train_metrics['accuracy'],
        'xgb_test_acc': xgb_test_metrics['accuracy'],
        'lstm_train_acc': lstm_train_metrics['accuracy'],
        'lstm_test_acc': lstm_test_metrics['accuracy'],
        'gru_train_acc': gru_train_metrics['accuracy'],
        'gru_test_acc': gru_test_metrics['accuracy'],
        'xgb_f1': xgb_test_metrics['f1'],
        'lstm_f1': lstm_test_metrics['f1'],
        'gru_f1': gru_test_metrics['f1']
    })
    
    print(f"\nResults for {ticker}:")
    print(f"  XGBoost:  Train={xgb_train_metrics['accuracy']:.2%}, Test={xgb_test_metrics['accuracy']:.2%}")
    print(f"  LSTM:     Train={lstm_train_metrics['accuracy']:.2%}, Test={lstm_test_metrics['accuracy']:.2%}")
    print(f"  GRU:      Train={gru_train_metrics['accuracy']:.2%}, Test={gru_test_metrics['accuracy']:.2%}")

# Save results
df = pd.DataFrame(results)
df.to_csv('results/model_comparison_sequence.csv', index=False)

# Summary
print("\n" + "="*70)
print("SUMMARY ACROSS ALL STOCKS")
print("="*70)
print(f"Average Test Accuracy:")
print(f"  XGBoost: {df['xgb_test_acc'].mean():.2%}")
print(f"  LSTM:    {df['lstm_test_acc'].mean():.2%}")
print(f"  GRU:     {df['gru_test_acc'].mean():.2%}")
print(f"\nAverage F1 Score:")
print(f"  XGBoost: {df['xgb_f1'].mean():.2%}")
print(f"  LSTM:    {df['lstm_f1'].mean():.2%}")
print(f"  GRU:     {df['gru_f1'].mean():.2%}")
```

**Time estimate:** 3 hours

#### Step 1.4: Run and Document (2 hours)

1. Install PyTorch: `pip install torch`
2. Run comparison script: `python scripts/compare_sequence_models.py`
3. Create `docs/SEQUENCE_MODELS_RESULTS.md` documenting findings
4. Update README with sequence model results

**Time estimate:** 2 hours

**Total Section 1 Time: 8 hours (1 day)**

**Expected outcome:** Even if LSTM/GRU don't outperform XGBoost, this is publishable. Daily stock data may be too noisy for sequence benefits—this is a valuable finding.

---

## Section 2: Multi-Year Testing (4-6 hours)

### Objective
Test model generalization across different market regimes

### Implementation Steps

#### Step 2.1: Create Multi-Year Validation Script (2 hours)

**File:** `scripts/validate_multiyear.py`

```python
"""
Test model performance across multiple years (2019, 2020, 2021).
Evaluates generalization and regime stability.
"""

import pandas as pd
from datetime import datetime

# Configuration
YEARS = [2019, 2020, 2021]
STOCKS = ['NFLX', 'NVDA', 'BABA', 'QCOM', 'MU', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']

results = []

for year in YEARS:
    print(f"\n{'='*70}")
    print(f"TESTING YEAR: {year}")
    print(f"{'='*70}")
    
    for ticker in STOCKS:
        # Fetch data for this year
        stock_data = fetch_stock_data(ticker, f'{year}-01-01', f'{year}-12-31')
        news_data = fetch_historical_news_kaggle(ticker, f'{year}-01-01', f'{year}-12-31')
        politician_data = fetch_politician_trades(ticker, f'{year}-01-01', f'{year}-12-31')
        
        # Standard pipeline
        news_sentiment = aggregate_daily_sentiment(news_data)
        X, y, dates = create_features(stock_data, news_sentiment, politician_data, ticker)
        X_clean = handle_missing_values(X, strategy='drop')
        y_clean = y.loc[X_clean.index]
        
        # Train/test split
        split_idx = int(len(X_clean) * 0.8)
        X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
        y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
        
        # Train XGBoost
        model, train_metrics, test_metrics = train_xgboost_model(X_train, y_train)
        test_metrics = evaluate_xgboost_model(model, X_test, y_test)
        
        results.append({
            'year': year,
            'ticker': ticker,
            'n_samples': len(X_clean),
            'train_acc': train_metrics['accuracy'],
            'test_acc': test_metrics['accuracy'],
            'overfit_gap': train_metrics['accuracy'] - test_metrics['accuracy'],
            'f1': test_metrics['f1'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall']
        })

# Save results
df = pd.DataFrame(results)
df.to_csv('results/multiyear_validation_results.csv', index=False)

# Analysis by year
print("\n" + "="*70)
print("PERFORMANCE BY YEAR")
print("="*70)
for year in YEARS:
    year_data = df[df['year'] == year]
    print(f"\n{year}:")
    print(f"  Avg Test Accuracy: {year_data['test_acc'].mean():.2%}")
    print(f"  Avg F1 Score: {year_data['f1'].mean():.2%}")
    print(f"  Overfitting Gap: {year_data['overfit_gap'].mean():.2%}")
    print(f"  Best Stock: {year_data.loc[year_data['test_acc'].idxmax()]['ticker']} ({year_data['test_acc'].max():.2%})")
    print(f"  Worst Stock: {year_data.loc[year_data['test_acc'].idxmin()]['ticker']} ({year_data['test_acc'].min():.2%})")
```

#### Step 2.2: Run and Analyze (2 hours)

1. Run script: `python scripts/validate_multiyear.py`
2. Analyze variance across years
3. Document in `docs/MULTIYEAR_RESULTS.md`

**Expected insights:**
- 2019: Bull market (expect higher accuracy)
- 2020: COVID volatility (expect lower accuracy or higher variance)
- 2021: Recovery (intermediate performance)

#### Step 2.3: Create Visualization (1 hour)

**File:** `scripts/visualize_multiyear.py`

Create plots:
- Line chart: Average accuracy by year
- Box plots: Accuracy distribution per year
- Heatmap: Per-stock, per-year performance

**Time estimate:** 1 hour

**Total Section 2 Time: 5 hours**

---

## Section 3: Systematic Ablation Studies (2-3 hours)

### Objective
Isolate marginal contribution of each feature category

### Implementation Steps

#### Step 3.1: Create Ablation Script (2 hours)

**File:** `scripts/feature_ablation.py`

```python
"""
Systematic ablation study to measure marginal contribution of each feature category.

Tests 4 configurations:
1. Technical only
2. Technical + Sentiment
3. Technical + Politician
4. Technical + Sentiment + Politician (full model)
"""

import pandas as pd

# Feature categories (from feature_importance_rankings.csv)
TECHNICAL_FEATURES = [
    'HL_spread', 'Price_change', 'SMA_50', 'Volume_change', 'MACD_diff',
    'SMA_10_20_cross', 'SMA_20', 'RSI', 'MACD', 'SMA_10', 'SMA_20_50_cross',
    'MACD_signal', 'amount_last_60d', 'trades_last_60d'
]

SENTIMENT_FEATURES = [
    'avg_sentiment_compound', 'avg_sentiment_positive', 
    'avg_sentiment_negative', 'news_count'
]

POLITICIAN_FEATURES = [
    'days_since_last_trade', 'net_flow_last_60d', 'net_flow_last_90d',
    'dollar_momentum_30d', 'trade_momentum_30d', 'net_flow_last_30d',
    'amount_last_30d', 'trades_last_90d', 'amount_last_90d', 'trades_last_30d'
]

CONFIGURATIONS = {
    'technical_only': TECHNICAL_FEATURES,
    'tech_sentiment': TECHNICAL_FEATURES + SENTIMENT_FEATURES,
    'tech_politician': TECHNICAL_FEATURES + POLITICIAN_FEATURES,
    'full_model': TECHNICAL_FEATURES + SENTIMENT_FEATURES + POLITICIAN_FEATURES
}

STOCKS = ['NFLX', 'NVDA', 'BABA', 'QCOM', 'MU', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
YEAR = 2019

results = []

for config_name, feature_list in CONFIGURATIONS.items():
    print(f"\n{'='*70}")
    print(f"TESTING CONFIGURATION: {config_name.upper()}")
    print(f"Features: {len(feature_list)}")
    print(f"{'='*70}")
    
    for ticker in STOCKS:
        # Fetch data
        stock_data = fetch_stock_data(ticker, f'{YEAR}-01-01', f'{YEAR}-12-31')
        news_data = fetch_historical_news_kaggle(ticker, f'{YEAR}-01-01', f'{YEAR}-12-31')
        politician_data = fetch_politician_trades(ticker, f'{YEAR}-01-01', f'{YEAR}-12-31')
        
        # Create features
        news_sentiment = aggregate_daily_sentiment(news_data)
        X, y, dates = create_features(stock_data, news_sentiment, politician_data, ticker)
        
        # Select only features in this configuration
        available_features = [f for f in feature_list if f in X.columns]
        X_selected = X[available_features]
        
        # Handle missing values
        X_clean = handle_missing_values(X_selected, strategy='drop')
        y_clean = y.loc[X_clean.index]
        
        # Train/test split
        split_idx = int(len(X_clean) * 0.8)
        X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
        y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
        
        # Train model
        model, train_metrics, test_metrics = train_xgboost_model(X_train, y_train)
        test_metrics = evaluate_xgboost_model(model, X_test, y_test)
        
        results.append({
            'configuration': config_name,
            'ticker': ticker,
            'n_features': len(available_features),
            'test_acc': test_metrics['accuracy'],
            'f1': test_metrics['f1']
        })
        
        print(f"  {ticker}: {test_metrics['accuracy']:.2%} ({len(available_features)} features)")

# Save results
df = pd.DataFrame(results)
df.to_csv('results/feature_ablation_results.csv', index=False)

# Marginal contributions
print("\n" + "="*70)
print("MARGINAL CONTRIBUTIONS")
print("="*70)

pivot = df.pivot_table(values='test_acc', index='ticker', columns='configuration')
pivot['sentiment_contribution'] = pivot['tech_sentiment'] - pivot['technical_only']
pivot['politician_contribution'] = pivot['tech_politician'] - pivot['technical_only']
pivot['full_vs_tech_sent'] = pivot['full_model'] - pivot['tech_sentiment']

print("\nAverage Marginal Contributions:")
print(f"  Sentiment adds: {pivot['sentiment_contribution'].mean():.2%}")
print(f"  Politician adds: {pivot['politician_contribution'].mean():.2%}")
print(f"  Adding politician to tech+sentiment: {pivot['full_vs_tech_sent'].mean():.2%}")
```

#### Step 3.2: Run and Document (1 hour)

1. Run: `python scripts/feature_ablation.py`
2. Create `docs/ABLATION_STUDY_RESULTS.md`
3. Highlight marginal contributions in README

**Total Section 3 Time: 3 hours**

**Expected outcome:** Quantify that politician features add ~2-3% accuracy on average, sentiment adds ~3-4%, and full model is best. This directly answers the paper's core research question.

---

## Section 4: Basic Economic Backtesting (1 day)

### Objective
Test economic viability with transaction costs

### Implementation

**File:** `scripts/economic_backtest.py`

```python
"""
Economic backtesting with transaction costs.
Tests if 56% accuracy translates to profitable trading.
"""

def backtest_strategy(predictions, actual_returns, transaction_cost=0.003):
    """
    Backtest simple long/short strategy.
    
    Parameters:
    -----------
    predictions : array
        Binary predictions (1 = buy, 0 = sell/short)
    actual_returns : array
        Actual next-day returns
    transaction_cost : float
        Round-trip transaction cost (default 0.3%)
    
    Returns:
    --------
    dict: Backtest metrics (cumulative return, Sharpe ratio, win rate)
    """
    # Trading signals
    positions = np.where(predictions == 1, 1, -1)  # 1 = long, -1 = short
    
    # Strategy returns (subtract transaction costs for each trade)
    strategy_returns = positions * actual_returns - transaction_cost
    
    # Cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    total_return = cumulative_returns[-1] - 1
    
    # Sharpe ratio (annualized)
    sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
    
    # Win rate
    win_rate = (strategy_returns > 0).mean()
    
    # Max drawdown
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'n_trades': len(predictions)
    }

# Test across transaction costs
TRANSACTION_COSTS = [0.001, 0.003, 0.005]  # 0.1%, 0.3%, 0.5%

results = []

for cost in TRANSACTION_COSTS:
    print(f"\nTransaction Cost: {cost:.1%}")
    
    for ticker in STOCKS:
        # ... (fetch data, create features, train model)
        
        # Get predictions and actual returns
        y_pred = model.predict(X_test)
        actual_returns = stock_data['Close'].pct_change().shift(-1)[X_test.index]
        
        # Backtest
        backtest_results = backtest_strategy(y_pred, actual_returns.values, cost)
        
        results.append({
            'ticker': ticker,
            'transaction_cost': cost,
            **backtest_results
        })

# Save and analyze
df = pd.DataFrame(results)
df.to_csv('results/economic_backtest_results.csv', index=False)

# Summary
for cost in TRANSACTION_COSTS:
    cost_data = df[df['transaction_cost'] == cost]
    print(f"\nTransaction Cost {cost:.1%}:")
    print(f"  Avg Total Return: {cost_data['total_return'].mean():.2%}")
    print(f"  Avg Sharpe Ratio: {cost_data['sharpe_ratio'].mean():.2f}")
    print(f"  Profitable Stocks: {(cost_data['total_return'] > 0).sum()}/10")
```

**Time estimate:** 6-8 hours

**Total Section 4 Time: 1 day**

---

## Section 5: Add ROC-AUC Metric (30 minutes)

### Implementation

**File:** Modify `src/model_xgboost.py`

```python
from sklearn.metrics import roc_auc_score, roc_curve

def evaluate_xgboost_model(model, X_test, y_test):
    """
    Enhanced evaluation with ROC-AUC.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    
    # Existing metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    
    # Add ROC-AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
```

**Time estimate:** 30 minutes

---

## Section 6: Extend to 3-Year Dataset (2-3 hours)

### Implementation

Modify `validate_mvp.py` to accept year parameter:

```python
# Configuration
YEARS = [2018, 2019, 2020]  # Three years

for year in YEARS:
    print(f"\nValidating year: {year}")
    # ... existing code with year variable
```

**Time estimate:** 2 hours

---

## Sections 7-8: Optional Enhancements

See detailed implementation in `PAPER_ALIGNMENT_MATRIX.md` Phase 3

---

## Execution Checklist

### Phase 1: Critical Fixes (2-3 days)

- [ ] Section 1: Implement LSTM/GRU models
  - [ ] Create `src/model_lstm.py`
  - [ ] Create `src/model_gru.py`
  - [ ] Create `scripts/compare_sequence_models.py`
  - [ ] Run experiments and document results
  - [ ] Update README with findings

- [ ] Section 2: Multi-year testing
  - [ ] Create `scripts/validate_multiyear.py`
  - [ ] Run for 2019, 2020, 2021
  - [ ] Create `docs/MULTIYEAR_RESULTS.md`
  - [ ] Create visualizations

- [ ] Section 3: Feature ablation
  - [ ] Create `scripts/feature_ablation.py`
  - [ ] Test 4 configurations
  - [ ] Document marginal contributions
  - [ ] Create `docs/ABLATION_STUDY_RESULTS.md`

### Phase 2: Strengthen Claims (1-2 days)

- [ ] Section 4: Economic backtesting
  - [ ] Create `scripts/economic_backtest.py`
  - [ ] Test multiple transaction costs
  - [ ] Calculate Sharpe ratios
  - [ ] Create cumulative return plots

- [ ] Section 5: Add ROC-AUC
  - [ ] Modify `src/model_xgboost.py`
  - [ ] Re-run validation scripts
  - [ ] Update all results CSVs

- [ ] Section 6: 3-year dataset
  - [ ] Modify validation scripts
  - [ ] Fetch 2018, 2019, 2020 data
  - [ ] Update documentation

### Final Steps

- [ ] Update `README.md` with complete results
- [ ] Create paper-ready visualizations
- [ ] Write `docs/PAPER_RESULTS_SUMMARY.md`
- [ ] Review all documentation for consistency
- [ ] Prepare code repository for publication

---

## Expected Time Investment

| Path | Days | Sections | Alignment | Publication Target |
|------|------|----------|-----------|-------------------|
| Path A | 2-3 | 1-3 | 87.5% | Mid-tier conferences |
| Path B | 3-5 | 1-6 | 95% | Top-tier conferences |
| Path C | 5-8 | 1-8 | 100% | Journal/elite venues |

**Recommended: Path B** (best value for time)

---

## Getting Started Now

```bash
# 1. Create branch for paper completion
git checkout -b paper-completion

# 2. Install additional dependencies
pip install torch scikit-learn matplotlib seaborn

# 3. Start with Section 1 (LSTM/GRU)
# Create src/model_lstm.py using template above

# 4. Track progress
# Check off items in execution checklist above
```

---

## Questions or Issues?

Refer to:
- `PAPER_GOALS_EFFICACY_ANALYSIS.md` - Detailed analysis
- `EXECUTIVE_SUMMARY_PAPER_EFFICACY.md` - Quick overview
- `PAPER_ALIGNMENT_MATRIX.md` - Gap identification

---

**Last Updated:** November 13, 2025  
**Status:** Ready to execute  
**Estimated completion:** 2-5 days depending on chosen path

