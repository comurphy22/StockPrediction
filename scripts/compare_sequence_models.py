"""
Sequence Model Comparison: XGBoost vs LSTM vs GRU

Tests whether sequence models (LSTM, GRU) outperform tabular methods (XGBoost)
on daily stock prediction. This addresses the paper's methodology requirement
to implement sequence models for capturing temporal dependencies.

Hypothesis:
Daily stock data may be too noisy for sequence models to provide benefit.
This experiment will empirically validate or refute this hypothesis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time

# Import configuration
from config import (
    VALIDATION_STOCKS, SEQUENCE_LENGTH, HIDDEN_SIZE, NUM_LAYERS,
    EPOCHS, LEARNING_RATE, check_api_keys, setup_logging
)

# Import data loaders
from data_loader import (
    fetch_stock_data,
    fetch_politician_trades,
    fetch_historical_news_kaggle,
    aggregate_daily_sentiment
)
from feature_engineering import create_features, handle_missing_values

# Import models
from model_xgboost import train_xgboost_model, evaluate_xgboost_model
from model_lstm import train_lstm_model
from model_gru import train_gru_model

# Setup logging
logger = setup_logging('sequence_model_comparison.log')

print("="*80)
print("SEQUENCE MODEL COMPARISON")
print("XGBoost (Tabular) vs LSTM vs GRU (Sequence)")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Check API keys
if not check_api_keys():
    print("\n⚠️  Please configure API keys in .env file")
    print("   Continuing with stock data only (for testing)\n")

# Configuration
STOCKS = VALIDATION_STOCKS[:5]  # Test on 5 stocks initially
YEAR = 2019  # Use 2019 (best news coverage)

print(f"[Configuration]")
print(f"   Test Year: {YEAR}")
print(f"   Stocks: {STOCKS}")
print(f"   Sequence Length: {SEQUENCE_LENGTH} days")
print(f"   LSTM/GRU Hidden Size: {HIDDEN_SIZE}")
print(f"   LSTM/GRU Epochs: {EPOCHS}")
print()

# Results storage
results = []

# Process each stock
for idx, ticker in enumerate(STOCKS, 1):
    print(f"\n{'='*80}")
    print(f"[{idx}/{len(STOCKS)}] Testing {ticker}")
    print(f"{'='*80}")
    
    stock_start_time = time.time()
    
    try:
        # Step 1: Fetch data
        print(f"\n   [1/8] Fetching data...")
        stock_data = fetch_stock_data(ticker, f'{YEAR}-01-01', f'{YEAR}-12-31')
        print(f"      Stock: {len(stock_data)} trading days")
        
        try:
            news_data = fetch_historical_news_kaggle(ticker, f'{YEAR}-01-01', f'{YEAR}-12-31')
            news_sentiment = aggregate_daily_sentiment(news_data)
            print(f"      News: {len(news_data)} articles")
        except:
            news_sentiment = pd.DataFrame()
            print(f"      News: 0 articles (continuing without sentiment)")
        
        try:
            pol_data = fetch_politician_trades(ticker)
            if not pol_data.empty:
                pol_data['date'] = pd.to_datetime(pol_data['date'])
                pol_data = pol_data[
                    (pol_data['date'] >= f'{YEAR}-01-01') & 
                    (pol_data['date'] <= f'{YEAR}-12-31')
                ]
            print(f"      Politician: {len(pol_data)} trades")
        except:
            pol_data = pd.DataFrame()
            print(f"      Politician: 0 trades (continuing without)")
        
        # Step 2: Create features
        print(f"\n   [2/8] Engineering features...")
        X, y, dates = create_features(stock_data, news_sentiment, pol_data, ticker)
        X_clean = handle_missing_values(X, strategy='drop')
        y_clean = y.loc[X_clean.index]
        print(f"      Features: {X_clean.shape}")
        
        # Check if we have enough data
        if len(X_clean) < 100:
            print(f"      ⚠️  Insufficient data (< 100 samples), skipping...")
            continue
        
        # Step 3: Train/Test Split
        print(f"\n   [3/8] Splitting data...")
        split_idx = int(len(X_clean) * 0.8)
        
        X_train = X_clean.iloc[:split_idx].values
        X_test = X_clean.iloc[split_idx:].values
        y_train = y_clean.iloc[:split_idx].values
        y_test = y_clean.iloc[split_idx:].values
        
        print(f"      Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Step 4: Normalize for neural networks
        print(f"\n   [4/8] Normalizing features for LSTM/GRU...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(f"      ✅ Standardized to mean=0, std=1")
        
        # =================================================================
        # Model 1: XGBoost (Tabular Baseline)
        # =================================================================
        
        print(f"\n   [5/8] Training XGBoost (baseline)...")
        
        xgb_start = time.time()
        xgb_model = train_xgboost_model(
            pd.DataFrame(X_train), 
            pd.Series(y_train),
            verbose=False
        )
        xgb_train_metrics = evaluate_xgboost_model(
            xgb_model,
            pd.DataFrame(X_train),
            pd.Series(y_train)
        )
        xgb_test_metrics = evaluate_xgboost_model(
            xgb_model,
            pd.DataFrame(X_test),
            pd.Series(y_test)
        )
        xgb_runtime = time.time() - xgb_start
        
        print(f"      Train: {xgb_train_metrics['accuracy']:.2%}")
        print(f"      Test:  {xgb_test_metrics['accuracy']:.2%}")
        print(f"      Time:  {xgb_runtime:.1f}s")
        
        # =================================================================
        # Model 2: LSTM (Sequence Model)
        # =================================================================
        
        print(f"\n   [6/8] Training LSTM...")
        
        lstm_start = time.time()
        lstm_model, lstm_train_metrics, lstm_test_metrics, lstm_history = train_lstm_model(
            X_train_scaled, y_train,
            X_test_scaled, y_test,
            sequence_length=SEQUENCE_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            verbose=False
        )
        lstm_runtime = time.time() - lstm_start
        
        print(f"      Train: {lstm_train_metrics['accuracy']:.2%}")
        print(f"      Test:  {lstm_test_metrics['accuracy']:.2%}")
        print(f"      Time:  {lstm_runtime:.1f}s")
        
        # =================================================================
        # Model 3: GRU (Alternative Sequence Model)
        # =================================================================
        
        print(f"\n   [7/8] Training GRU...")
        
        gru_start = time.time()
        gru_model, gru_train_metrics, gru_test_metrics, gru_history = train_gru_model(
            X_train_scaled, y_train,
            X_test_scaled, y_test,
            sequence_length=SEQUENCE_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            verbose=False
        )
        gru_runtime = time.time() - gru_start
        
        print(f"      Train: {gru_train_metrics['accuracy']:.2%}")
        print(f"      Test:  {gru_test_metrics['accuracy']:.2%}")
        print(f"      Time:  {gru_runtime:.1f}s")
        
        # =================================================================
        # Step 8: Compare Results
        # =================================================================
        
        print(f"\n   [8/8] Comparison Summary:")
        print(f"      {'Model':<10} {'Train':<8} {'Test':<8} {'F1':<8} {'Time':<8}")
        print(f"      {'-'*50}")
        print(f"      {'XGBoost':<10} {xgb_train_metrics['accuracy']:>7.2%} {xgb_test_metrics['accuracy']:>7.2%} {xgb_test_metrics['f1_score']:>7.2%} {xgb_runtime:>7.1f}s")
        print(f"      {'LSTM':<10} {lstm_train_metrics['accuracy']:>7.2%} {lstm_test_metrics['accuracy']:>7.2%} {lstm_test_metrics['f1']:>7.2%} {lstm_runtime:>7.1f}s")
        print(f"      {'GRU':<10} {gru_train_metrics['accuracy']:>7.2%} {gru_test_metrics['accuracy']:>7.2%} {gru_test_metrics['f1']:>7.2%} {gru_runtime:>7.1f}s")
        
        # Determine winner
        test_accs = {
            'XGBoost': xgb_test_metrics['accuracy'],
            'LSTM': lstm_test_metrics['accuracy'],
            'GRU': gru_test_metrics['accuracy']
        }
        winner = max(test_accs, key=test_accs.get)
        print(f"\n      🏆 Winner: {winner} ({test_accs[winner]:.2%})")
        
        # Store results
        result = {
            'ticker': ticker,
            'n_samples': len(X_clean),
            'n_features': X_clean.shape[1],
            
            # XGBoost
            'xgb_train_acc': xgb_train_metrics['accuracy'],
            'xgb_test_acc': xgb_test_metrics['accuracy'],
            'xgb_f1': xgb_test_metrics['f1_score'],
            'xgb_runtime': xgb_runtime,
            
            # LSTM
            'lstm_train_acc': lstm_train_metrics['accuracy'],
            'lstm_test_acc': lstm_test_metrics['accuracy'],
            'lstm_f1': lstm_test_metrics['f1'],
            'lstm_runtime': lstm_runtime,
            
            # GRU
            'gru_train_acc': gru_train_metrics['accuracy'],
            'gru_test_acc': gru_test_metrics['accuracy'],
            'gru_f1': gru_test_metrics['f1'],
            'gru_runtime': gru_runtime,
            
            # Comparison
            'winner': winner,
            'winner_acc': test_accs[winner]
        }
        results.append(result)
        
        stock_runtime = time.time() - stock_start_time
        print(f"\n   ✅ {ticker} completed in {stock_runtime/60:.1f} minutes")
        
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        continue

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}")

if results:
    df = pd.DataFrame(results)
    output_path = 'results/sequence_model_comparison.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Saved results to: {output_path}")
    
    # ========================================================================
    # AGGREGATE ANALYSIS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS ACROSS ALL STOCKS")
    print(f"{'='*80}")
    
    print(f"\n📊 Average Performance:")
    print(f"   XGBoost:  Test={df['xgb_test_acc'].mean():.2%} (±{df['xgb_test_acc'].std():.2%})")
    print(f"   LSTM:     Test={df['lstm_test_acc'].mean():.2%} (±{df['lstm_test_acc'].std():.2%})")
    print(f"   GRU:      Test={df['gru_test_acc'].mean():.2%} (±{df['gru_test_acc'].std():.2%})")
    
    print(f"\n📊 Average F1 Score:")
    print(f"   XGBoost:  {df['xgb_f1'].mean():.2%}")
    print(f"   LSTM:     {df['lstm_f1'].mean():.2%}")
    print(f"   GRU:      {df['gru_f1'].mean():.2%}")
    
    print(f"\n⏱️  Average Training Time:")
    print(f"   XGBoost:  {df['xgb_runtime'].mean():.1f}s")
    print(f"   LSTM:     {df['lstm_runtime'].mean():.1f}s")
    print(f"   GRU:      {df['gru_runtime'].mean():.1f}s")
    
    print(f"\n🏆 Winner Count:")
    winner_counts = df['winner'].value_counts()
    for model, count in winner_counts.items():
        print(f"   {model}: {count}/{len(df)} stocks")
    
    # Statistical comparison
    xgb_better = (df['xgb_test_acc'] > df['lstm_test_acc']).sum()
    lstm_better = (df['lstm_test_acc'] > df['xgb_test_acc']).sum()
    
    print(f"\n📈 XGBoost vs LSTM:")
    print(f"   XGBoost better: {xgb_better}/{len(df)} stocks")
    print(f"   LSTM better:    {lstm_better}/{len(df)} stocks")
    
    # Overfitting comparison
    xgb_overfit = (df['xgb_train_acc'] - df['xgb_test_acc']).mean()
    lstm_overfit = (df['lstm_train_acc'] - df['lstm_test_acc']).mean()
    gru_overfit = (df['gru_train_acc'] - df['gru_test_acc']).mean()
    
    print(f"\n📉 Average Overfitting Gap:")
    print(f"   XGBoost:  {xgb_overfit:.2%}")
    print(f"   LSTM:     {lstm_overfit:.2%}")
    print(f"   GRU:      {gru_overfit:.2%}")

else:
    print("❌ No results to save")

# ============================================================================
# FINAL CONCLUSIONS
# ============================================================================

print(f"\n{'='*80}")
print("CONCLUSIONS FOR PAPER")
print(f"{'='*80}")

if results and len(results) >= 3:
    avg_xgb = df['xgb_test_acc'].mean()
    avg_lstm = df['lstm_test_acc'].mean()
    avg_gru = df['gru_test_acc'].mean()
    
    best_model = max([('XGBoost', avg_xgb), ('LSTM', avg_lstm), ('GRU', avg_gru)], 
                     key=lambda x: x[1])
    
    improvement = best_model[1] - min(avg_xgb, avg_lstm, avg_gru)
    
    print(f"\n1️⃣  Overall Winner: {best_model[0]} ({best_model[1]:.2%} average)")
    print(f"   Improvement over worst model: +{improvement:.2%}")
    
    if avg_xgb > max(avg_lstm, avg_gru):
        print(f"\n2️⃣  Finding: Tabular method (XGBoost) outperforms sequence models")
        print(f"   Interpretation: Daily stock data too noisy for temporal patterns")
        print(f"   Paper implication: Negative result is scientifically valuable")
    else:
        print(f"\n2️⃣  Finding: Sequence models provide benefit over tabular methods")
        print(f"   Interpretation: Temporal dependencies captured successfully")
        print(f"   Paper implication: LSTM/GRU justified for stock prediction")
    
    print(f"\n3️⃣  Overfitting:")
    if xgb_overfit < min(lstm_overfit, gru_overfit):
        print(f"   XGBoost shows less overfitting than sequence models")
        print(f"   May be better suited for limited training data")
    else:
        print(f"   Sequence models generalize as well or better than XGBoost")
    
    print(f"\n4️⃣  Computational Cost:")
    time_ratio = df['lstm_runtime'].mean() / df['xgb_runtime'].mean()
    print(f"   LSTM takes {time_ratio:.1f}x longer than XGBoost")
    print(f"   Trade-off: accuracy vs training time")

print(f"\n📝 Next Steps:")
print(f"   1. Review results/sequence_model_comparison.csv")
print(f"   2. Create visualizations (accuracy comparison, training curves)")
print(f"   3. Document findings in paper's Model Comparison section")
print(f"   4. Consider ensemble: XGBoost + LSTM voting")

print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}\n")

