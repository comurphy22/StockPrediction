"""
Multi-Year Validation with Feature Selection

Tests XGBoost using ONLY the top 20 most important features
to improve sample/feature ratio from 1.6:1 to 5:1+
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
import time

from config import VALIDATION_YEARS, VALIDATION_STOCKS, TRAIN_TEST_SPLIT, TOP_FEATURES
from data_loader import (
    fetch_stock_data,
    fetch_politician_trades,
    fetch_historical_news_kaggle,
    aggregate_daily_sentiment
)
from feature_engineering import create_features, handle_missing_values
from model_xgboost import train_xgboost_model, evaluate_xgboost_model

# Use only top 20 features (from config.py TOP_FEATURES)
SELECTED_FEATURES = TOP_FEATURES[:20]

print("="*80)
print("MULTI-YEAR VALIDATION WITH FEATURE SELECTION")
print("="*80)
print(f"Using top {len(SELECTED_FEATURES)} features only:")
print(f"  {', '.join(SELECTED_FEATURES[:5])}...")
print(f"\nExpected improvement:")
print(f"  - Sample/feature ratio: 1.6:1 → 5:1+")
print(f"  - Reduced overfitting from feature noise")
print(f"  - More stable cross-validation\n")

results = []

for year in VALIDATION_YEARS[:2]:  # Test 2018-2019 first
    print(f"\n{'='*80}")
    print(f"YEAR {year}")
    print(f"{'='*80}\n")
    
    for ticker in VALIDATION_STOCKS:
        print(f"{ticker} - {year}")
        print("-"*80)
        
        try:
            # Fetch data
            print(f"   [1/7] Fetching stock data...", end=" ", flush=True)
            stock_data = fetch_stock_data(ticker, f'{year}-01-01', f'{year}-12-31')
            print(f"✅ {len(stock_data)} days")
            
            print(f"   [2/7] Loading news sentiment...", end=" ", flush=True)
            news_data = fetch_historical_news_kaggle(ticker, f'{year}-01-01', f'{year}-12-31')
            news_sentiment = aggregate_daily_sentiment(news_data)
            print(f"✅ {len(news_data)} articles")
            
            print(f"   [3/7] Loading politician trades...", end=" ", flush=True)
            trades_data = fetch_politician_trades(ticker)
            print(f"✅ {len(trades_data)} trades")
            
            print(f"   [4/7] Engineering features...", end=" ", flush=True)
            X, y, dates = create_features(stock_data, news_sentiment, trades_data)
            print(f"✅ {len(X.columns)} features created")
            
            print(f"   [5/7] Selecting top 20 features...", end=" ", flush=True)
            # Select only features that exist in the dataset
            available_features = [f for f in SELECTED_FEATURES if f in X.columns]
            X = X[available_features]
            print(f"✅ Using {len(available_features)} features")
            
            print(f"   [6/7] Cleaning data...", end=" ", flush=True)
            X_clean = handle_missing_values(X, strategy='drop')
            y_clean = y.loc[X_clean.index]
            # Reset index to avoid misalignment during train-test split
            X_clean = X_clean.reset_index(drop=True)
            y_clean = y_clean.reset_index(drop=True)
            X, y = X_clean, y_clean
            print(f"✅ {len(X)} samples (ratio: {len(X)/len(available_features):.1f}:1)")
            
            # Split
            split_idx = int(len(X) * TRAIN_TEST_SPLIT)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"   [7/7] Training XGBoost...", end=" ", flush=True)
            model = train_xgboost_model(X_train, y_train, verbose=False)
            
            train_metrics = evaluate_xgboost_model(model, X_train, y_train)
            test_metrics = evaluate_xgboost_model(model, X_test, y_test)
            
            gap = train_metrics['accuracy'] - test_metrics['accuracy']
            
            results.append({
                'year': year,
                'ticker': ticker,
                'n_samples': len(X),
                'n_features': len(available_features),
                'sample_feature_ratio': len(X) / len(available_features),
                'train_acc': train_metrics['accuracy'],
                'test_acc': test_metrics['accuracy'],
                'overfit_gap': gap,
                'f1': test_metrics['f1_score'],
                'news_articles': len(news_data)
            })
            
            print(f"Train: {train_metrics['accuracy']:.1%}, Test: {test_metrics['accuracy']:.1%}, Gap: {gap:.1%}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

# Save and analyze results
results_df = pd.DataFrame(results)
output_path = 'results/feature_selection_validation_results.csv'
results_df.to_csv(output_path, index=False)

print("\n" + "="*80)
print("FEATURE SELECTION RESULTS COMPARISON")
print("="*80 + "\n")

# Compare to previous results
print("Previous (61 features):")
print("  2018: 48.84% test acc, 51.16% overfitting gap")
print("  2019: 54.77% test acc, 45.04% overfitting gap\n")

print(f"Current ({len(available_features)} features):")
for year in [2018, 2019]:
    year_results = results_df[results_df['year'] == year]
    if len(year_results) > 0:
        avg_test = year_results['test_acc'].mean()
        avg_gap = year_results['overfit_gap'].mean()
        avg_ratio = year_results['sample_feature_ratio'].mean()
        print(f"  {year}: {avg_test:.2%} test acc, {avg_gap:.2%} overfitting gap, {avg_ratio:.1f}:1 ratio")

print(f"\n✅ Results saved to: {output_path}")

