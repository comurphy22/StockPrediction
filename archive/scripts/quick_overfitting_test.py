"""
Simplified Overfitting Fix - Quick Hyperparameter Test
Tests regularization on BABA, NFLX, TSLA with pre-loaded datasets
"""

import sys
import os
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

from data_loader import fetch_stock_data, fetch_politician_trades
from data_loader_optimized import fetch_news_with_keywords, aggregate_daily_sentiment
from feature_engineering import create_features, handle_missing_values

print("="*70)
print("QUICK OVERFITTING TEST")
print("="*70)

YEAR = 2019
TRAIN_SPLIT = 0.8
TEST_STOCKS = ['BABA', 'NFLX', 'TSLA']

# Load feature rankings
rankings = pd.read_csv('Results/feature_importance_rankings.csv')
top_features = rankings.head(25)['feature'].tolist()

# Configurations to test
configs = {
    'Baseline': {'max_depth': 6, 'min_child_weight': 1, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1},
    'Reduced Depth': {'max_depth': 3, 'min_child_weight': 1, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1},
    'Regularized': {'max_depth': 4, 'min_child_weight': 5, 'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
    'Conservative': {'max_depth': 3, 'min_child_weight': 10, 'gamma': 0.2, 'reg_alpha': 0.5, 'reg_lambda': 2.0},
}

results = []

for ticker in TEST_STOCKS:
    print(f"\n{'='*70}")
    print(f"Loading data for {ticker}...")
    print(f"{'='*70}")
    
    # Load all data once
    stock_data = fetch_stock_data(ticker, f'{YEAR}-01-01', f'{YEAR}-12-31')
    news_data = fetch_news_with_keywords(ticker, f'{YEAR}-01-01', f'{YEAR}-12-31')
    news_sentiment = aggregate_daily_sentiment(news_data)
    pol_data = fetch_politician_trades(ticker)
    
    # Create features
    X_all, y_all, dates = create_features(stock_data, news_sentiment, pol_data)
    X_all = handle_missing_values(X_all, strategy='drop')
    y_all = y_all.loc[X_all.index]
    
    available_features = [f for f in top_features if f in X_all.columns]
    X = X_all[available_features]
    y = y_all
    
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"   Samples: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")
    
    for config_name, params in configs.items():
        print(f"\n  Testing {config_name}... ", end="", flush=True)
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            **params
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        gap = train_acc - test_acc
        
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'ticker': ticker,
            'config': config_name,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'gap': gap,
            'f1': f1
        })
        
        print(f"Train: {train_acc:.1%} | Test: {test_acc:.1%} | Gap: {gap:+.1%} | F1: {f1:.1%}")

# Summary
print("\n" + "="*70)
print("SUMMARY BY CONFIGURATION")
print("="*70)

df = pd.DataFrame(results)
summary = df.groupby('config').agg({
    'train_acc': 'mean',
    'test_acc': 'mean',
    'gap': 'mean',
    'f1': 'mean'
}).round(4)

summary['test_pct'] = (summary['test_acc'] * 100).round(2)
summary['gap_pct'] = (summary['gap'] * 100).round(2)
summary['f1_pct'] = (summary['f1'] * 100).round(2)

print(f"\n{'Config':<15} {'Test Acc':<10} {'Overfit Gap':<12} {'F1 Score':<10}")
print("-"*70)
for config in configs.keys():
    row = summary.loc[config]
    print(f"{config:<15} {row['test_pct']:>8.2f}%  {row['gap_pct']:>10.2f}%  {row['f1_pct']:>8.2f}%")

# Find best
best_test = summary['test_acc'].idxmax()
best_gap = summary['gap'].idxmin()

print("\n🏆 BEST:")
print(f"  Highest Test Acc: {best_test}")
print(f"  Lowest Overfit Gap: {best_gap}")

# Save
df.to_csv('Results/overfitting_quick_test.csv', index=False)
print("\n✅ Saved: Results/overfitting_quick_test.csv")
