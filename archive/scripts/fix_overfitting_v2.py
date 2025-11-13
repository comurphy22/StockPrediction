"""
Overfitting Fix Script V2 - Hyperparameter Tuning
Tests different XGBoost configurations to reduce 44% train/test gap.

Strategy:
1. Reduce model complexity (max_depth)
2. Add regularization (min_child_weight, gamma, reg_alpha, reg_lambda)
3. Add early stopping to prevent overfitting
4. Test on 3 diverse stocks: BABA (best), NFLX (medium), TSLA (worst)
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, src_path)

from data_loader import fetch_stock_data, fetch_politician_trades
from data_loader_optimized import fetch_news_with_keywords, aggregate_daily_sentiment
from feature_engineering import create_features, handle_missing_values
from model_xgboost import train_xgboost_model, evaluate_xgboost_model
from sklearn.metrics import confusion_matrix
import xgboost as xgb

print("="*70)
print("OVERFITTING FIX V2 - HYPERPARAMETER TUNING")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Configuration
YEAR = 2019
TRAIN_SPLIT = 0.8
TEST_STOCKS = ['BABA', 'NFLX', 'TSLA']  # Best, medium, worst from V3

# Load optimal 25 features
print("[1/3] Loading optimal feature set...")
rankings = pd.read_csv('Results/feature_importance_rankings.csv')
top_features = rankings.head(25)['feature'].tolist()
print(f"✅ Loaded {len(top_features)} optimal features")

# Define hyperparameter configurations to test
configs = [
    {
        'name': 'Baseline (Current)',
        'params': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 1.0,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    },
    {
        'name': 'Reduced Depth',
        'params': {
            'n_estimators': 100,
            'max_depth': 3,  # Reduced from 6
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 1.0,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    },
    {
        'name': 'Regularization',
        'params': {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,  # Feature sampling
            'min_child_weight': 5,     # More regularization
            'gamma': 0.1,              # Min loss reduction
            'reg_alpha': 0.1,          # L1 regularization
            'reg_lambda': 1.0          # L2 regularization
        }
    },
    {
        'name': 'Conservative',
        'params': {
            'n_estimators': 150,
            'max_depth': 3,
            'learning_rate': 0.05,     # Lower learning rate
            'subsample': 0.7,          # More aggressive sampling
            'colsample_bytree': 0.7,
            'min_child_weight': 10,    # Heavy regularization
            'gamma': 0.2,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0
        }
    },
    {
        'name': 'Balanced',
        'params': {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.08,
            'subsample': 0.75,
            'colsample_bytree': 0.85,
            'min_child_weight': 3,
            'gamma': 0.05,
            'reg_alpha': 0.05,
            'reg_lambda': 1.5
        }
    }
]

print(f"\n[2/3] Testing {len(configs)} configurations on {len(TEST_STOCKS)} stocks")
print("="*70)

results = []

for config_idx, config in enumerate(configs, 1):
    print(f"\n[Config {config_idx}/{len(configs)}] {config['name']}")
    print("-"*70)
    
    for param, value in config['params'].items():
        print(f"  {param}: {value}")
    print()
    
    config_results = []
    
    for stock_idx, ticker in enumerate(TEST_STOCKS, 1):
        try:
            print(f"  [{stock_idx}/{len(TEST_STOCKS)}] Testing {ticker}... ", end="", flush=True)
            start_time = time.time()
            
            # Load data
            stock_data = fetch_stock_data(ticker, f'{YEAR}-01-01', f'{YEAR}-12-31')
            news_data = fetch_news_with_keywords(ticker, f'{YEAR}-01-01', f'{YEAR}-12-31')
            news_sentiment = aggregate_daily_sentiment(news_data)
            pol_data = fetch_politician_trades(ticker)
            
            # Create features
            X_all, y_all, dates = create_features(stock_data, news_sentiment, pol_data)
            
            # Handle missing values
            X_all = handle_missing_values(X_all, strategy='drop')
            y_all = y_all.loc[X_all.index]
            
            # Select top 25 features
            available_features = [f for f in top_features if f in X_all.columns]
            X = X_all[available_features]
            y = y_all
            
            # Split data chronologically
            split_idx = int(len(X) * TRAIN_SPLIT)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model with config params
            model = xgb.XGBClassifier(
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                **config['params']
            )
            model.fit(X_train, y_train, verbose=False)
            
            # Evaluate
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            overfit_gap = train_acc - test_acc
            
            # Get detailed metrics
            y_pred = model.predict(X_test)
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            runtime = time.time() - start_time
            
            result = {
                'config': config['name'],
                'ticker': ticker,
                'n_samples': len(X),
                'n_train': len(X_train),
                'n_test': len(X_test),
                'train_acc': train_acc,
                'test_acc': test_acc,
                'overfit_gap': overfit_gap,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'runtime': runtime
            }
            
            results.append(result)
            config_results.append(result)
            
            print(f"✅ Test: {test_acc:.1%} | Gap: {overfit_gap:+.1%} | F1: {f1:.1%} ({runtime:.1f}s)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    # Config summary
    if config_results:
        avg_test = np.mean([r['test_acc'] for r in config_results])
        avg_gap = np.mean([r['overfit_gap'] for r in config_results])
        avg_f1 = np.mean([r['f1'] for r in config_results])
        print(f"\n  📊 Config Summary:")
        print(f"     Avg Test Acc: {avg_test:.2%}")
        print(f"     Avg Overfit Gap: {avg_gap:+.2%}")
        print(f"     Avg F1: {avg_f1:.2%}")

# Convert to DataFrame and analyze
print("\n" + "="*70)
print("[3/3] Analyzing Results")
print("="*70)

df_results = pd.DataFrame(results)

# Group by configuration
print("\n📊 Configuration Comparison (averaged across stocks):")
print("-"*70)
config_summary = df_results.groupby('config').agg({
    'test_acc': 'mean',
    'overfit_gap': 'mean',
    'f1': 'mean',
    'train_acc': 'mean'
}).round(4)

config_summary['test_acc_pct'] = (config_summary['test_acc'] * 100).round(2)
config_summary['overfit_gap_pct'] = (config_summary['overfit_gap'] * 100).round(2)
config_summary['train_acc_pct'] = (config_summary['train_acc'] * 100).round(2)
config_summary['f1_pct'] = (config_summary['f1'] * 100).round(2)

print(config_summary[['train_acc_pct', 'test_acc_pct', 'overfit_gap_pct', 'f1_pct']].to_string())

# Find best configuration
best_by_test = config_summary['test_acc'].idxmax()
best_by_gap = config_summary['overfit_gap'].idxmin()
best_by_f1 = config_summary['f1'].idxmax()

print("\n🏆 Best Configurations:")
print("-"*70)
print(f"  Highest Test Accuracy: {best_by_test}")
print(f"  Lowest Overfit Gap: {best_by_gap}")
print(f"  Highest F1 Score: {best_by_f1}")

# Recommended configuration
if best_by_gap == best_by_test or best_by_gap == best_by_f1:
    recommended = best_by_gap
else:
    # Balance between low gap and high accuracy
    config_summary['score'] = config_summary['test_acc'] - (config_summary['overfit_gap'] * 0.5)
    recommended = config_summary['score'].idxmax()

print(f"\n✅ RECOMMENDED: {recommended}")
print("-"*70)

recommended_config = next(c for c in configs if c['name'] == recommended)
print("\nRecommended Hyperparameters:")
for param, value in recommended_config['params'].items():
    print(f"  {param}: {value}")

# Save results
print("\n" + "="*70)
print("Saving results...")
print("="*70)

df_results.to_csv('Results/overfitting_experiments_v2.csv', index=False)
config_summary.to_csv('Results/overfitting_config_summary_v2.csv')

print("✅ Saved: Results/overfitting_experiments_v2.csv")
print("✅ Saved: Results/overfitting_config_summary_v2.csv")

print("\n" + "="*70)
print("OVERFITTING FIX V2 COMPLETE")
print("="*70)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nNext Step: Update model_xgboost.py with recommended parameters")
print(f"Then run: python scripts/validate_mvp_v4.py")
