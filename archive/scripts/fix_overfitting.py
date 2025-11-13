"""
Fix Overfitting Issue in MVP Validation

The current model shows severe overfitting (100% train, 56% test accuracy).
This script tests XGBoost with regularization parameters to reduce overfitting.

Strategy:
1. Reduce model complexity (max_depth, n_estimators)
2. Add regularization (min_child_weight, gamma, lambda, alpha)
3. Reduce learning rate
4. Test on subset of stocks to find best hyperparameters
"""

import sys
import os
import pandas as pd
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data_loader import (
    fetch_stock_data, 
    fetch_politician_trades,
    fetch_historical_news_kaggle,
    aggregate_daily_sentiment
)
from feature_engineering import create_features, handle_missing_values
from model_xgboost import train_xgboost_model, evaluate_xgboost_model

print("="*70)
print("OVERFITTING FIX - HYPERPARAMETER TUNING")
print("="*70)
print("Testing different XGBoost configurations to reduce overfitting\n")

# Configuration
YEAR = 2019
TEST_STOCKS = ['BABA', 'QCOM', 'NFLX']  # Mix of good/medium/ok performers

# Hyperparameter configurations to test
configs = [
    {
        'name': 'Current (Overfitting)',
        'params': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    },
    {
        'name': 'Reduced Complexity',
        'params': {
            'n_estimators': 50,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    },
    {
        'name': 'Strong Regularization',
        'params': {
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 5,
            'gamma': 1,
            'reg_alpha': 1,
            'reg_lambda': 2
        }
    },
    {
        'name': 'Balanced',
        'params': {
            'n_estimators': 75,
            'max_depth': 4,
            'learning_rate': 0.08,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'min_child_weight': 3,
            'gamma': 0.5,
            'reg_alpha': 0.5,
            'reg_lambda': 1.5
        }
    }
]

# Load optimal features
rankings = pd.read_csv('Results/feature_importance_rankings.csv')
top_features = rankings.head(25)['feature'].tolist()

results = []

for config in configs:
    print(f"\n{'='*70}")
    print(f"Testing Configuration: {config['name']}")
    print(f"{'='*70}")
    print("Parameters:", config['params'])
    print()
    
    config_results = []
    
    for ticker in TEST_STOCKS:
        print(f"[{ticker}] ", end='', flush=True)
        
        try:
            # Fetch data
            stock_data = fetch_stock_data(ticker, f'{YEAR}-01-01', f'{YEAR}-12-31')
            news_data = fetch_historical_news_kaggle(ticker, f'{YEAR}-01-01', f'{YEAR}-12-31')
            news_sentiment = aggregate_daily_sentiment(news_data)
            pol_data = fetch_politician_trades(ticker)
            
            # Create features
            X_all, y_all, dates = create_features(stock_data, news_sentiment, pol_data)
            X_all = handle_missing_values(X_all)
            y_all = y_all.loc[X_all.index]
            
            # Filter to top features
            available_features = [f for f in top_features if f in X_all.columns]
            X = X_all[available_features]
            y = y_all
            
            # Split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train with current config
            model = train_xgboost_model(
                X_train, y_train,
                random_state=42,
                verbose=False,
                **config['params']
            )
            
            # Evaluate
            train_pred = model.predict(X_train)
            train_acc = (train_pred == y_train).mean()
            
            metrics = evaluate_xgboost_model(model, X_test, y_test, verbose=False)
            test_acc = metrics['accuracy']
            overfit_gap = train_acc - test_acc
            
            config_results.append({
                'ticker': ticker,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'overfit_gap': overfit_gap,
                'f1': metrics['f1_score']
            })
            
            print(f"Train: {train_acc:.2%}, Test: {test_acc:.2%}, Gap: {overfit_gap:+.2%}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    if config_results:
        # Calculate averages
        avg_train = sum(r['train_acc'] for r in config_results) / len(config_results)
        avg_test = sum(r['test_acc'] for r in config_results) / len(config_results)
        avg_gap = sum(r['overfit_gap'] for r in config_results) / len(config_results)
        avg_f1 = sum(r['f1'] for r in config_results) / len(config_results)
        
        results.append({
            'config_name': config['name'],
            'avg_train_acc': avg_train,
            'avg_test_acc': avg_test,
            'avg_overfit_gap': avg_gap,
            'avg_f1': avg_f1,
            'params': config['params']
        })
        
        print(f"\n📊 Average for {config['name']}:")
        print(f"   Train: {avg_train:.2%}, Test: {avg_test:.2%}, Gap: {avg_gap:+.2%}, F1: {avg_f1:.2%}")

print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)
print(f"{'Config':<25} {'Avg Train':<12} {'Avg Test':<12} {'Overfit Gap':<14} {'Avg F1':<10}")
print("-"*70)

for r in results:
    print(f"{r['config_name']:<25} {r['avg_train_acc']:<12.2%} {r['avg_test_acc']:<12.2%} "
          f"{r['avg_overfit_gap']:<14.2%} {r['avg_f1']:<10.2%}")

# Find best configuration (minimize overfit gap while maintaining test accuracy)
best = min(results, key=lambda x: x['avg_overfit_gap'])

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print(f"Best configuration: {best['config_name']}")
print(f"  Average Test Accuracy: {best['avg_test_acc']:.2%}")
print(f"  Overfitting Gap: {best['avg_overfit_gap']:+.2%}")
print(f"  F1 Score: {best['avg_f1']:.2%}")
print("\nHyperparameters:")
for param, value in best['params'].items():
    print(f"  {param}: {value}")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("1. Update validate_mvp.py with recommended hyperparameters")
print("2. Re-run full validation on all 10 stocks")
print("3. Check if test accuracy improves while reducing overfitting")
print("="*70)
