"""
Overfitting Fix Experiments
Tests different XGBoost hyperparameters to reduce overfitting gap (currently 44%)

Strategy:
1. Reduce max_depth (from 6 to 3, 4)
2. Add regularization (alpha, lambda)
3. Increase min_child_weight
4. Compare train vs test accuracy gaps

Target: Reduce gap from 44% to <30% while maintaining test accuracy ~55%+
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import time
from itertools import product

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, src_path)

from data_loader import fetch_stock_data, fetch_politician_trades, fetch_historical_news_kaggle, aggregate_daily_sentiment
from feature_engineering import create_features, handle_missing_values
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

print("="*80)
print("OVERFITTING FIX EXPERIMENTS")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Configuration
YEAR = 2019
TRAIN_SPLIT = 0.8
# Test on ALL 10 stocks to validate if results scale
TEST_STOCKS = ['NFLX', 'NVDA', 'BABA', 'QCOM', 'MU', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']

# Load optimal 25 features
print("[1/4] Loading optimal feature set...")
rankings = pd.read_csv('results/feature_importance_rankings.csv')
top_features = rankings.head(25)['feature'].tolist()
print(f"✅ Loaded {len(top_features)} features")
print()

# Define hyperparameter grid
print("[2/4] Defining hyperparameter grid...")
param_grid = {
    'max_depth': [3, 4, 5, 6],  # Reduce from current 6
    'min_child_weight': [1, 3, 5],  # Increase from current 1
    'alpha': [0, 0.1, 1.0],  # L1 regularization
    'gamma': [0, 0.1, 0.5],  # Minimum loss reduction
    'lambda': [1, 5, 10],  # L2 regularization (default=1)
}

# Base parameters (unchanged)
base_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.1,
    'n_estimators': 100,
    'random_state': 42,
    'use_label_encoder': False
}

print("Parameter ranges:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")
print()

# Experiment configurations to test (focused on key comparisons)
experiments = [
    # Baseline (current V1)
    {'name': 'Baseline', 'max_depth': 6, 'min_child_weight': 1, 'alpha': 0, 'gamma': 0, 'lambda': 1},
    
    # Alpha=1.0 (the "winner" from initial experiments)
    {'name': 'Alpha=1.0', 'max_depth': 6, 'min_child_weight': 1, 'alpha': 1.0, 'gamma': 0, 'lambda': 1},
    
    # Also test a few variations to understand the pattern
    {'name': 'Alpha=0.5', 'max_depth': 6, 'min_child_weight': 1, 'alpha': 0.5, 'gamma': 0, 'lambda': 1},
    {'name': 'MaxDepth=4', 'max_depth': 4, 'min_child_weight': 1, 'alpha': 0, 'gamma': 0, 'lambda': 1},
    {'name': 'Depth4+Alpha1.0', 'max_depth': 4, 'min_child_weight': 1, 'alpha': 1.0, 'gamma': 0, 'lambda': 1},
]

print(f"[3/4] Testing {len(experiments)} configurations on {len(TEST_STOCKS)} stocks...")
print("="*80)
print()

# Results storage
all_results = []
all_stock_details = []  # Store per-stock results for detailed analysis

# Test each configuration
for exp_idx, exp_config in enumerate(experiments, 1):
    exp_name = exp_config['name']
    print(f"[{exp_idx}/{len(experiments)}] Testing: {exp_name}")
    print("-"*80)
    
    # Build model params
    model_params = base_params.copy()
    model_params.update({k: v for k, v in exp_config.items() if k != 'name'})
    
    # Test on each stock
    stock_results = []
    
    for ticker in TEST_STOCKS:
        try:
            # Load data
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
            
            # Skip if insufficient samples
            if len(X) < 30:
                continue
            
            # Train/test split
            split_idx = int(len(X) * TRAIN_SPLIT)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            model = xgb.XGBClassifier(**model_params)
            model.fit(X_train, y_train, verbose=False)
            
            # Evaluate
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            overfit_gap = train_acc - test_acc
            
            stock_result = {
                'ticker': ticker,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'overfit_gap': overfit_gap,
                'n_samples': len(X),
                'n_features': len(available_features)
            }
            stock_results.append(stock_result)
            
            # Also store with experiment details for comprehensive analysis
            all_stock_details.append({
                'experiment': exp_name,
                **stock_result,
                **exp_config
            })
            
            print(f"   {ticker}: Train={train_acc:.1%}, Test={test_acc:.1%}, Gap={overfit_gap:.1%}")
            
        except Exception as e:
            print(f"   {ticker}: ❌ Error - {e}")
            continue
    
    if stock_results:
        # Calculate average metrics
        avg_train = np.mean([r['train_acc'] for r in stock_results])
        avg_test = np.mean([r['test_acc'] for r in stock_results])
        avg_gap = np.mean([r['overfit_gap'] for r in stock_results])
        
        result = {
            'experiment': exp_name,
            'n_stocks': len(stock_results),
            'avg_train_acc': avg_train,
            'avg_test_acc': avg_test,
            'avg_overfit_gap': avg_gap,
            'gap_reduction': (0.44 - avg_gap) * 100,  # vs baseline 44%
            **exp_config
        }
        all_results.append(result)
        
        print(f"   📊 Average: Train={avg_train:.1%}, Test={avg_test:.1%}, Gap={avg_gap:.1%}")
        print()
    else:
        print(f"   ⚠️  No successful stocks")
        print()

# Save results
print("="*80)
print("[4/4] Saving results...")
results_df = pd.DataFrame(all_results)

# Remove redundant name column
if 'name' in results_df.columns:
    results_df = results_df.drop('name', axis=1)

results_df.to_csv('results/overfitting_experiments.csv', index=False)
print(f"✅ Saved summary to results/overfitting_experiments.csv")

# Save detailed per-stock results
if all_stock_details:
    details_df = pd.DataFrame(all_stock_details)
    # Remove redundant 'name' column if present
    if 'name' in details_df.columns:
        details_df = details_df.drop('name', axis=1)
    details_df.to_csv('results/overfitting_experiments_detailed.csv', index=False)
    print(f"✅ Saved detailed per-stock results to results/overfitting_experiments_detailed.csv")
print()

# Summary
print("="*80)
print("EXPERIMENT RESULTS SUMMARY")
print("="*80)
print()

# Sort by smallest overfit gap
results_df_sorted = results_df.sort_values('avg_overfit_gap')

print(f"{'Experiment':<20} {'Avg Train':<12} {'Avg Test':<12} {'Overfit Gap':<14} {'Gap Reduction':<15}")
print("-"*80)

for _, row in results_df_sorted.iterrows():
    exp = row['experiment']
    train = f"{row['avg_train_acc']:.1%}"
    test = f"{row['avg_test_acc']:.1%}"
    gap = f"{row['avg_overfit_gap']:.1%}"
    reduction = f"{row['gap_reduction']:.1%}"
    
    print(f"{exp:<20} {train:<12} {test:<12} {gap:<14} {reduction:<15}")

print()
print("="*80)
print("BEST CONFIGURATIONS:")
print("="*80)

# Top 3 by smallest gap
print("\n1. SMALLEST OVERFIT GAP (Top 3):")
print("-"*80)
for idx, (_, row) in enumerate(results_df_sorted.head(3).iterrows(), 1):
    print(f"{idx}. {row['experiment']}")
    print(f"   max_depth={row['max_depth']}, min_child_weight={row['min_child_weight']}, "
          f"alpha={row['alpha']}, lambda={row['lambda']}")
    print(f"   Gap: {row['avg_overfit_gap']:.1%} (improvement: {row['gap_reduction']:.1%})")
    print(f"   Test Accuracy: {row['avg_test_acc']:.1%}")
    print()

# Top 3 by best test accuracy
print("2. HIGHEST TEST ACCURACY (Top 3):")
print("-"*80)
results_df_by_test = results_df.sort_values('avg_test_acc', ascending=False)
for idx, (_, row) in enumerate(results_df_by_test.head(3).iterrows(), 1):
    print(f"{idx}. {row['experiment']}")
    print(f"   max_depth={row['max_depth']}, min_child_weight={row['min_child_weight']}, "
          f"alpha={row['alpha']}, lambda={row['lambda']}")
    print(f"   Test Accuracy: {row['avg_test_acc']:.1%}")
    print(f"   Gap: {row['avg_overfit_gap']:.1%}")
    print()

# Recommendations
print("="*80)
print("RECOMMENDATIONS:")
print("="*80)
best_config = results_df_sorted.iloc[0]
print(f"\n✅ RECOMMENDED CONFIGURATION: {best_config['experiment']}")
print(f"   Parameters:")
print(f"   - max_depth = {best_config['max_depth']}")
print(f"   - min_child_weight = {best_config['min_child_weight']}")
print(f"   - alpha = {best_config['alpha']}")
print(f"   - lambda = {best_config['lambda']}")
print(f"   - gamma = {best_config['gamma']}")
print(f"\n   Expected Performance:")
print(f"   - Test Accuracy: {best_config['avg_test_acc']:.1%}")
print(f"   - Overfit Gap: {best_config['avg_overfit_gap']:.1%} (vs baseline 44%)")
print(f"   - Gap Reduction: {best_config['gap_reduction']:.1%}")
print()

print("="*80)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
