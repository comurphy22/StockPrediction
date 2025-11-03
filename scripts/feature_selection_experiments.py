"""
Feature Selection Experiments: Find Optimal Feature Count

Tests different feature counts (5, 10, 15, 20, 25, ALL) to find optimal balance
between performance and overfitting.

Methodology:
1. Use top N features from importance rankings
2. Test on best-performing stocks (BABA, QCOM, NVDA)
3. Measure: train accuracy, test accuracy, overfitting gap
4. Compare consistency across stocks

Goal: Identify optimal feature count for MVP (target: 10-15 features)
"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

from data_loader import (
    fetch_stock_data, 
    fetch_politician_trades,
    fetch_historical_news_kaggle,
    aggregate_daily_sentiment
)
from feature_engineering import create_features, handle_missing_values
from model import train_model, evaluate_model
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FEATURE SELECTION EXPERIMENTS")
print("="*70)
print("\nObjective: Find optimal feature count for MVP")
print("Method: Test 5, 10, 15, 20, 25, ALL features across 3 stocks")
print("Metrics: Train accuracy, Test accuracy, Overfitting gap")
print("="*70)

# Load feature rankings
rankings_df = pd.read_csv('feature_importance_rankings.csv')
feature_ranking = rankings_df['feature'].tolist()

print(f"\n✅ Loaded feature rankings: {len(feature_ranking)} features")
print(f"   Top 5: {feature_ranking[:5]}")

# Test configuration
TEST_TICKERS = ['BABA', 'QCOM', 'NVDA']
START_DATE = '2019-01-01'
END_DATE = '2019-12-31'

# Feature counts to test
FEATURE_COUNTS = [5, 10, 15, 20, 25, 'ALL']

print(f"\nTest Configuration:")
print(f"  Tickers: {TEST_TICKERS}")
print(f"  Period: {START_DATE} to {END_DATE} (2019)")
print(f"  Feature counts: {FEATURE_COUNTS}")
print(f"  Model: Random Forest (n_estimators=100, max_depth=10)")
print("="*70)

# Store all results
all_results = []

for ticker in TEST_TICKERS:
    print(f'\n{"="*70}')
    print(f'{ticker} - Feature Selection Experiments')
    print("="*70)
    
    try:
        # Fetch data
        print(f"[1/4] Fetching data...")
        stock_data = fetch_stock_data(ticker, START_DATE, END_DATE)
        
        news_data = fetch_historical_news_kaggle(ticker, START_DATE, END_DATE)
        sentiment_data = aggregate_daily_sentiment(news_data) if not news_data.empty else pd.DataFrame()
        
        politician_data = fetch_politician_trades(ticker)
        if not politician_data.empty:
            politician_data['date'] = pd.to_datetime(politician_data['date'])
            politician_data = politician_data[
                (politician_data['date'] >= START_DATE) & 
                (politician_data['date'] <= END_DATE)
            ]
        
        print(f"      ✅ Stock: {len(stock_data)} days, News: {len(sentiment_data)} days, Trades: {len(politician_data)}")
        
        # Create all features
        print(f"[2/4] Creating all features...")
        sentiment_use = sentiment_data if not sentiment_data.empty else None
        X_all, y_all, feature_names = create_features(stock_data, sentiment_use, politician_data)
        X_all = handle_missing_values(X_all, strategy='drop')
        y_all = y_all.loc[X_all.index]
        
        print(f"      ✅ {X_all.shape[1]} total features, {len(X_all)} samples")
        
        # Split data
        split_idx = int(len(X_all) * 0.8)
        X_train_all = X_all[:split_idx]
        X_test_all = X_all[split_idx:]
        y_train = y_all[:split_idx]
        y_test = y_all[split_idx:]
        
        print(f"      ✅ Train: {len(X_train_all)}, Test: {len(X_test_all)}")
        
        # Test each feature count
        print(f"\n[3/4] Testing feature counts...")
        
        ticker_results = []
        
        for n_features in FEATURE_COUNTS:
            if n_features == 'ALL':
                selected_features = X_all.columns.tolist()
                n = len(selected_features)
            else:
                # Select top N features that exist in the dataset
                selected_features = [f for f in feature_ranking[:n_features] if f in X_all.columns]
                n = len(selected_features)
            
            # Subset data to selected features
            X_train = X_train_all[selected_features]
            X_test = X_test_all[selected_features]
            
            # Train model
            model = train_model(
                X_train, y_train,
                model_type='random_forest',
                n_estimators=100,
                max_depth=10,
                random_state=42,
                verbose=False
            )
            
            # Evaluate on train set
            train_metrics = evaluate_model(model, X_train, y_train, verbose=False)
            train_acc = train_metrics['accuracy']
            
            # Evaluate on test set
            test_metrics = evaluate_model(model, X_test, y_test, verbose=False)
            test_acc = test_metrics['accuracy']
            
            # Calculate overfitting gap
            overfit_gap = train_acc - test_acc
            
            # Store result
            result = {
                'ticker': ticker,
                'n_features': n,
                'feature_set': n_features,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'overfit_gap': overfit_gap,
                'train_f1': train_metrics['f1_score'],
                'test_f1': test_metrics['f1_score'],
                'features': selected_features[:5]  # Store first 5 for reference
            }
            
            ticker_results.append(result)
            all_results.append(result)
            
            # Status indicator
            if overfit_gap > 0.10:
                status = "❌ High overfit"
            elif overfit_gap > 0.05:
                status = "⚠️  Moderate overfit"
            else:
                status = "✅ Good"
            
            print(f"      {n:>3} features: Train={train_acc:.4f}, Test={test_acc:.4f}, "
                  f"Gap={overfit_gap:+.4f} {status}")
        
        # Best result for this ticker
        print(f"\n[4/4] Best result for {ticker}:")
        best = max(ticker_results, key=lambda x: x['test_acc'])
        print(f"      🏆 {best['n_features']} features: Test accuracy = {best['test_acc']:.4f}")
        print(f"         Features: {best['features']}")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

# ===== AGGREGATE ANALYSIS =====
if all_results:
    print(f"\n\n{'='*70}")
    print("AGGREGATE ANALYSIS: Feature Count Comparison")
    print(f"{'='*70}\n")
    
    results_df = pd.DataFrame(all_results)
    
    # Group by feature count
    summary = results_df.groupby('n_features').agg({
        'train_acc': ['mean', 'std'],
        'test_acc': ['mean', 'std'],
        'overfit_gap': ['mean', 'std']
    }).round(4)
    
    print("📊 AVERAGE PERFORMANCE BY FEATURE COUNT")
    print(f"{'─'*70}")
    print(f"{'N':<4} {'Train Acc':<12} {'Test Acc':<12} {'Overfit Gap':<15} {'Status'}")
    print(f"{'─'*70}")
    
    for n_features in sorted(results_df['n_features'].unique()):
        subset = results_df[results_df['n_features'] == n_features]
        train_mean = subset['train_acc'].mean()
        train_std = subset['train_acc'].std()
        test_mean = subset['test_acc'].mean()
        test_std = subset['test_acc'].std()
        gap_mean = subset['overfit_gap'].mean()
        
        if gap_mean > 0.10:
            status = "❌"
        elif gap_mean > 0.05:
            status = "⚠️ "
        else:
            status = "✅"
        
        print(f"{n_features:<4} {train_mean:.4f}±{train_std:.4f} {test_mean:.4f}±{test_std:.4f}  "
              f"{gap_mean:>+7.4f}        {status}")
    
    print(f"{'─'*70}\n")
    
    # Find optimal feature count
    print("🎯 OPTIMAL FEATURE COUNT ANALYSIS")
    print(f"{'─'*70}")
    
    # Method 1: Highest average test accuracy
    best_test_acc = results_df.groupby('n_features')['test_acc'].mean().idxmax()
    best_test_acc_value = results_df.groupby('n_features')['test_acc'].mean().max()
    
    # Method 2: Best test accuracy with low overfitting (gap < 0.05)
    low_overfit = results_df[results_df['overfit_gap'] < 0.05]
    if not low_overfit.empty:
        best_balanced = low_overfit.groupby('n_features')['test_acc'].mean().idxmax()
        best_balanced_value = low_overfit.groupby('n_features')['test_acc'].mean().max()
    else:
        best_balanced = best_test_acc
        best_balanced_value = best_test_acc_value
    
    # Method 3: Elbow method - find diminishing returns
    test_acc_by_n = results_df.groupby('n_features')['test_acc'].mean().sort_index()
    improvements = test_acc_by_n.diff()
    small_improvements = improvements[improvements < 0.01]
    elbow_point = small_improvements.index[0] if len(small_improvements) > 0 else test_acc_by_n.index[-1]
    
    print(f"  Method 1 (Highest test accuracy): {best_test_acc} features ({best_test_acc_value:.4f})")
    print(f"  Method 2 (Best with gap<0.05):    {best_balanced} features ({best_balanced_value:.4f})")
    print(f"  Method 3 (Elbow point):            {elbow_point} features (diminishing returns)")
    print(f"{'─'*70}\n")
    
    # Stock-by-stock best
    print("📈 BEST FEATURE COUNT PER STOCK")
    print(f"{'─'*70}")
    print(f"{'Ticker':<8} {'Best N':<8} {'Test Acc':<12} {'Overfit Gap':<15}")
    print(f"{'─'*70}")
    
    for ticker in TEST_TICKERS:
        ticker_data = results_df[results_df['ticker'] == ticker]
        best_row = ticker_data.loc[ticker_data['test_acc'].idxmax()]
        print(f"{best_row['ticker']:<8} {best_row['n_features']:<8} "
              f"{best_row['test_acc']:.4f}       {best_row['overfit_gap']:>+7.4f}")
    
    print(f"{'─'*70}\n")
    
    # Recommendation
    print("💡 RECOMMENDATION")
    print(f"{'─'*70}")
    
    # Calculate recommendation based on multiple factors
    scores = {}
    for n in results_df['n_features'].unique():
        subset = results_df[results_df['n_features'] == n]
        test_acc = subset['test_acc'].mean()
        gap = subset['overfit_gap'].mean()
        consistency = 1 - subset['test_acc'].std()  # Lower std = higher consistency
        
        # Score: prioritize test accuracy, penalize overfitting, reward consistency
        score = test_acc - (gap * 0.5) + (consistency * 0.1)
        scores[n] = score
    
    recommended_n = max(scores, key=scores.get)
    recommended_subset = results_df[results_df['n_features'] == recommended_n]
    
    print(f"  🏆 Recommended: {recommended_n} features")
    print(f"     Average test accuracy: {recommended_subset['test_acc'].mean():.4f}")
    print(f"     Average overfit gap:   {recommended_subset['overfit_gap'].mean():+.4f}")
    print(f"     Consistency (1-std):   {1 - recommended_subset['test_acc'].std():.4f}")
    print()
    
    # Get top N features for recommendation
    top_features = [f for f in feature_ranking[:recommended_n] if f in X_all.columns]
    
    print(f"  📋 Top {recommended_n} features:")
    for i, feat in enumerate(top_features, 1):
        category = rankings_df[rankings_df['feature'] == feat]['category'].values[0]
        importance = rankings_df[rankings_df['feature'] == feat]['mean_importance'].values[0]
        print(f"     {i:2d}. {feat:<35} ({category}, {importance:.4f})")
    
    print(f"\n{'─'*70}\n")
    
    # Comparison with ALL features
    all_features_results = results_df[results_df['feature_set'] == 'ALL']
    if not all_features_results.empty:
        print("📊 COMPARISON: Recommended vs ALL Features")
        print(f"{'─'*70}")
        
        rec_test = recommended_subset['test_acc'].mean()
        all_test = all_features_results['test_acc'].mean()
        
        rec_gap = recommended_subset['overfit_gap'].mean()
        all_gap = all_features_results['overfit_gap'].mean()
        
        print(f"  Test Accuracy:")
        print(f"    {recommended_n} features: {rec_test:.4f}")
        print(f"    ALL features:  {all_test:.4f}")
        print(f"    Difference:    {rec_test - all_test:+.4f}")
        print()
        print(f"  Overfitting Gap:")
        print(f"    {recommended_n} features: {rec_gap:+.4f}")
        print(f"    ALL features:  {all_gap:+.4f}")
        print(f"    Difference:    {rec_gap - all_gap:+.4f}")
        print()
        
        if rec_test >= all_test - 0.01 and rec_gap < all_gap:
            print(f"  ✅ WINNER: {recommended_n} features (better or equal test acc, less overfitting)")
        elif rec_test > all_test:
            print(f"  ✅ WINNER: {recommended_n} features (better test accuracy)")
        else:
            print(f"  ⚠️  ALL features marginally better, but {recommended_n} features recommended for MVP")
        
        print(f"{'─'*70}\n")
    
    # Save results
    output_file = 'feature_selection_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"✅ Results saved to: {output_file}")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print(f"1. Use {recommended_n} features for MVP")
    print(f"2. Update feature_engineering.py to filter to top {recommended_n} features")
    print(f"3. Proceed with XGBoost implementation (Wednesday)")
    print(f"4. MVP validation on best stocks (Thursday)")
    print(f"{'='*70}\n")

else:
    print("\n❌ NO RESULTS - Check errors above")

print("✅ Feature selection experiments complete!")
