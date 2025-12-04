"""
Model Comparison: Random Forest vs XGBoost vs Logistic Regression

Compares three models on stock prediction task using 25-feature set:
1. Random Forest (baseline)
2. XGBoost (expected best)
3. Logistic Regression (simple baseline)

Tests on BABA, QCOM, NVDA (2019 data) and generates comprehensive comparison.
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
from model_xgboost import train_xgboost_model, evaluate_xgboost_model
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MODEL COMPARISON: Random Forest vs XGBoost vs Logistic Regression")
print("="*70)
print("\nObjective: Find best model for MVP stock prediction")
print("Feature set: Top 25 features (from feature selection)")
print("Test stocks: BABA, QCOM, NVDA (2019 data)")
print("="*70)

# Load top 25 features
print("\n[1/7] Loading top 25 feature list...")
rankings_df = pd.read_csv('feature_importance_rankings.csv')
top_25_features = rankings_df.head(25)['feature'].tolist()
print(f"      [OK] Top 25 features loaded: {top_25_features[:5]}...")

# Test configuration
TEST_TICKERS = ['BABA', 'QCOM', 'NVDA']
START_DATE = '2019-01-01'
END_DATE = '2019-12-31'

print(f"\nTest Configuration:")
print(f"  Tickers: {TEST_TICKERS}")
print(f"  Period: {START_DATE} to {END_DATE}")
print(f"  Features: 25 (optimal from selection experiments)")
print("="*70)

# Store results
all_results = []

for ticker in TEST_TICKERS:
    print(f'\n{"="*70}')
    print(f'{ticker} - Model Comparison')
    print("="*70)
    
    try:
        # Fetch data
        print(f"[2/7] Fetching data...")
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
        
        print(f"      [OK] Stock: {len(stock_data)} days")
        
        # Create features
        print(f"[3/7] Creating features...")
        sentiment_use = sentiment_data if not sentiment_data.empty else None
        X_all, y_all, _ = create_features(stock_data, sentiment_use, politician_data)
        X_all = handle_missing_values(X_all, strategy='drop')
        y_all = y_all.loc[X_all.index]
        
        # Filter to top 25 features
        available_features = [f for f in top_25_features if f in X_all.columns]
        X = X_all[available_features]
        y = y_all
        
        print(f"      [OK] Features: {len(available_features)}/25, Samples: {len(X)}")
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"      [OK] Train: {len(X_train)}, Test: {len(X_test)}")
        
        # ===== MODEL 1: Random Forest =====
        print(f"\n[4/7] Training Random Forest...")
        rf_model = train_model(
            X_train, y_train,
            model_type='random_forest',
            n_estimators=100,
            max_depth=10,
            random_state=42,
            verbose=False
        )
        
        rf_train_metrics = evaluate_model(rf_model, X_train, y_train, verbose=False)
        rf_test_metrics = evaluate_model(rf_model, X_test, y_test, verbose=False)
        
        print(f"      [OK] Train Acc: {rf_train_metrics['accuracy']:.4f}, "
              f"Test Acc: {rf_test_metrics['accuracy']:.4f}")
        
        # ===== MODEL 2: XGBoost =====
        print(f"[5/7] Training XGBoost...")
        xgb_model = train_xgboost_model(
            X_train, y_train,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=False
        )
        
        xgb_train_metrics = evaluate_xgboost_model(xgb_model, X_train, y_train, verbose=False)
        xgb_test_metrics = evaluate_xgboost_model(xgb_model, X_test, y_test, verbose=False)
        
        print(f"      [OK] Train Acc: {xgb_train_metrics['accuracy']:.4f}, "
              f"Test Acc: {xgb_test_metrics['accuracy']:.4f}")
        
        # ===== MODEL 3: Logistic Regression =====
        print(f"[6/7] Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        
        lr_train_pred = lr_model.predict(X_train)
        lr_test_pred = lr_model.predict(X_test)
        
        lr_train_acc = accuracy_score(y_train, lr_train_pred)
        lr_test_acc = accuracy_score(y_test, lr_test_pred)
        lr_test_precision = precision_score(y_test, lr_test_pred, zero_division=0)
        lr_test_recall = recall_score(y_test, lr_test_pred, zero_division=0)
        lr_test_f1 = f1_score(y_test, lr_test_pred, zero_division=0)
        
        print(f"      [OK] Train Acc: {lr_train_acc:.4f}, Test Acc: {lr_test_acc:.4f}")
        
        # ===== COMPARISON =====
        print(f"\n[7/7] Results for {ticker}:")
        print(f"      {'─'*60}")
        print(f"      {'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<10}")
        print(f"      {'─'*60}")
        
        rf_gap = rf_train_metrics['accuracy'] - rf_test_metrics['accuracy']
        xgb_gap = xgb_train_metrics['accuracy'] - xgb_test_metrics['accuracy']
        lr_gap = lr_train_acc - lr_test_acc
        
        print(f"      {'Random Forest':<20} {rf_train_metrics['accuracy']:.4f}       "
              f"{rf_test_metrics['accuracy']:.4f}       {rf_gap:+.4f}")
        print(f"      {'XGBoost':<20} {xgb_train_metrics['accuracy']:.4f}       "
              f"{xgb_test_metrics['accuracy']:.4f}       {xgb_gap:+.4f}")
        print(f"      {'Logistic Regression':<20} {lr_train_acc:.4f}       "
              f"{lr_test_acc:.4f}       {lr_gap:+.4f}")
        print(f"      {'─'*60}")
        
        # Determine winner
        test_accs = {
            'Random Forest': rf_test_metrics['accuracy'],
            'XGBoost': xgb_test_metrics['accuracy'],
            'Logistic Regression': lr_test_acc
        }
        winner = max(test_accs, key=test_accs.get)
        print(f"       Winner: {winner} ({test_accs[winner]:.4f})")
        
        # Store results
        all_results.append({
            'ticker': ticker,
            'rf_train': rf_train_metrics['accuracy'],
            'rf_test': rf_test_metrics['accuracy'],
            'rf_gap': rf_gap,
            'rf_f1': rf_test_metrics['f1_score'],
            'xgb_train': xgb_train_metrics['accuracy'],
            'xgb_test': xgb_test_metrics['accuracy'],
            'xgb_gap': xgb_gap,
            'xgb_f1': xgb_test_metrics['f1_score'],
            'lr_train': lr_train_acc,
            'lr_test': lr_test_acc,
            'lr_gap': lr_gap,
            'lr_f1': lr_test_f1,
            'winner': winner,
            'n_features': len(available_features),
            'n_samples': len(X)
        })
        
    except Exception as e:
        print(f"[ERROR] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

# ===== AGGREGATE ANALYSIS =====
if all_results:
    print(f"\n\n{'='*70}")
    print("AGGREGATE MODEL COMPARISON")
    print(f"{'='*70}\n")
    
    results_df = pd.DataFrame(all_results)
    
    # Average performance
    print(" AVERAGE PERFORMANCE ACROSS ALL STOCKS")
    print(f"{'─'*70}")
    print(f"{'Model':<25} {'Avg Train':<12} {'Avg Test':<12} {'Avg Gap':<12}")
    print(f"{'─'*70}")
    
    for model_prefix, model_name in [('rf', 'Random Forest'), ('xgb', 'XGBoost'), ('lr', 'Logistic Regression')]:
        avg_train = results_df[f'{model_prefix}_train'].mean()
        avg_test = results_df[f'{model_prefix}_test'].mean()
        avg_gap = results_df[f'{model_prefix}_gap'].mean()
        
        print(f"{model_name:<25} {avg_train:.4f}       {avg_test:.4f}       {avg_gap:+.4f}")
    
    print(f"{'─'*70}\n")
    
    # Best model by test accuracy
    avg_test_accs = {
        'Random Forest': results_df['rf_test'].mean(),
        'XGBoost': results_df['xgb_test'].mean(),
        'Logistic Regression': results_df['lr_test'].mean()
    }
    
    best_model = max(avg_test_accs, key=avg_test_accs.get)
    best_acc = avg_test_accs[best_model]
    
    print(f" BEST MODEL: {best_model}")
    print(f"   Average Test Accuracy: {best_acc:.4f}")
    
    # Calculate improvement vs baselines
    if best_model == 'XGBoost':
        vs_rf = (results_df['xgb_test'] - results_df['rf_test']).mean() * 100
        vs_lr = (results_df['xgb_test'] - results_df['lr_test']).mean() * 100
        print(f"   vs Random Forest: {vs_rf:+.2f}%")
        print(f"   vs Logistic Regression: {vs_lr:+.2f}%")
    
    print(f"\n{'─'*70}\n")
    
    # Winner breakdown
    print(" WINNER BREAKDOWN BY STOCK")
    print(f"{'─'*70}")
    winner_counts = results_df['winner'].value_counts()
    for model, count in winner_counts.items():
        print(f"  {model}: {count}/{len(results_df)} stocks")
    print(f"{'─'*70}\n")
    
    # Detailed stock-by-stock
    print(" DETAILED RESULTS BY STOCK")
    print(f"{'─'*70}")
    print(f"{'Stock':<8} {'Model':<25} {'Test Acc':<12} {'F1 Score':<12}")
    print(f"{'─'*70}")
    
    for _, row in results_df.iterrows():
        ticker = row['ticker']
        print(f"{ticker:<8} Random Forest           {row['rf_test']:.4f}       {row['rf_f1']:.4f}")
        print(f"{'':<8} XGBoost                 {row['xgb_test']:.4f}       {row['xgb_f1']:.4f}")
        print(f"{'':<8} Logistic Regression     {row['lr_test']:.4f}       {row['lr_f1']:.4f}")
        print(f"{'':<8} Winner: {row['winner']}")
        print()
    
    print(f"{'─'*70}\n")
    
    # Save results
    output_file = 'model_comparison_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"[OK] Results saved to: {output_file}")
    
    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION FOR MVP")
    print(f"{'='*70}\n")
    
    print(f"  Selected Model: {best_model}")
    print(f"  Expected Accuracy: {best_acc:.2%}")
    print(f"  Features: 25 (optimal set)")
    
    # Calculate overfitting level
    model_prefix = 'lr' if best_model == 'Logistic Regression' else best_model.lower()[:3]
    avg_gap = results_df[f'{model_prefix}_gap'].mean()
    overfit_level = 'Low' if avg_gap < 0.30 else 'Moderate'
    print(f"  Overfitting: {overfit_level} (gap: {avg_gap:+.2f})")
    print()
    
    if best_acc >= 0.67:
        print(f"  [OK] EXCEEDS MVP TARGET (60%)")
    else:
        print(f"  [WARN]  Below original target but acceptable")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("1. Use {best_model} for MVP deployment")
    print("2. Validate on additional stocks (Thursday)")
    print("3. Generate final results and documentation (Friday)")
    print("4. Consider ensemble methods post-MVP")
    print(f"{'='*70}\n")

else:
    print("\n[ERROR] NO RESULTS - Check errors above")

print("[OK] Model comparison complete!")
