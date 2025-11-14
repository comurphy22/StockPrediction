"""
Multi-Year Validation Script

Tests XGBoost model performance across multiple years (2018, 2019, 2020)
to evaluate generalization and regime stability.

Data Coverage Confirmed:
- 2018: 332,601 news articles (bull market)
- 2019: 368,574 news articles (peak coverage, stable market)
- 2020:  96,343 news articles (COVID crash + recovery)

This provides three distinct market regimes for robust evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
import time

# Import configuration
from config import (
    VALIDATION_YEARS, VALIDATION_STOCKS, TRAIN_TEST_SPLIT,
    check_api_keys, setup_logging
)

# Import data loaders and models
from data_loader import (
    fetch_stock_data,
    fetch_politician_trades,
    fetch_historical_news_kaggle,
    aggregate_daily_sentiment
)
from feature_engineering import create_features, handle_missing_values
from model_xgboost import train_xgboost_model, evaluate_xgboost_model

# Setup logging
logger = setup_logging('multiyear_validation.log')

print("="*80)
print("MULTI-YEAR VALIDATION - MARKET REGIME ANALYSIS")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Check API keys
if not check_api_keys():
    print("\n⚠️  Please configure API keys in .env file")
    print("   See ENV_SETUP_INSTRUCTIONS.md for help\n")
    sys.exit(1)

# Configuration
YEARS = VALIDATION_YEARS  # [2018, 2019, 2020]
STOCKS = VALIDATION_STOCKS  # Top 8 from MVP

print(f"[Configuration]")
print(f"   Years: {YEARS}")
print(f"   Stocks: {STOCKS}")
print(f"   Train/Test Split: {int(TRAIN_TEST_SPLIT*100)}/{int((1-TRAIN_TEST_SPLIT)*100)}")
print()

# Results storage
results = []
detailed_results = []

# Process each year
for year_idx, year in enumerate(YEARS, 1):
    print(f"\n{'='*80}")
    print(f"YEAR {year} ({year_idx}/{len(YEARS)})")
    print(f"{'='*80}")
    
    # Market context
    market_context = {
        2018: "Bull market, strong economy",
        2019: "Stable market, peak news coverage",
        2020: "COVID-19 crash + recovery, high volatility"
    }
    print(f"Market Context: {market_context.get(year, 'N/A')}")
    print()
    
    year_start_time = time.time()
    
    # Process each stock
    for stock_idx, ticker in enumerate(STOCKS, 1):
        print(f"\n[{stock_idx}/{len(STOCKS)}] {ticker} - {year}")
        print("-"*80)
        
        stock_start_time = time.time()
        
        try:
            # Step 1: Fetch stock data
            print(f"   [1/7] Fetching stock data...", end=" ")
            stock_data = fetch_stock_data(ticker, f'{year}-01-01', f'{year}-12-31')
            print(f"✅ {len(stock_data)} trading days")
            
            # Step 2: Load news sentiment
            print(f"   [2/7] Loading news sentiment...", end=" ")
            news_data = fetch_historical_news_kaggle(ticker, f'{year}-01-01', f'{year}-12-31')
            news_sentiment = aggregate_daily_sentiment(news_data)
            print(f"✅ {len(news_data)} articles")
            
            # Step 3: Load politician trades
            print(f"   [3/7] Loading politician trades...", end=" ")
            pol_data = fetch_politician_trades(ticker)
            
            # Filter to year
            if not pol_data.empty:
                pol_data['date'] = pd.to_datetime(pol_data['date'])
                pol_data = pol_data[
                    (pol_data['date'] >= f'{year}-01-01') & 
                    (pol_data['date'] <= f'{year}-12-31')
                ]
            print(f"✅ {len(pol_data)} trades")
            
            # Step 4: Create features
            print(f"   [4/7] Engineering features...", end=" ")
            X, y, dates = create_features(
                stock_data, 
                news_sentiment, 
                pol_data, 
                ticker,
                add_market_features=True
            )
            print(f"✅ {X.shape[1]} features")
            
            # Step 5: Handle missing values
            print(f"   [5/7] Cleaning data...", end=" ")
            X_clean = handle_missing_values(X, strategy='drop')
            y_clean = y.loc[X_clean.index]
            dates_clean = dates.loc[X_clean.index] if dates is not None else None
            print(f"✅ {len(X_clean)} samples")
            
            # Check if we have enough data
            if len(X_clean) < 50:
                print(f"   ⚠️  Insufficient data (< 50 samples), skipping...")
                continue
            
            # Step 6: Train/Test Split
            print(f"   [6/7] Splitting data...", end=" ")
            split_idx = int(len(X_clean) * TRAIN_TEST_SPLIT)
            
            X_train = X_clean.iloc[:split_idx]
            X_test = X_clean.iloc[split_idx:]
            y_train = y_clean.iloc[:split_idx]
            y_test = y_clean.iloc[split_idx:]
            
            print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Step 7: Train and Evaluate
            print(f"   [7/7] Training XGBoost...", end=" ")
            
            # Train model
            model = train_xgboost_model(X_train, y_train, verbose=False)
            
            # Evaluate on train and test
            train_metrics = evaluate_xgboost_model(model, X_train, y_train)
            test_metrics = evaluate_xgboost_model(model, X_test, y_test)
            
            train_acc = train_metrics['accuracy']
            test_acc = test_metrics['accuracy']
            overfit_gap = train_acc - test_acc
            
            print(f"✅ Train: {train_acc:.2%}, Test: {test_acc:.2%}, Gap: {overfit_gap:.2%}")
            
            # Calculate runtime
            runtime = time.time() - stock_start_time
            
            # Store results
            result = {
                'year': year,
                'ticker': ticker,
                'n_samples': len(X_clean),
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_features': X_clean.shape[1],
                'news_articles': len(news_data),
                'politician_trades': len(pol_data),
                'train_acc': train_acc,
                'test_acc': test_acc,
                'overfit_gap': overfit_gap,
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1': test_metrics['f1_score'],
                'runtime_sec': runtime
            }
            results.append(result)
            
            print(f"   ✅ Completed in {runtime:.1f}s")
            
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    year_runtime = time.time() - year_start_time
    print(f"\n{'='*80}")
    print(f"Year {year} completed in {year_runtime/60:.1f} minutes")
    print(f"{'='*80}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}")

if results:
    # Save to CSV
    df = pd.DataFrame(results)
    output_path = 'results/multiyear_validation_results.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Saved results to: {output_path}")
    
    # ========================================================================
    # ANALYSIS BY YEAR
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("PERFORMANCE BY YEAR")
    print(f"{'='*80}")
    
    for year in YEARS:
        year_data = df[df['year'] == year]
        if len(year_data) == 0:
            continue
        
        print(f"\n{year} ({market_context.get(year, 'N/A')}):")
        print(f"   Stocks Tested: {len(year_data)}")
        print(f"   Avg Test Accuracy: {year_data['test_acc'].mean():.2%} (±{year_data['test_acc'].std():.2%})")
        print(f"   Avg F1 Score: {year_data['f1'].mean():.2%}")
        print(f"   Avg Overfitting Gap: {year_data['overfit_gap'].mean():.2%}")
        print(f"   Best Stock: {year_data.loc[year_data['test_acc'].idxmax(), 'ticker']} ({year_data['test_acc'].max():.2%})")
        print(f"   Worst Stock: {year_data.loc[year_data['test_acc'].idxmin(), 'ticker']} ({year_data['test_acc'].min():.2%})")
        print(f"   Avg News Articles: {year_data['news_articles'].mean():.0f}")
        print(f"   Avg Politician Trades: {year_data['politician_trades'].mean():.0f}")
    
    # ========================================================================
    # CROSS-YEAR COMPARISON
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("CROSS-YEAR COMPARISON")
    print(f"{'='*80}")
    
    # Average performance by year
    year_summary = df.groupby('year').agg({
        'test_acc': ['mean', 'std'],
        'f1': 'mean',
        'overfit_gap': 'mean'
    }).round(4)
    
    print("\n   Performance Summary:")
    print(year_summary)
    
    # Year-to-year variance
    year_means = df.groupby('year')['test_acc'].mean()
    max_year = year_means.idxmax()
    min_year = year_means.idxmin()
    variance = year_means.max() - year_means.min()
    
    print(f"\n   Year-to-Year Variance:")
    print(f"      Best Year: {max_year} ({year_means[max_year]:.2%})")
    print(f"      Worst Year: {min_year} ({year_means[min_year]:.2%})")
    print(f"      Difference: {variance:.2%}")
    
    # ========================================================================
    # PER-STOCK CONSISTENCY
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("PER-STOCK CONSISTENCY ACROSS YEARS")
    print(f"{'='*80}")
    
    # Pivot table: stocks x years
    pivot = df.pivot_table(values='test_acc', index='ticker', columns='year')
    
    print("\n   Accuracy by Stock and Year:")
    print(pivot.to_string())
    
    # Stock consistency (std dev across years)
    stock_consistency = pivot.std(axis=1).sort_values()
    
    print(f"\n   Most Consistent Stocks (low variance across years):")
    for ticker, std in stock_consistency.head(3).items():
        avg_acc = pivot.loc[ticker].mean()
        print(f"      {ticker}: {avg_acc:.2%} avg (±{std:.2%})")
    
    print(f"\n   Least Consistent Stocks (high variance across years):")
    for ticker, std in stock_consistency.tail(3).items():
        avg_acc = pivot.loc[ticker].mean()
        print(f"      {ticker}: {avg_acc:.2%} avg (±{std:.2%})")

else:
    print("❌ No results to save")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("VALIDATION COMPLETE")
print(f"{'='*80}")

if results:
    overall_avg = df['test_acc'].mean()
    overall_std = df['test_acc'].std()
    
    print(f"\n📊 Overall Results:")
    print(f"   Total Experiments: {len(df)}")
    print(f"   Average Test Accuracy: {overall_avg:.2%} (±{overall_std:.2%})")
    print(f"   Best Single Result: {df['test_acc'].max():.2%} ({df.loc[df['test_acc'].idxmax(), 'ticker']} {df.loc[df['test_acc'].idxmax(), 'year']})")
    print(f"   Worst Single Result: {df['test_acc'].min():.2%} ({df.loc[df['test_acc'].idxmin(), 'ticker']} {df.loc[df['test_acc'].idxmin(), 'year']})")

print(f"\n🎯 Key Insights:")
print(f"   - Model tested across 3 distinct market regimes")
print(f"   - Performance variance indicates regime-dependent behavior")
print(f"   - Use these results to calibrate paper's generalizability claims")

print(f"\n📄 Next Steps:")
print(f"   1. Review results/multiyear_validation_results.csv")
print(f"   2. Create visualizations (year comparison, stock consistency)")
print(f"   3. Document in docs/MULTIYEAR_VALIDATION_RESULTS.md")
print(f"   4. Update paper with multi-year findings")

print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}\n")

