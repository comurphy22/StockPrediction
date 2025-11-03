"""
Feature Importance Analysis: Identify Most Valuable Features

Analyzes all 37 features (12 technical + 2 basic pol + 23 advanced pol)
Tests on best-performing stocks from multi-year validation
Ranks features by Random Forest importance scores
Saves results for feature selection experiments

Current Feature Set:
- Technical: 12 features (SMA, RSI, MACD, etc.)
- Basic Politician: 2 features (buy_count, trade_amount)
- Advanced Politician: 23 features (net indices, momentum, quality)
Total: 37 features
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
from model import train_model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)
print("\nObjective: Identify top 10-15 features from 37 total features")
print("Method: Random Forest feature importance on best-performing stocks")
print("="*70)

# Use best-performing stocks from multi-year validation
# BABA: +9.89%, QCOM: +4.61%, NVDA: +1.55% (from 2019 data)
TEST_TICKERS = ['BABA', 'QCOM', 'NVDA']

# Use 2019 data (best year: +4.37% improvement)
START_DATE = '2019-01-01'
END_DATE = '2019-12-31'

print(f"\nTest Configuration:")
print(f"  Tickers: {TEST_TICKERS}")
print(f"  Period: {START_DATE} to {END_DATE} (2019 - best year)")
print(f"  Model: Random Forest (n_estimators=100, max_depth=10)")
print("="*70)

# Store feature importance scores from each stock
all_importances = []

for ticker in TEST_TICKERS:
    print(f'\n{"="*70}')
    print(f'{ticker} - Feature Importance Extraction')
    print("="*70)
    
    try:
        # Fetch data
        print(f"[1/6] Fetching stock data...")
        stock_data = fetch_stock_data(ticker, START_DATE, END_DATE)
        print(f"      ✅ {len(stock_data)} days")
        
        print(f"[2/6] Fetching news sentiment...")
        news_data = fetch_historical_news_kaggle(ticker, START_DATE, END_DATE)
        if not news_data.empty:
            sentiment_data = aggregate_daily_sentiment(news_data)
            print(f"      ✅ {len(news_data)} news → {len(sentiment_data)} days with sentiment")
        else:
            sentiment_data = pd.DataFrame()
            print(f"      ⚠️  No news data")
        
        print(f"[3/6] Fetching politician trades...")
        politician_data = fetch_politician_trades(ticker)
        if not politician_data.empty:
            politician_data['date'] = pd.to_datetime(politician_data['date'])
            politician_data = politician_data[
                (politician_data['date'] >= START_DATE) & 
                (politician_data['date'] <= END_DATE)
            ]
            print(f"      ✅ {len(politician_data)} trades in range")
        else:
            print(f"      ⚠️  No trades")
        
        # Create all features
        print(f"[4/6] Creating all features...")
        sentiment_use = sentiment_data if not sentiment_data.empty else None
        X, y, feature_names = create_features(stock_data, sentiment_use, politician_data)
        X = handle_missing_values(X, strategy='drop')
        y = y.loc[X.index]
        
        print(f"      ✅ {X.shape[1]} features, {len(X)} samples")
        print(f"      Features: {list(X.columns)}")
        
        # Train model
        print(f"[5/6] Training Random Forest...")
        split = int(len(X) * 0.8)
        model = train_model(
            X[:split], y[:split],
            model_type='random_forest', 
            n_estimators=100, 
            max_depth=10,
            random_state=42, 
            verbose=False
        )
        print(f"      ✅ Model trained")
        
        # Extract feature importances
        print(f"[6/6] Extracting feature importances...")
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances,
            'ticker': ticker
        }).sort_values('importance', ascending=False)
        
        all_importances.append(feature_importance_df)
        
        print(f"\n      Top 10 Features for {ticker}:")
        print(f"      {'─'*60}")
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"      {row['feature']:<30} {row['importance']:.4f}")
        print(f"      {'─'*60}")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

# ===== AGGREGATE ANALYSIS =====
if all_importances:
    print(f"\n\n{'='*70}")
    print("AGGREGATED FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*70}\n")
    
    # Combine all importances
    combined_df = pd.concat(all_importances, ignore_index=True)
    
    # Calculate average importance across all stocks
    avg_importance = combined_df.groupby('feature')['importance'].agg([
        ('mean_importance', 'mean'),
        ('std_importance', 'std'),
        ('min_importance', 'min'),
        ('max_importance', 'max')
    ]).reset_index()
    
    # Sort by mean importance
    avg_importance = avg_importance.sort_values('mean_importance', ascending=False)
    
    # Categorize features
    def categorize_feature(name):
        name_lower = name.lower()
        if 'sentiment' in name_lower or 'news' in name_lower:
            return 'Sentiment'
        elif any(x in name_lower for x in ['net_', 'buy_percentage', 'conviction', 
                                            'momentum', 'flow', 'days_since', 
                                            'sale_count', 'trade_index']):
            return 'Advanced Politician'
        elif 'politician' in name_lower:
            return 'Basic Politician'
        else:
            return 'Technical'
    
    avg_importance['category'] = avg_importance['feature'].apply(categorize_feature)
    
    # Print top 20 features
    print("📊 TOP 20 FEATURES (by average importance)")
    print(f"{'─'*70}")
    print(f"{'Rank':<6} {'Feature':<30} {'Avg':<10} {'Std':<10} {'Category'}")
    print(f"{'─'*70}")
    
    for idx, row in avg_importance.head(20).iterrows():
        rank = list(avg_importance.index).index(idx) + 1
        print(f"{rank:<6} {row['feature']:<30} {row['mean_importance']:.4f}    "
              f"{row['std_importance']:.4f}    {row['category']}")
    
    print(f"{'─'*70}\n")
    
    # Category breakdown
    print("📈 IMPORTANCE BY CATEGORY")
    print(f"{'─'*70}")
    category_stats = avg_importance.groupby('category').agg({
        'mean_importance': ['sum', 'mean', 'count']
    }).round(4)
    category_stats.columns = ['Total', 'Average', 'Count']
    category_stats = category_stats.sort_values('Total', ascending=False)
    print(category_stats)
    print(f"{'─'*70}\n")
    
    # Recommendations
    print("🎯 FEATURE SELECTION RECOMMENDATIONS")
    print(f"{'─'*70}")
    
    # Top features by cumulative importance
    avg_importance['cumulative_importance'] = avg_importance['mean_importance'].cumsum()
    total_importance = avg_importance['mean_importance'].sum()
    avg_importance['cumulative_pct'] = (avg_importance['cumulative_importance'] / total_importance) * 100
    
    # Find how many features for 80%, 90%, 95% importance
    n_80 = len(avg_importance[avg_importance['cumulative_pct'] <= 80])
    n_90 = len(avg_importance[avg_importance['cumulative_pct'] <= 90])
    n_95 = len(avg_importance[avg_importance['cumulative_pct'] <= 95])
    
    print(f"  • Top {n_80} features capture 80% of importance")
    print(f"  • Top {n_90} features capture 90% of importance")
    print(f"  • Top {n_95} features capture 95% of importance")
    print()
    print(f"  Recommended range: {n_80}-{n_90} features")
    print(f"  Target for experiments: 5, 10, 15, 20, 25 features")
    print(f"{'─'*70}\n")
    
    # Top feature sets for testing
    print("🔬 SUGGESTED FEATURE SETS FOR EXPERIMENTS")
    print(f"{'─'*70}")
    
    for n in [5, 10, 15, 20]:
        top_n = avg_importance.head(n)
        category_counts = top_n['category'].value_counts()
        print(f"\nTop {n} Features:")
        print(f"  Features: {list(top_n['feature'].values)}")
        print(f"  Breakdown: ", end="")
        for cat, count in category_counts.items():
            print(f"{cat}={count}, ", end="")
        print(f"  Cumulative: {top_n['cumulative_pct'].iloc[-1]:.1f}%")
    
    print(f"\n{'─'*70}\n")
    
    # Save results
    output_file = 'feature_importance_rankings.csv'
    avg_importance.to_csv(output_file, index=False)
    print(f"✅ Results saved to: {output_file}")
    
    # Save detailed results (per stock)
    detailed_file = 'feature_importance_detailed.csv'
    combined_df.to_csv(detailed_file, index=False)
    print(f"✅ Detailed results saved to: {detailed_file}")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("1. Review top features and category breakdown")
    print("2. Run feature selection experiments with 5, 10, 15, 20, 25 features")
    print("3. Compare accuracy vs. overfitting for each feature set")
    print("4. Select optimal feature count (target: 10-15 features)")
    print("5. Document final feature set with justification")
    print(f"{'='*70}\n")
    
else:
    print("\n❌ NO RESULTS - Check errors above")

print("✅ Feature importance analysis complete!")
