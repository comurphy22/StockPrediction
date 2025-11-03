"""
Multi-Year Hypothesis Validation: Do Sentiment + Politician Trading Features Improve Stock Prediction?

This script tests whether adding news sentiment + congressional trading data improves prediction
accuracy compared to technical indicators alone.

Tests across multiple years (2017-2019) for robustness
Tests on multiple tickers: AAPL, MSFT, GOOGL, JPM, NVDA
Compares: Technical-only vs. +Sentiment vs. +Both features
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
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MULTI-YEAR HYPOTHESIS VALIDATION: Multi-Signal Stock Prediction")
print("="*70)
print("\nResearch Question:")
print("Do news sentiment + politician trading signals improve stock prediction?")
print("\nMethodology:")
print("- Test across multiple years (2017-2019) for robustness")
print("- Compare 3 models: Technical-only, +Sentiment, +Sentiment+Politician")
print("- Test on 5 major stocks")
print("- Use 80/20 train/test split per year")
print("- Measure consistency across years")
print("="*70)

# Test configuration
# Using tickers with consistent 2017-2019 coverage in the Kaggle dataset
# These tickers have 500+ articles per year across all 3 test years
TICKERS = ['NFLX', 'NVDA', 'BABA', 'QCOM', 'MU']

# Multi-year validation (avoiding COVID period)
TEST_PERIODS = [
    ('2017-01-01', '2017-12-31', '2017'),
    ('2018-01-01', '2018-12-31', '2018'),
    ('2019-01-01', '2019-12-31', '2019'),
]

print(f"\n📅 Test Periods: 2017, 2018, 2019 (3 years, pre-COVID)")
print(f"   Testing consistency across different market conditions")
print("="*70)

# Store results by year
all_results = {}

# Loop over years
for start_date, end_date, year_label in TEST_PERIODS:
    print(f'\n\n{"#"*70}')
    print(f'# YEAR: {year_label} ({start_date} to {end_date})')
    print(f'{"#"*70}\n')
    
    year_results = {}
    
    for ticker in TICKERS:
        print(f'\n{"="*70}')
        print(f'{ticker} - {year_label}')
        print("="*70)
        
        try:
            # Fetch data
            print(f"[1/8] Fetching stock data...")
            stock_data = fetch_stock_data(ticker, start_date, end_date)
            print(f"      ✅ {len(stock_data)} days")
            
            print(f"[2/8] Fetching news sentiment...")
            news_data = fetch_historical_news_kaggle(ticker, start_date, end_date)
            if not news_data.empty:
                sentiment_data = aggregate_daily_sentiment(news_data)
                print(f"      ✅ {len(news_data)} news → {len(sentiment_data)} days with sentiment")
            else:
                sentiment_data = pd.DataFrame()
                print(f"      ⚠️  No news data")
            
            print(f"[3/8] Fetching politician trades...")
            politician_data = fetch_politician_trades(ticker)
            if not politician_data.empty:
                politician_data['date'] = pd.to_datetime(politician_data['date'])
                politician_data = politician_data[
                    (politician_data['date'] >= start_date) & 
                    (politician_data['date'] <= end_date)
                ]
                print(f"      ✅ {len(politician_data)} trades in range")
            else:
                print(f"      ⚠️  No trades")
            
            # TEST 1: Technical Only
            print(f"[4/8] Technical only...")
            X_tech, y_tech, _ = create_features(stock_data, None, None)
            X_tech = handle_missing_values(X_tech, strategy='drop')
            y_tech = y_tech.loc[X_tech.index]
            
            split = int(len(X_tech) * 0.8)
            model_tech = train_model(
                X_tech[:split], y_tech[:split],
                model_type='random_forest', n_estimators=100, max_depth=10,
                random_state=42, verbose=False
            )
            metrics_tech = evaluate_model(model_tech, X_tech[split:], y_tech[split:], verbose=False)
            print(f"      ✅ Acc={metrics_tech['accuracy']:.4f}, F1={metrics_tech['f1_score']:.4f}")
            
            # TEST 2: Technical + Sentiment
            print(f"[5/8] + Sentiment...")
            sentiment_use = sentiment_data if not sentiment_data.empty else None
            X_sent, y_sent, _ = create_features(stock_data, sentiment_use, None)
            X_sent = handle_missing_values(X_sent, strategy='drop')
            y_sent = y_sent.loc[X_sent.index]
            
            sent_features = [c for c in X_sent.columns if 'sentiment' in c.lower() or 'news' in c.lower()]
            
            split = int(len(X_sent) * 0.8)
            model_sent = train_model(
                X_sent[:split], y_sent[:split],
                model_type='random_forest', n_estimators=100, max_depth=10,
                random_state=42, verbose=False
            )
            metrics_sent = evaluate_model(model_sent, X_sent[split:], y_sent[split:], verbose=False)
            print(f"      ✅ Acc={metrics_sent['accuracy']:.4f}, F1={metrics_sent['f1_score']:.4f} ({len(sent_features)} sent features)")
            
            # TEST 3: All Features
            print(f"[6/8] + Sentiment + Politician...")
            X_full, y_full, _ = create_features(stock_data, sentiment_use, politician_data)
            X_full = handle_missing_values(X_full, strategy='drop')
            y_full = y_full.loc[X_full.index]
            
            pol_features = [c for c in X_full.columns if 'politician' in c.lower()]
            
            split = int(len(X_full) * 0.8)
            model_full = train_model(
                X_full[:split], y_full[:split],
                model_type='random_forest', n_estimators=100, max_depth=10,
                random_state=42, verbose=False
            )
            metrics_full = evaluate_model(model_full, X_full[split:], y_full[split:], verbose=False)
            print(f"      ✅ Acc={metrics_full['accuracy']:.4f}, F1={metrics_full['f1_score']:.4f} ({len(pol_features)} pol features)")
            
            # Calculate improvements
            sent_improvement = (metrics_sent['accuracy'] - metrics_tech['accuracy']) * 100
            full_improvement = (metrics_full['accuracy'] - metrics_tech['accuracy']) * 100
            
            # Store results
            year_results[ticker] = {
                'tech_acc': metrics_tech['accuracy'],
                'sent_acc': metrics_sent['accuracy'],
                'full_acc': metrics_full['accuracy'],
                'tech_f1': metrics_tech['f1_score'],
                'sent_f1': metrics_sent['f1_score'],
                'full_f1': metrics_full['f1_score'],
                'sent_improvement': sent_improvement,
                'full_improvement': full_improvement,
                'n_samples': len(X_full),
                'n_news': len(sentiment_data) if not sentiment_data.empty else 0,
                'n_trades': len(politician_data) if not politician_data.empty else 0
            }
            
            # Print result
            print(f"\n{'─'*70}")
            print(f"Technical:   {metrics_tech['accuracy']:.4f}")
            print(f"+ Sentiment: {metrics_sent['accuracy']:.4f} ({sent_improvement:+.2f}%)")
            print(f"+ Both:      {metrics_full['accuracy']:.4f} ({full_improvement:+.2f}%)")
            
            if full_improvement > 5:
                print("✅ EXCELLENT")
            elif full_improvement > 2:
                print("✅ GOOD")
            elif full_improvement > 0:
                print("⚠️  MARGINAL")
            else:
                print("❌ NO BENEFIT")
            print(f"{'─'*70}")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            year_results[ticker] = {'error': str(e)}
    
    # Store year results
    all_results[year_label] = year_results
    
    # Yearly summary
    successful = {k: v for k, v in year_results.items() if 'error' not in v}
    if successful:
        avg_sent = sum(d['sent_improvement'] for d in successful.values()) / len(successful)
        avg_full = sum(d['full_improvement'] for d in successful.values()) / len(successful)
        
        print(f"\n{'='*70}")
        print(f"SUMMARY FOR {year_label}")
        print(f"{'='*70}")
        print(f"Avg Sentiment Improvement: {avg_sent:+.2f}%")
        print(f"Avg Full Improvement:      {avg_full:+.2f}%")
        print(f"Successful tests:          {len(successful)}/{len(TICKERS)}")
        print(f"{'='*70}")

# ===== MULTI-YEAR ANALYSIS =====
print(f"\n\n{'='*70}")
print("MULTI-YEAR ANALYSIS: 2017-2019")
print(f"{'='*70}\n")

# Collect all successful results across years
all_successful = {}
for year, year_data in all_results.items():
    for ticker, data in year_data.items():
        if 'error' not in data:
            key = f"{ticker}_{year}"
            all_successful[key] = {**data, 'year': year, 'ticker': ticker}

if all_successful:
    # Overall averages
    overall_sent_avg = sum(d['sent_improvement'] for d in all_successful.values()) / len(all_successful)
    overall_full_avg = sum(d['full_improvement'] for d in all_successful.values()) / len(all_successful)
    
    print(f"📊 OVERALL RESULTS ({len(all_successful)} ticker-year combinations)")
    print(f"{'─'*70}")
    print(f"Average Sentiment Improvement: {overall_sent_avg:+.2f}%")
    print(f"Average Full Model Improvement: {overall_full_avg:+.2f}%")
    print(f"{'─'*70}\n")
    
    # Year-by-year comparison
    print(f"📅 YEAR-BY-YEAR BREAKDOWN")
    print(f"{'─'*70}")
    print(f"{'Year':<6} {'N':<4} {'Sent Δ':<10} {'Full Δ':<10} {'Status'}")
    print(f"{'─'*70}")
    
    for year in ['2017', '2018', '2019']:
        year_data = [d for d in all_successful.values() if d['year'] == year]
        if year_data:
            avg_sent = sum(d['sent_improvement'] for d in year_data) / len(year_data)
            avg_full = sum(d['full_improvement'] for d in year_data) / len(year_data)
            status = "✅" if avg_full > 2 else "⚠️ " if avg_full > 0 else "❌"
            print(f"{year:<6} {len(year_data):<4} {avg_sent:>+8.2f}% {avg_full:>+8.2f}%   {status}")
    
    print(f"{'─'*70}\n")
    
    # Stock-by-stock analysis
    print(f"📈 STOCK-BY-STOCK CONSISTENCY")
    print(f"{'─'*70}")
    print(f"{'Ticker':<8} {'2017':<10} {'2018':<10} {'2019':<10} {'Avg':<10}")
    print(f"{'─'*70}")
    
    for ticker in TICKERS:
        improvements = []
        line = f"{ticker:<8} "
        for year in ['2017', '2018', '2019']:
            year_ticker_data = [d for d in all_successful.values() 
                               if d['year'] == year and d['ticker'] == ticker]
            if year_ticker_data:
                imp = year_ticker_data[0]['full_improvement']
                improvements.append(imp)
                line += f"{imp:>+8.2f}%  "
            else:
                line += f"{'N/A':<10} "
        
        if improvements:
            avg = sum(improvements) / len(improvements)
            line += f"{avg:>+8.2f}%"
        else:
            line += "N/A"
        print(line)
    
    print(f"{'─'*70}\n")
    
    # Final verdict
    print(f"{'='*70}")
    print("🎯 FINAL VERDICT")
    print(f"{'='*70}\n")
    
    if overall_sent_avg > 3:
        print(f"✅ SENTIMENT SIGNALS: STRONG (+{overall_sent_avg:.2f}%)")
        print("   Sentiment analysis adds significant value")
    elif overall_sent_avg > 1:
        print(f"⚠️  SENTIMENT SIGNALS: MODERATE (+{overall_sent_avg:.2f}%)")
        print("   Sentiment shows promise but needs improvement")
    elif overall_sent_avg > 0:
        print(f"⚠️  SENTIMENT SIGNALS: MARGINAL (+{overall_sent_avg:.2f}%)")
        print("   Sentiment barely helps - investigate data quality")
    else:
        print(f"❌ SENTIMENT SIGNALS: NO BENEFIT ({overall_sent_avg:.2f}%)")
        print("   Sentiment not helping - may need different approach")
    
    print()
    
    if overall_full_avg > 5:
        print(f"✅ MULTI-SIGNAL MODEL: EXCELLENT (+{overall_full_avg:.2f}%)")
        print("   Strong evidence for multi-signal approach")
        print("\n🎯 RECOMMENDATION: Proceed with MVP")
        print("   • Integrate sentiment + politician features")
        print("   • Add advanced politician features (23 net indices)")
        print("   • Try XGBoost for better performance")
        print("   • Prepare for publication")
    elif overall_full_avg > 2:
        print(f"✅ MULTI-SIGNAL MODEL: PROMISING (+{overall_full_avg:.2f}%)")
        print("   Good evidence for multi-signal approach")
        print("\n🎯 RECOMMENDATION: Continue with caution")
        print("   • Proceed with MVP but monitor performance")
        print("   • Focus on feature engineering")
        print("   • Consider ensemble methods")
    elif overall_full_avg > 0:
        print(f"⚠️  MULTI-SIGNAL MODEL: MARGINAL (+{overall_full_avg:.2f}%)")
        print("   Weak evidence - needs investigation")
        print("\n🎯 RECOMMENDATION: Investigate further")
        print("   • Check data quality issues")
        print("   • Try different model architectures")
        print("   • Consider feature selection")
    else:
        print(f"❌ MULTI-SIGNAL MODEL: NO BENEFIT ({overall_full_avg:.2f}%)")
        print("   Signals not helping")
        print("\n🎯 RECOMMENDATION: Pivot strategy")
        print("   • Focus on technical features only")
        print("   • Explore alternative data sources")
        print("   • Re-evaluate hypothesis")
    
    print(f"\n{'='*70}")
    print("✅ Multi-year validation complete!")
    print(f"{'='*70}")

else:
    print("❌ NO SUCCESSFUL RESULTS - Check errors above")
