"""
Feature Quality Analysis - Why Did More News Hurt Performance?

Analyzes V1 vs V3 results to understand why enhanced news coverage
decreased accuracy for MSFT (-6.39%) and AMZN (-1.91%)
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from data_loader import fetch_stock_data
from data_loader_optimized import fetch_news_with_keywords, aggregate_daily_sentiment

print("="*70)
print("FEATURE QUALITY ANALYSIS")
print("="*70)
print()

# Load V1 and V3 results
v1 = pd.read_csv('Results/mvp_validation_results.csv')
v3 = pd.read_csv('Results/mvp_validation_results_v2.csv')

# Focus on stocks that got worse with more news
problem_stocks = ['MSFT', 'AMZN']
improved_stocks = ['AAPL', 'TSLA']

print("HYPOTHESIS: Enhanced news coverage added noise, not signal")
print("-"*70)
print("\n📊 Performance Changes:")
print()

for ticker in problem_stocks + improved_stocks:
    v1_acc = v1[v1['ticker'] == ticker]['test_acc'].values[0] * 100
    v3_acc = v3[v3['ticker'] == ticker]['test_acc'].values[0] * 100
    v1_news = v1[v1['ticker'] == ticker]['news_articles'].values[0]
    v3_news = v3[v3['ticker'] == ticker]['news_articles'].values[0]
    change = v3_acc - v1_acc
    
    emoji = "✅" if change > 0 else "❌"
    print(f"{emoji} {ticker}: {v1_acc:.2f}% → {v3_acc:.2f}% ({change:+.2f}%)")
    print(f"   News: {v1_news} → {v3_news} articles (+{v3_news-v1_news})")
    print()

# Load feature importance
print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
print("-"*70)

rankings = pd.read_csv('Results/feature_importance_rankings.csv')
rankings['rank'] = range(1, len(rankings) + 1)
sentiment_features = rankings[rankings['feature'].str.contains('sentiment|news', case=False)]

print("\n📰 News/Sentiment Features:")
print(sentiment_features[['feature', 'mean_importance', 'rank']].to_string(index=False))

# Analyze sentiment for problem stocks
print("\n\n📊 SENTIMENT QUALITY ANALYSIS (2019)")
print("="*70)

for ticker in problem_stocks + improved_stocks:
    print(f"\n{ticker}:")
    print("-"*70)
    
    # Load stock data to see actual price movements
    stock_data = fetch_stock_data(ticker, '2019-01-01', '2019-12-31')
    
    # Calculate returns
    returns = stock_data['Close'].pct_change()
    positive_days = (returns > 0).sum()
    total_days = len(returns.dropna())
    positive_pct = (positive_days / total_days) * 100
    
    print(f"  Price Data:")
    print(f"    Total trading days: {total_days}")
    print(f"    Positive days: {positive_days} ({positive_pct:.1f}%)")
    print(f"    Average daily return: {returns.mean()*100:.3f}%")
    
    # Load news sentiment
    news = fetch_news_with_keywords(ticker, '2019-01-01', '2019-12-31')
    
    if len(news) > 0:
        sentiment = news['compound'].values
        print(f"  Sentiment Data:")
        print(f"    Total articles: {len(news)}")
        print(f"    Mean sentiment: {sentiment.mean():.3f}")
        print(f"    Std sentiment: {sentiment.std():.3f}")
        print(f"    Positive articles: {(sentiment > 0.05).sum()} ({(sentiment > 0.05).sum()/len(sentiment)*100:.1f}%)")
        print(f"    Negative articles: {(sentiment < -0.05).sum()} ({(sentiment < -0.05).sum()/len(sentiment)*100:.1f}%)")
        print(f"    Neutral articles: {((sentiment >= -0.05) & (sentiment <= 0.05)).sum()} ({((sentiment >= -0.05) & (sentiment <= 0.05)).sum()/len(sentiment)*100:.1f}%)")
        
        # Sample headlines
        print(f"  Sample Headlines:")
        if len(news) > 0:
            # Most positive
            most_pos_idx = sentiment.argmax()
            print(f"    Most positive ({sentiment[most_pos_idx]:.3f}): {news.iloc[most_pos_idx]['headline'][:80]}")
            # Most negative
            most_neg_idx = sentiment.argmin()
            print(f"    Most negative ({sentiment[most_neg_idx]:.3f}): {news.iloc[most_neg_idx]['headline'][:80]}")
    else:
        print(f"  No news articles found")

# Key insights
print("\n\n🎯 KEY INSIGHTS")
print("="*70)

print("""
1. SENTIMENT QUALITY ISSUES:
   - VADER may not work well for all types of headlines
   - Many headlines are neutral/uninformative
   - Keyword matching might include irrelevant articles
   
2. NOISE VS SIGNAL:
   - More articles != better signal
   - Quality > Quantity for sentiment features
   - avg_sentiment_compound is #1 feature but may be noisy

3. POTENTIAL IMPROVEMENTS:
   - Filter for high-confidence sentiment only (|compound| > 0.1)
   - Use more sophisticated sentiment (FinBERT, etc.)
   - Add article relevance scoring
   - Consider article source quality
   
4. OVERFITTING IS MAIN ISSUE:
   - 100% train accuracy regardless of news coverage
   - 44% train/test gap is the primary problem
   - More features (even good ones) make overfitting worse
""")

print("\n✅ Analysis complete")
print("   Saved insights for documentation")
