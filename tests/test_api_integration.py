"""
Test script to verify API integrations are working correctly.
Tests both NewsAPI and Quiver Quantitative API.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from data_loader import fetch_stock_data, fetch_news_sentiment, fetch_politician_trades
from datetime import datetime, timedelta

print("="*70)
print("API Integration Test Script")
print("="*70)

# Test configuration
TEST_TICKER = "AAPL"
TEST_DAYS_BACK = 7

print(f"\nTest Configuration:")
print(f"  Ticker: {TEST_TICKER}")
print(f"  Days back for news: {TEST_DAYS_BACK}")
print("="*70)

# Test 1: Stock Data (baseline - should always work)
print("\n[TEST 1] Fetching stock data...")
print("-"*70)
try:
    stock_data = fetch_stock_data(TEST_TICKER, "2024-10-01", "2024-11-01")
    print(f"✅ Stock data fetch: SUCCESS")
    print(f"   - Rows: {len(stock_data)}")
    print(f"   - Columns: {list(stock_data.columns)}")
except Exception as e:
    print(f"❌ Stock data fetch: FAILED")
    print(f"   Error: {str(e)}")

# Test 2: News Sentiment
print("\n[TEST 2] Fetching news sentiment...")
print("-"*70)
try:
    news_data = fetch_news_sentiment(TEST_TICKER, days_back=TEST_DAYS_BACK)
    
    if news_data.empty:
        print(f"⚠️  News sentiment fetch: WARNING - No data returned")
    else:
        print(f"✅ News sentiment fetch: SUCCESS")
        print(f"   - Articles: {len(news_data)}")
        print(f"   - Date range: {news_data['date'].min()} to {news_data['date'].max()}")
        print(f"   - Avg sentiment: {news_data['sentiment_compound'].mean():.3f}")
        print(f"   - Columns: {list(news_data.columns)}")
        
        # Show sample headlines
        print(f"\n   Sample headlines:")
        for idx, row in news_data.head(3).iterrows():
            sentiment_emoji = "😊" if row['sentiment_compound'] > 0 else "😟" if row['sentiment_compound'] < 0 else "😐"
            print(f"   {sentiment_emoji} [{row['sentiment_compound']:+.2f}] {row['headline'][:60]}...")
            
except Exception as e:
    print(f"❌ News sentiment fetch: FAILED")
    print(f"   Error: {str(e)}")
    import traceback
    traceback.print_exc()

# Test 3: Politician Trading Data
print("\n[TEST 3] Fetching politician trading data...")
print("-"*70)
try:
    politician_data = fetch_politician_trades(TEST_TICKER)
    
    if politician_data.empty:
        print(f"⚠️  Politician trades fetch: WARNING - No data returned")
        print(f"   This may be normal if no recent congressional trades for {TEST_TICKER}")
    else:
        print(f"✅ Politician trades fetch: SUCCESS")
        print(f"   - Trades: {len(politician_data)}")
        print(f"   - Date range: {politician_data['date'].min().date()} to {politician_data['date'].max().date()}")
        print(f"   - Columns: {list(politician_data.columns)}")
        
        # Show transaction breakdown
        if 'transaction_type' in politician_data.columns:
            print(f"\n   Transaction breakdown:")
            for trans_type, count in politician_data['transaction_type'].value_counts().items():
                print(f"   - {trans_type}: {count}")
        
        # Show sample trades
        print(f"\n   Sample recent trades:")
        for idx, row in politician_data.head(3).iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
            politician = row.get('politician_name', 'Unknown')
            trans_type = row.get('transaction_type', 'Unknown')
            amount = row.get('amount', 0)
            print(f"   - {date_str} | {politician[:20]:20s} | {trans_type:10s} | ${amount:,.0f}")
            
except Exception as e:
    print(f"❌ Politician trades fetch: FAILED")
    print(f"   Error: {str(e)}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print("\n✅ = Success | ⚠️  = Warning (no data) | ❌ = Failed")
print("\nIf all tests show ✅ or ⚠️ , the API integration is working correctly!")
print("If you see ❌, check:")
print("  1. API keys in src/config.py are correct")
print("  2. You have internet connection")
print("  3. API services are not down")
print("  4. You haven't exceeded API rate limits")
print("="*70)
