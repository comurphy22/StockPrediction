"""
Daily Prediction Tracker - Log predictions for later validation

Runs model on specified tickers, generates predictions, and logs them
with timestamps for later comparison against actual outcomes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

try:
    from config import NEWS_API_KEY, QUIVER_API_KEY
except:
    NEWS_API_KEY = None
    QUIVER_API_KEY = None
    
from data_loader import (
    fetch_stock_data,
    fetch_politician_trades,
    fetch_news_sentiment,
    fetch_historical_news_kaggle,
    aggregate_daily_sentiment
)
from feature_engineering import create_features, handle_missing_values
from model_xgboost import train_xgboost_model

# Tickers to track (based on validation performance)
TRACKING_TICKERS = [
    'WFC',    # 70% accuracy - Tier 1
    'BABA',   # 68% accuracy - Tier 1
    'PFE',    # 61% accuracy - Tier 1
    'NFLX',   # 46% accuracy - Tier 2
    'GOOGL',  # 50% accuracy - Tier 2
]

# Output file
PREDICTIONS_LOG = 'results/daily_predictions_log.csv'

def get_current_price(ticker):
    """Fetch current/latest price for ticker."""
    try:
        # Get last 5 days to ensure we have latest close
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        
        if len(stock_data) > 0:
            latest_close = stock_data.iloc[-1]['Close']
            latest_date = stock_data.iloc[-1]['Date']
            return latest_close, latest_date
        return None, None
    except Exception as e:
        print(f"   Error fetching price for {ticker}: {e}")
        return None, None

def generate_prediction(ticker):
    """Generate prediction for a ticker."""
    try:
        print(f"\n{'─'*60}")
        print(f"Generating prediction for {ticker}...")
        print(f"{'─'*60}")
        
        # Get historical data (need more to account for missing values)
        # Fetch 1 year of data to ensure 100+ clean samples after dropping NaNs
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Fetch data
        print(f"   [1/5] Fetching stock data...")
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        
        # Get news
        print(f"   [2/5] Fetching news...")
        try:
            if NEWS_API_KEY:
                news_data = fetch_news_sentiment(ticker, days_back=7, api_key=NEWS_API_KEY)
            else:
                news_start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                news_data = fetch_historical_news_kaggle(ticker, news_start, end_date)
            news_sentiment = aggregate_daily_sentiment(news_data)
        except:
            news_sentiment = pd.DataFrame()
        
        # Get politician trades
        print(f"   [3/5] Fetching politician trades...")
        try:
            trades_data = fetch_politician_trades(ticker, api_key=QUIVER_API_KEY)
        except:
            trades_data = pd.DataFrame()
        
        # Create features
        print(f"   [4/5] Engineering features & training model...")
        X, y, dates = create_features(stock_data, news_sentiment, trades_data)
        
        # Use forward fill instead of drop for live predictions
        # (drop strategy removes too many samples for recent data)
        X_clean = handle_missing_values(X, strategy='ffill')
        y_clean = y.loc[X_clean.index]
        
        # Check if we have enough samples
        if len(X_clean) < 50:
            print(f"   ⚠️  Warning: Only {len(X_clean)} samples after cleaning (need 50+)")
            print(f"   Skipping {ticker} - insufficient data")
            return None
        
        # Check class balance
        class_counts = y_clean.value_counts()
        if len(class_counts) < 2:
            print(f"   ⚠️  Warning: Only one class in target variable")
            print(f"   Skipping {ticker} - imbalanced data")
            return None
        
        # Train model on all available data
        model = train_xgboost_model(X_clean, y_clean, verbose=False)
        
        # Get latest features for prediction
        X_latest = X.iloc[[-1]]
        
        # Make prediction
        print(f"   [5/5] Generating prediction...")
        prediction = model.predict(X_latest)[0]
        probability = model.predict_proba(X_latest)[0]
        confidence = probability[1]  # Probability of UP
        
        # Get current price
        current_price, price_date = get_current_price(ticker)
        
        return {
            'ticker': ticker,
            'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'price_date': price_date,
            'current_price': current_price,
            'predicted_direction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': confidence,
            'signal': 'BUY' if (prediction == 1 and confidence > 0.60) else 
                     'SELL' if (prediction == 0 and confidence < 0.40) else 'HOLD',
            'model_raw_prediction': int(prediction),
            'training_samples': len(X_clean),
            'features_used': len(X_clean.columns)
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def save_prediction(prediction_dict):
    """Save prediction to CSV log."""
    # Convert to DataFrame
    new_row = pd.DataFrame([prediction_dict])
    
    # Append to existing log or create new
    if os.path.exists(PREDICTIONS_LOG):
        existing = pd.read_csv(PREDICTIONS_LOG)
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row
    
    updated.to_csv(PREDICTIONS_LOG, index=False)
    print(f"   ✅ Saved to {PREDICTIONS_LOG}")

def display_summary(predictions):
    """Display summary of today's predictions."""
    print("\n" + "="*80)
    print("TODAY'S PREDICTIONS SUMMARY")
    print("="*80 + "\n")
    
    # Create DataFrame
    df = pd.DataFrame(predictions)
    
    # Sort by confidence (highest first)
    df = df.sort_values('confidence', ascending=False)
    
    print("Ranked by Confidence:\n")
    for idx, row in df.iterrows():
        signal_emoji = "🟢" if row['signal'] == 'BUY' else "🔴" if row['signal'] == 'SELL' else "⚪"
        print(f"  {signal_emoji} {row['ticker']:<6} "
              f"{row['predicted_direction']:<5} "
              f"Confidence: {row['confidence']*100:>5.1f}% "
              f"Signal: {row['signal']:<5} "
              f"Price: ${row['current_price']:.2f}")
    
    # Action summary
    print(f"\n\nACTION ITEMS:")
    buys = df[df['signal'] == 'BUY']
    sells = df[df['signal'] == 'SELL']
    
    if len(buys) > 0:
        print(f"  🟢 BUY SIGNALS: {', '.join(buys['ticker'].tolist())}")
    if len(sells) > 0:
        print(f"  🔴 SELL SIGNALS: {', '.join(sells['ticker'].tolist())}")
    if len(buys) == 0 and len(sells) == 0:
        print(f"  ⚪ NO STRONG SIGNALS - HOLD")
    
    print(f"\n\n📝 All predictions logged to: {PREDICTIONS_LOG}")
    print("="*80)

print("="*80)
print("📊 DAILY PREDICTION TRACKER")
print("="*80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Tracking {len(TRACKING_TICKERS)} tickers: {', '.join(TRACKING_TICKERS)}")
print("="*80 + "\n")

# Generate predictions for all tickers
predictions = []
for ticker in TRACKING_TICKERS:
    result = generate_prediction(ticker)
    if result:
        predictions.append(result)
        save_prediction(result)
        time.sleep(1)  # Brief pause between tickers

# Display summary
if predictions:
    display_summary(predictions)
    
    print("\n" + "="*80)
    print("TRACKING INSTRUCTIONS")
    print("="*80)
    print("""
1. Record these predictions in LIVE_TRADING_LOG.md
2. Tomorrow, check actual outcomes and update the log
3. Calculate: (Tomorrow's Close - Today's Close) / Today's Close
4. Mark predictions as ✅ CORRECT or ❌ WRONG
5. Track win rate over time to validate model

Example tracking format:
- WFC: Predicted UP at $84.28 → Outcome: $85.11 (+0.98%) ✅ CORRECT
""")
    print("="*80)

print("\n✅ Prediction tracking complete!")

