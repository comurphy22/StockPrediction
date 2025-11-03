"""
Feature engineering for stock prediction model.
Creates technical indicators and prepares features for modeling.
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator


def create_features(df: pd.DataFrame, 
                   sentiment_df: pd.DataFrame = None,
                   politician_df: pd.DataFrame = None) -> tuple:
    """
    Create features from stock data, including technical indicators.
    Optionally merge in sentiment and politician trade data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Stock data DataFrame with columns: Date, Open, High, Low, Close, Volume
    sentiment_df : pd.DataFrame, optional
        Daily sentiment scores from news
    politician_df : pd.DataFrame, optional
        Politician trading data
    
    Returns:
    --------
    tuple: (X, y)
        X: Feature DataFrame
        y: Target variable (1 if next day close > today's close, 0 otherwise)
    """
    # Make a copy to avoid modifying the original
    features_df = df.copy()
    
    # Ensure Date column is datetime
    if 'Date' in features_df.columns:
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        features_df = features_df.sort_values('Date').reset_index(drop=True)
    
    # === TECHNICAL INDICATORS ===
    
    # Simple Moving Averages
    print("Creating SMA indicators...")
    sma_10 = SMAIndicator(close=features_df['Close'], window=10)
    sma_20 = SMAIndicator(close=features_df['Close'], window=20)
    sma_50 = SMAIndicator(close=features_df['Close'], window=50)
    
    features_df['SMA_10'] = sma_10.sma_indicator()
    features_df['SMA_20'] = sma_20.sma_indicator()
    features_df['SMA_50'] = sma_50.sma_indicator()
    
    # Relative Strength Index
    print("Creating RSI indicator...")
    rsi = RSIIndicator(close=features_df['Close'], window=14)
    features_df['RSI'] = rsi.rsi()
    
    # MACD
    print("Creating MACD indicators...")
    macd = MACD(close=features_df['Close'])
    features_df['MACD'] = macd.macd()
    features_df['MACD_signal'] = macd.macd_signal()
    features_df['MACD_diff'] = macd.macd_diff()
    
    # Price-based features
    features_df['Price_change'] = features_df['Close'].pct_change()
    features_df['Volume_change'] = features_df['Volume'].pct_change()
    
    # High-Low spread
    features_df['HL_spread'] = (features_df['High'] - features_df['Low']) / features_df['Close']
    
    # Moving average crossovers
    features_df['SMA_10_20_cross'] = (features_df['SMA_10'] - features_df['SMA_20']) / features_df['Close']
    features_df['SMA_20_50_cross'] = (features_df['SMA_20'] - features_df['SMA_50']) / features_df['Close']
    
    # === MERGE SENTIMENT DATA (if provided) ===
    if sentiment_df is not None and not sentiment_df.empty:
        print("Merging sentiment data...")
        # Ensure date columns are compatible
        if 'Date' in features_df.columns:
            merge_key = 'Date'
            if 'date' in sentiment_df.columns:
                sentiment_df = sentiment_df.rename(columns={'date': 'Date'})
            sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
            
            # Remove timezone info to avoid merge conflicts
            if hasattr(features_df['Date'].dtype, 'tz') and features_df['Date'].dtype.tz is not None:
                features_df['Date'] = features_df['Date'].dt.tz_localize(None)
            if hasattr(sentiment_df['Date'].dtype, 'tz') and sentiment_df['Date'].dtype.tz is not None:
                sentiment_df['Date'] = sentiment_df['Date'].dt.tz_localize(None)
        else:
            merge_key = 'date'
        
        features_df = features_df.merge(
            sentiment_df, 
            on=merge_key, 
            how='left'
        )
        
        # Fill missing sentiment values with neutral (0)
        sentiment_cols = [col for col in features_df.columns if 'sentiment' in col.lower()]
        for col in sentiment_cols:
            features_df[col] = features_df[col].fillna(0)
    else:
        print("No sentiment data provided - skipping sentiment features")
    
    # === MERGE POLITICIAN TRADE DATA (if provided) ===
    if politician_df is not None and not politician_df.empty:
        print("Merging politician trade data...")
        
        # Aggregate politician trades by date
        # Count number of buys and sells per day
        politician_df['date'] = pd.to_datetime(politician_df['date'])
        
        # Remove timezone info to avoid merge conflicts
        if hasattr(politician_df['date'].dtype, 'tz') and politician_df['date'].dtype.tz is not None:
            politician_df['date'] = politician_df['date'].dt.tz_localize(None)
        
        # Create buy/sell indicators
        politician_agg = politician_df.groupby('date').agg({
            'transaction_type': lambda x: sum(x.str.lower().str.contains('buy', na=False)),
            'amount': 'sum'
        }).reset_index()
        
        politician_agg.columns = ['Date', 'politician_buy_count', 'politician_trade_amount']
        
        # Ensure Date column in features_df is also timezone-naive
        if 'Date' in features_df.columns:
            if hasattr(features_df['Date'].dtype, 'tz') and features_df['Date'].dtype.tz is not None:
                features_df['Date'] = features_df['Date'].dt.tz_localize(None)
        
        # Merge with features
        features_df = features_df.merge(
            politician_agg,
            on='Date',
            how='left'
        )
        
        # Fill missing values with 0 (no trades)
        features_df['politician_buy_count'] = features_df['politician_buy_count'].fillna(0)
        features_df['politician_trade_amount'] = features_df['politician_trade_amount'].fillna(0)
        
        # === ADVANCED POLITICIAN FEATURES ===
        print("Creating advanced politician features...")
        try:
            from advanced_politician_features import create_advanced_politician_features
            
            # Create advanced features aligned with our date index
            # Pass features_df as stock_df (it has the Date column)
            adv_features = create_advanced_politician_features(
                features_df[['Date']] if 'Date' in features_df.columns else features_df,
                politician_df
            )
            
            # Merge advanced features
            if not adv_features.empty:
                # Merge on Date
                if 'Date' in adv_features.columns and 'Date' in features_df.columns:
                    features_df = features_df.merge(adv_features, on='Date', how='left')
                else:
                    features_df = pd.concat([features_df, adv_features], axis=1)
                print(f"      ✅ Added {len(adv_features.columns)-1 if 'Date' in adv_features.columns else len(adv_features.columns)} advanced politician features")
            else:
                print("      ⚠️  No advanced features generated (insufficient data)")
        except Exception as e:
            print(f"      ⚠️  Could not create advanced features: {e}")
    else:
        print("No politician trade data provided - skipping politician features")
    
    # === CREATE TARGET VARIABLE ===
    print("Creating target variable...")
    # Target: 1 if tomorrow's close is higher than today's close, 0 otherwise
    features_df['Next_Close'] = features_df['Close'].shift(-1)
    features_df['Target'] = (features_df['Next_Close'] > features_df['Close']).astype(int)
    
    # Drop the last row (no target available)
    features_df = features_df[:-1].copy()
    
    # === PREPARE X AND Y ===
    
    # Drop columns that shouldn't be features
    columns_to_drop = ['Date', 'Next_Close', 'Target', 'Open', 'High', 'Low', 'Close', 
                       'Adj Close', 'Volume', 'Dividends', 'Stock Splits']
    
    # Only drop columns that actually exist
    columns_to_drop = [col for col in columns_to_drop if col in features_df.columns]
    
    X = features_df.drop(columns=columns_to_drop)
    y = features_df['Target']
    
    # Keep the Date column for reference (separate from X)
    dates = features_df['Date'] if 'Date' in features_df.columns else None
    
    print(f"\nFeature engineering complete!")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"\nTarget distribution:\n{y.value_counts()}")
    print(f"Class balance: {y.mean():.2%} positive (price increase)")
    
    return X, y, dates


def handle_missing_values(X: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in feature DataFrame.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature DataFrame
    strategy : str
        Strategy for handling missing values:
        - 'drop': Drop rows with any missing values
        - 'ffill': Forward fill missing values
        - 'mean': Fill with column mean
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values handled
    """
    print(f"\nMissing values before handling:\n{X.isnull().sum()[X.isnull().sum() > 0]}")
    
    if strategy == 'drop':
        X_clean = X.dropna()
        print(f"Dropped {len(X) - len(X_clean)} rows with missing values")
    elif strategy == 'ffill':
        X_clean = X.fillna(method='ffill').fillna(0)
        print("Forward filled missing values")
    elif strategy == 'mean':
        X_clean = X.fillna(X.mean())
        print("Filled missing values with column means")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"Missing values after handling: {X_clean.isnull().sum().sum()}")
    
    return X_clean


if __name__ == "__main__":
    # Example usage
    from data_loader import fetch_stock_data
    
    print("=== Feature Engineering Example ===\n")
    
    # Fetch sample data
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Create features (baseline - no sentiment or politician data)
    X, y, dates = create_features(stock_data)
    
    # Handle missing values
    X_clean = handle_missing_values(X, strategy='drop')
    
    # Align y with cleaned X
    y_clean = y.loc[X_clean.index]
    
    print(f"\nFinal dataset shape: X={X_clean.shape}, y={y_clean.shape}")
    print(f"\nSample features:\n{X_clean.head()}")
