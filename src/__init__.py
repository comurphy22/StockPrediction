"""
Stock Movement Prediction with News Sentiment and Politician Position Signals

This package provides tools for predicting stock price movements using:
- Technical indicators (SMA, RSI, MACD)
- News sentiment analysis
- Politician trading signals
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main functions for easy access
from .data_loader import (
    fetch_stock_data,
    fetch_politician_trades,
    fetch_news_sentiment,
    aggregate_daily_sentiment
)

from .feature_engineering import (
    create_features,
    handle_missing_values
)

from .model import (
    train_model,
    evaluate_model,
    backtest_strategy
)

__all__ = [
    'fetch_stock_data',
    'fetch_politician_trades',
    'fetch_news_sentiment',
    'aggregate_daily_sentiment',
    'create_features',
    'handle_missing_values',
    'train_model',
    'evaluate_model',
    'backtest_strategy',
]
