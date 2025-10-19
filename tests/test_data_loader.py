"""
Unit tests for data_loader module.

To run tests:
    pytest tests/test_data_loader.py
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import sys
sys.path.append('../src')

from src.data_loader import (
    fetch_stock_data,
    fetch_news_sentiment,
    aggregate_daily_sentiment
)


class TestFetchStockData:
    """Test cases for fetch_stock_data function."""
    
    def test_fetch_stock_data_valid_ticker(self):
        """Test fetching data for a valid ticker."""
        ticker = "AAPL"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        df = fetch_stock_data(ticker, start_date, end_date)
        
        # Check that we got a DataFrame
        assert isinstance(df, pd.DataFrame)
        
        # Check that it's not empty
        assert len(df) > 0
        
        # Check required columns exist
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_fetch_stock_data_invalid_ticker(self):
        """Test fetching data for an invalid ticker."""
        ticker = "INVALID_TICKER_12345"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        with pytest.raises(ValueError):
            fetch_stock_data(ticker, start_date, end_date)
    
    def test_fetch_stock_data_date_range(self):
        """Test that returned data is within the specified date range."""
        ticker = "MSFT"
        start_date = "2023-01-01"
        end_date = "2023-01-31"
        
        df = fetch_stock_data(ticker, start_date, end_date)
        
        # Convert to datetime for comparison
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Check date range
        assert df['Date'].min() >= pd.to_datetime(start_date)
        assert df['Date'].max() <= pd.to_datetime(end_date)


class TestFetchNewsSentiment:
    """Test cases for fetch_news_sentiment function."""
    
    def test_fetch_news_sentiment_returns_dataframe(self):
        """Test that the function returns a DataFrame."""
        ticker = "AAPL"
        df = fetch_news_sentiment(ticker, days_back=7)
        
        assert isinstance(df, pd.DataFrame)
    
    def test_fetch_news_sentiment_columns(self):
        """Test that returned DataFrame has required columns."""
        ticker = "AAPL"
        df = fetch_news_sentiment(ticker, days_back=7)
        
        required_columns = ['date', 'headline', 'sentiment_compound']
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_sentiment_score_range(self):
        """Test that sentiment scores are in valid range [-1, 1]."""
        ticker = "AAPL"
        df = fetch_news_sentiment(ticker, days_back=7)
        
        if len(df) > 0:
            assert df['sentiment_compound'].between(-1, 1).all(), \
                "Sentiment scores should be between -1 and 1"


class TestAggregateDailySentiment:
    """Test cases for aggregate_daily_sentiment function."""
    
    def test_aggregate_empty_dataframe(self):
        """Test aggregation with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['date', 'sentiment_compound', 'headline'])
        result = aggregate_daily_sentiment(empty_df)
        
        assert isinstance(result, pd.DataFrame)
        assert 'date' in result.columns
        assert 'avg_sentiment' in result.columns or 'avg_sentiment_compound' in result.columns
    
    def test_aggregate_groups_by_date(self):
        """Test that aggregation groups by date correctly."""
        # Create sample data
        data = {
            'date': ['2023-01-01', '2023-01-01', '2023-01-02'],
            'sentiment_compound': [0.5, 0.3, -0.2],
            'sentiment_positive': [0.6, 0.4, 0.1],
            'sentiment_negative': [0.1, 0.2, 0.5],
            'headline': ['News 1', 'News 2', 'News 3']
        }
        df = pd.DataFrame(data)
        
        result = aggregate_daily_sentiment(df)
        
        # Should have 2 unique dates
        assert len(result) == 2
        
        # Check that news count is correct
        assert 'news_count' in result.columns
        first_date_count = result[result['date'] == '2023-01-01']['news_count'].values[0]
        assert first_date_count == 2


# Pytest fixtures
@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'Date': dates,
        'Open': [100 + i for i in range(100)],
        'High': [105 + i for i in range(100)],
        'Low': [95 + i for i in range(100)],
        'Close': [102 + i for i in range(100)],
        'Volume': [1000000 + i * 1000 for i in range(100)]
    }
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
