"""
Prediction Service
==================
Generates stock predictions using trained ML models.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd

# Add the project src directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from app.config import settings
from app.services.model_service import model_service
from app.models.schemas import (
    PredictionResponse, 
    SignalType, 
    DirectionType,
    DataSourceInfo
)


class PredictionService:
    """Service for generating stock predictions."""
    
    def __init__(self):
        # Import ML modules from existing code
        try:
            from data_loader import fetch_stock_data, fetch_politician_trades
            from feature_engineering import create_features, handle_missing_values
            
            self.fetch_stock_data = fetch_stock_data
            self.fetch_politician_trades = fetch_politician_trades
            self.create_features = create_features
            self.handle_missing_values = handle_missing_values
            
            # Try to load TOP_FEATURES from config
            try:
                from config import TOP_FEATURES
                self.top_features = TOP_FEATURES[:20] if TOP_FEATURES else None
            except ImportError:
                self.top_features = None
                
        except ImportError as e:
            print(f"Warning: Could not import ML modules: {e}")
            raise RuntimeError(f"Failed to load ML modules from src/: {e}")
    
    def generate_prediction(self, ticker: str, lookback_days: int = 180) -> PredictionResponse:
        """
        Generate a prediction for a stock.
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Days of historical data for training
            
        Returns:
            PredictionResponse with signal, confidence, and metadata
        """
        ticker = ticker.upper()
        
        # Get or train model
        model_data = model_service.get_or_train_model(ticker, lookback_days)
        model = model_data['model']
        features = model_data['features']
        
        # Fetch latest data for prediction
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        print(f"Fetching latest data for {ticker}...")
        stock_data = self.fetch_stock_data(ticker, start_date, end_date)
        
        try:
            trades_data = self.fetch_politician_trades(ticker)
            # Filter to recent trades
            if not trades_data.empty and 'date' in trades_data.columns:
                trades_data['date'] = pd.to_datetime(trades_data['date'])
                cutoff = datetime.now() - timedelta(days=90)
                trades_data = trades_data[trades_data['date'] >= cutoff]
        except Exception as e:
            print(f"Warning: Could not fetch politician trades: {e}")
            trades_data = pd.DataFrame()
        
        # Create features
        X, y, dates = self.create_features(stock_data, pd.DataFrame(), trades_data, ticker=ticker)
        
        # Select features to match training
        if self.top_features:
            available = [f for f in self.top_features if f in X.columns]
            X = X[available] if available else X
        
        # Align features with model
        for feat in features:
            if feat not in X.columns:
                X[feat] = 0
        X = X[features]
        
        # Get latest data point
        X_latest = X.iloc[[-1]]
        
        # Make prediction
        prediction = model.predict(X_latest)[0]
        probabilities = model.predict_proba(X_latest)[0]
        
        # Calculate confidence (probability of predicted class)
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        # Determine signal type
        signal = self._determine_signal(prediction, confidence)
        
        # Get current price
        current_price = float(stock_data.iloc[-1]['Close'])
        
        # Determine tier
        tier = self._get_ticker_tier(ticker)
        
        # Calculate next trading day
        next_trading_day = self._get_next_trading_day()
        
        return PredictionResponse(
            ticker=ticker,
            signal=signal,
            confidence=float(confidence),
            predicted_direction=DirectionType.UP if prediction == 1 else DirectionType.DOWN,
            current_price=current_price,
            prediction_date=datetime.now(),
            prediction_for=next_trading_day,
            data_sources=DataSourceInfo(
                stock_days=len(stock_data),
                politician_trades=len(trades_data),
                news_articles=0  # Could be expanded
            ),
            tier=tier
        )
    
    def _determine_signal(self, prediction: int, confidence: float) -> SignalType:
        """Determine the trading signal based on prediction and confidence."""
        if prediction == 1:  # UP
            if confidence >= settings.strong_signal_threshold:
                return SignalType.STRONG_BUY
            elif confidence >= settings.signal_threshold:
                return SignalType.BUY
            else:
                return SignalType.HOLD
        else:  # DOWN
            if confidence >= settings.strong_signal_threshold:
                return SignalType.STRONG_SELL
            elif confidence >= settings.signal_threshold:
                return SignalType.SELL
            else:
                return SignalType.HOLD
    
    def _get_ticker_tier(self, ticker: str) -> int:
        """Get the accuracy tier for a ticker."""
        if ticker in settings.tier1_stocks:
            return 1
        elif ticker in settings.tier2_stocks:
            return 2
        elif ticker in settings.tier3_stocks:
            return 3
        else:
            return 2  # Default to tier 2 for unknown stocks
    
    def _get_next_trading_day(self) -> str:
        """Get the next trading day (skip weekends)."""
        today = datetime.now()
        next_day = today + timedelta(days=1)
        
        # Skip weekends
        while next_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
            next_day += timedelta(days=1)
        
        return next_day.strftime('%Y-%m-%d')


# Singleton instance
prediction_service = PredictionService()

