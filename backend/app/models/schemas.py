"""
Pydantic Schemas
================
Request and response models for API endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class SignalType(str, Enum):
    """Trading signal types."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class DirectionType(str, Enum):
    """Predicted price direction."""
    UP = "UP"
    DOWN = "DOWN"


# ============== Prediction Schemas ==============

class PredictionRequest(BaseModel):
    """Request for stock prediction."""
    ticker: str = Field(..., description="Stock ticker symbol", example="WFC")
    lookback_days: int = Field(180, description="Days of historical data for training", ge=30, le=365)


class DataSourceInfo(BaseModel):
    """Information about data sources used."""
    stock_days: int = Field(..., description="Number of stock data days")
    politician_trades: int = Field(..., description="Number of politician trades found")
    news_articles: int = Field(0, description="Number of news articles analyzed")


class PredictionResponse(BaseModel):
    """Single stock prediction response."""
    ticker: str
    signal: SignalType
    confidence: float = Field(..., ge=0, le=1, description="Model confidence (0-1)")
    predicted_direction: DirectionType
    current_price: float
    prediction_date: datetime
    prediction_for: str = Field(..., description="Date this prediction is for (next trading day)")
    data_sources: DataSourceInfo
    tier: int = Field(..., description="Stock tier (1=high accuracy, 3=low)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "WFC",
                "signal": "BUY",
                "confidence": 0.72,
                "predicted_direction": "UP",
                "current_price": 45.67,
                "prediction_date": "2024-01-15T16:30:00",
                "prediction_for": "2024-01-16",
                "data_sources": {
                    "stock_days": 180,
                    "politician_trades": 23,
                    "news_articles": 45
                },
                "tier": 1
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for multiple stock predictions."""
    tickers: List[str] = Field(..., description="List of ticker symbols")
    lookback_days: int = Field(180, ge=30, le=365)


class BatchPredictionResponse(BaseModel):
    """Multiple stock predictions response."""
    predictions: List[PredictionResponse]
    generated_at: datetime
    total_count: int
    processing_time_seconds: float


class PortfolioRecommendation(BaseModel):
    """Portfolio-level recommendations."""
    strong_buys: List[PredictionResponse]
    buys: List[PredictionResponse]
    holds: List[PredictionResponse]
    sells: List[PredictionResponse]
    strong_sells: List[PredictionResponse]
    generated_at: datetime
    summary: str


# ============== Backtest Schemas ==============

class BacktestRequest(BaseModel):
    """Request for backtesting."""
    ticker: str
    start_date: str = Field(..., example="2019-01-01")
    end_date: str = Field(..., example="2019-12-31")
    initial_capital: float = Field(10000.0, ge=1000)
    transaction_cost: float = Field(0.001, ge=0, le=0.1, description="Transaction cost as decimal")


class BacktestResult(BaseModel):
    """Backtest results."""
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    buy_hold_return: float
    excess_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    accuracy: float


# ============== Data Schemas ==============

class StockDataPoint(BaseModel):
    """Single stock data point."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class PoliticianTrade(BaseModel):
    """Politician trade record."""
    date: str
    politician: str
    transaction_type: str  # BUY or SELL
    amount: str
    ticker: str


class FeatureImportance(BaseModel):
    """Feature importance ranking."""
    feature: str
    importance: float
    category: str  # technical, sentiment, politician, market


class ModelInfo(BaseModel):
    """Information about a trained model."""
    ticker: str
    trained_at: datetime
    lookback_days: int
    num_features: int
    top_features: List[FeatureImportance]
    cached: bool

