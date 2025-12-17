"""
Data Routes
===========
API endpoints for stock data and feature information.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta

from app.models.schemas import StockDataPoint, PoliticianTrade, FeatureImportance, ModelInfo
from app.services.data_service import data_service
from app.services.model_service import model_service

router = APIRouter()


@router.get("/stock/{ticker}")
async def get_stock_data(
    ticker: str,
    days: int = Query(30, ge=1, le=365, description="Number of days of data")
):
    """
    Get recent stock price data.
    
    Returns OHLCV data for the specified number of days.
    """
    try:
        data = data_service.get_stock_data(ticker.upper(), days)
        return {
            "ticker": ticker.upper(),
            "data": data,
            "count": len(data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/politician-trades/{ticker}")
async def get_politician_trades(
    ticker: str,
    days: int = Query(90, ge=1, le=365, description="Number of days to look back")
):
    """
    Get recent politician trading activity for a stock.
    
    Returns congressional trading disclosures from the STOCK Act.
    """
    try:
        trades = data_service.get_politician_trades(ticker.upper(), days)
        return {
            "ticker": ticker.upper(),
            "trades": trades,
            "count": len(trades),
            "period_days": days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/importance")
async def get_feature_importance(
    ticker: Optional[str] = Query(None, description="Get importance for specific ticker model")
):
    """
    Get feature importance rankings.
    
    Shows which features contribute most to predictions.
    """
    try:
        if ticker:
            importance = model_service.get_feature_importance(ticker.upper())
        else:
            importance = model_service.get_global_feature_importance()
        
        return {
            "ticker": ticker.upper() if ticker else "global",
            "features": importance,
            "categories": {
                "technical": "Technical indicators (SMA, RSI, MACD, etc.)",
                "sentiment": "News sentiment scores",
                "politician": "Congressional trading signals",
                "market": "Market context (SPY, VIX, etc.)"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/{ticker}")
async def get_model_info(ticker: str):
    """
    Get information about the trained model for a ticker.
    """
    try:
        info = model_service.get_model_info(ticker.upper())
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validation-results")
async def get_validation_results():
    """
    Get historical validation results from experiments.
    """
    try:
        results = data_service.get_validation_results()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-status")
async def get_market_status():
    """
    Check if the market is open and get current time context.
    """
    now = datetime.now()
    
    # Simple market hours check (9:30 AM - 4:00 PM ET, weekdays)
    is_weekday = now.weekday() < 5
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_market_hours = is_weekday and market_open <= now <= market_close
    
    return {
        "current_time": now.isoformat(),
        "is_market_open": is_market_hours,
        "market_open_time": "09:30 ET",
        "market_close_time": "16:00 ET",
        "recommendation": "Run predictions after market close for next-day signals" if is_market_hours else "Generate predictions for next trading day"
    }

