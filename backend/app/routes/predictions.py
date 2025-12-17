"""
Prediction Routes
=================
API endpoints for stock predictions.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List
from datetime import datetime
import time

from app.models.schemas import (
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PortfolioRecommendation,
    SignalType
)
from app.services.prediction_service import prediction_service
from app.config import settings

router = APIRouter()


@router.get("/{ticker}", response_model=PredictionResponse)
async def get_prediction(
    ticker: str,
    lookback_days: int = Query(180, ge=30, le=365, description="Days of historical data")
):
    """
    Generate prediction for a single stock.
    
    - **ticker**: Stock symbol (e.g., WFC, AAPL, GOOGL)
    - **lookback_days**: Days of historical data for training (default: 180)
    
    Returns a BUY/SELL/HOLD signal with confidence level.
    """
    ticker = ticker.upper()
    
    try:
        result = prediction_service.generate_prediction(ticker, lookback_days)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/batch", response_model=BatchPredictionResponse)
async def get_batch_predictions(request: BatchPredictionRequest):
    """
    Generate predictions for multiple stocks.
    
    Useful for getting a portfolio-wide view of predictions.
    """
    start_time = time.time()
    predictions = []
    
    for ticker in request.tickers:
        try:
            pred = prediction_service.generate_prediction(
                ticker.upper(), 
                request.lookback_days
            )
            predictions.append(pred)
        except Exception as e:
            print(f"Warning: Failed to predict {ticker}: {e}")
            continue
    
    processing_time = time.time() - start_time
    
    return BatchPredictionResponse(
        predictions=predictions,
        generated_at=datetime.now(),
        total_count=len(predictions),
        processing_time_seconds=round(processing_time, 2)
    )


@router.get("/portfolio/recommendations", response_model=PortfolioRecommendation)
async def get_portfolio_recommendations(
    include_tier3: bool = Query(False, description="Include low-accuracy tier 3 stocks")
):
    """
    Get ranked portfolio recommendations across all tracked stocks.
    
    Stocks are grouped by signal strength:
    - **STRONG_BUY**: High confidence buy signals (>70%)
    - **BUY**: Moderate confidence buy signals (60-70%)
    - **HOLD**: Low confidence signals
    - **SELL**: Moderate confidence sell signals
    - **STRONG_SELL**: High confidence sell signals
    """
    tickers = settings.tier1_stocks + settings.tier2_stocks
    if include_tier3:
        tickers += settings.tier3_stocks
    
    all_predictions = []
    
    for ticker in tickers:
        try:
            pred = prediction_service.generate_prediction(ticker)
            all_predictions.append(pred)
        except Exception as e:
            print(f"Warning: Failed to predict {ticker}: {e}")
            continue
    
    # Sort by confidence
    all_predictions.sort(key=lambda x: x.confidence, reverse=True)
    
    # Group by signal type
    strong_buys = [p for p in all_predictions if p.signal == SignalType.STRONG_BUY]
    buys = [p for p in all_predictions if p.signal == SignalType.BUY]
    holds = [p for p in all_predictions if p.signal == SignalType.HOLD]
    sells = [p for p in all_predictions if p.signal == SignalType.SELL]
    strong_sells = [p for p in all_predictions if p.signal == SignalType.STRONG_SELL]
    
    # Generate summary
    if strong_buys:
        summary = f"Strong buy opportunities: {', '.join([p.ticker for p in strong_buys])}"
    elif buys:
        summary = f"Moderate buy signals: {', '.join([p.ticker for p in buys])}"
    elif sells or strong_sells:
        summary = "Consider reducing positions in underperforming stocks"
    else:
        summary = "No strong signals - hold current positions"
    
    return PortfolioRecommendation(
        strong_buys=strong_buys,
        buys=buys,
        holds=holds,
        sells=sells,
        strong_sells=strong_sells,
        generated_at=datetime.now(),
        summary=summary
    )


@router.get("/tickers/available")
async def get_available_tickers():
    """
    Get list of available tickers grouped by accuracy tier.
    """
    return {
        "tier1": {
            "stocks": settings.tier1_stocks,
            "description": "High accuracy (60-70%)",
            "recommended": True
        },
        "tier2": {
            "stocks": settings.tier2_stocks,
            "description": "Moderate accuracy (50-58%)",
            "recommended": True
        },
        "tier3": {
            "stocks": settings.tier3_stocks,
            "description": "Low accuracy (38-43%)",
            "recommended": False,
            "warning": "Use with caution - limited predictive value"
        }
    }

