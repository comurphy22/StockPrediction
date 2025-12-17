"""
Backtest Routes
===============
API endpoints for backtesting trading strategies.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from app.models.schemas import BacktestRequest, BacktestResult
from app.services.backtest_service import backtest_service

router = APIRouter()


@router.post("/run", response_model=BacktestResult)
async def run_backtest(request: BacktestRequest):
    """
    Run a backtest for a trading strategy on historical data.
    
    - **ticker**: Stock symbol to backtest
    - **start_date**: Backtest start date (YYYY-MM-DD)
    - **end_date**: Backtest end date (YYYY-MM-DD)
    - **initial_capital**: Starting capital (default: $10,000)
    - **transaction_cost**: Transaction cost as decimal (default: 0.1%)
    
    Returns performance metrics including Sharpe ratio, win rate, and excess returns.
    """
    try:
        result = backtest_service.run_backtest(
            ticker=request.ticker.upper(),
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            transaction_cost=request.transaction_cost
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@router.get("/results/{ticker}")
async def get_cached_backtest(ticker: str):
    """
    Get cached backtest results for a ticker if available.
    """
    # Try to load from results folder
    try:
        result = backtest_service.get_cached_result(ticker.upper())
        if result:
            return result
        raise HTTPException(status_code=404, detail=f"No cached backtest for {ticker}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_backtest_summary():
    """
    Get summary of all available backtest results.
    """
    try:
        summary = backtest_service.get_all_results_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

