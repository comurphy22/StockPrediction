"""
Stock Prediction API - FastAPI Backend
=======================================
RESTful API for ML-powered stock predictions using politician trading signals.
"""

import sys
from pathlib import Path

# Add backend directory to path for imports
BACKEND_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routes import predictions, backtest, data
from app.services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    print("🚀 Starting Stock Prediction API...")
    print("📊 Pre-loading models for tier 1 stocks...")
    
    # Pre-load models for high-accuracy stocks
    tier1_stocks = ['WFC', 'PFE', 'BABA']
    for ticker in tier1_stocks:
        try:
            model_service.get_or_train_model(ticker)
            print(f"   ✓ {ticker} model ready")
        except Exception as e:
            print(f"   ✗ {ticker} model failed: {e}")
    
    print("✅ API ready to serve predictions!")
    yield
    print("👋 Shutting down API...")


app = FastAPI(
    title="Stock Prediction API",
    description="ML-powered stock predictions using politician trading signals, news sentiment, and technical indicators.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Next.js dev server
        "http://127.0.0.1:3000",
        "http://localhost:5173",      # Vite dev server (alternative)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtest"])
app.include_router(data.router, prefix="/api/data", tags=["Data"])


@app.get("/")
def root():
    """API root endpoint."""
    return {
        "name": "Stock Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "models_cached": len(model_service.model_cache),
        "available_tickers": ['WFC', 'PFE', 'BABA', 'NFLX', 'GOOGL', 'FDX', 'NVDA', 'TSLA']
    }

