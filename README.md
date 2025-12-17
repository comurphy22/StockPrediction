# Stock Prediction with Politician Trading Signals

ML system predicting stock movements using congressional trading data, news sentiment, and technical indicators.

## Quick Start

```bash
# Setup
git clone https://github.com/comurphy22/StockPrediction.git
cd StockPrediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run predictions
python scripts/live_prediction_demo.py
```

## Web App

```bash
# Backend (Terminal 1)
cd backend && pip install -r requirements.txt
PYTHONPATH=../src uvicorn app.main:app --port 8000

# Frontend (Terminal 2)
cd frontend && npm install && npm run dev
```

Open **http://localhost:3000**

## API Keys

Create `.env`:
```
QUIVER_API_KEY=your_key  # Required - quiverquant.com
NEWS_API_KEY=your_key    # Optional - newsapi.org
```

## Results

| Sector | Stock | Accuracy |
|--------|-------|----------|
| Financials | WFC | **70%** |
| Healthcare | PFE | 60% |
| Tech | GOOGL | 50% |

Politician signals work best for **financial/healthcare sectors**.

## Authors

Conner Murphy & William Coleman
