# Stock Prediction Web Application

A modern web interface for the Stock Prediction ML system, built with **FastAPI** (backend) and **Next.js** (frontend).

## 🚀 Quick Start

### Option 1: Using the Startup Script (Recommended)

```bash
# Make the script executable
chmod +x start_web_app.sh

# Run the script
./start_web_app.sh
```

This will:
1. Set up a Python virtual environment
2. Install backend dependencies
3. Install frontend dependencies
4. Start both servers

### Option 2: Manual Setup

#### Backend (FastAPI)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"

# Start the server
cd backend
PYTHONPATH=../src uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend (Next.js)

```bash
# In a new terminal
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Option 3: Using Docker

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## 🌐 Access the Application

| Service | URL | Description |
|---------|-----|-------------|
| **Dashboard** | http://localhost:3000 | Main web interface |
| **API** | http://localhost:8000 | FastAPI backend |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger docs |
| **Health Check** | http://localhost:8000/health | Server status |

## 📁 Project Structure

```
StockPrediction/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py            # FastAPI app entry point
│   │   ├── config.py          # Configuration
│   │   ├── routes/            # API endpoints
│   │   │   ├── predictions.py # /api/predictions/*
│   │   │   ├── backtest.py    # /api/backtest/*
│   │   │   └── data.py        # /api/data/*
│   │   ├── services/          # Business logic
│   │   │   ├── model_service.py
│   │   │   ├── prediction_service.py
│   │   │   ├── data_service.py
│   │   │   └── backtest_service.py
│   │   └── models/
│   │       └── schemas.py     # Pydantic models
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                   # Next.js frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx     # Root layout
│   │   │   ├── page.tsx       # Dashboard page
│   │   │   └── globals.css    # Global styles
│   │   ├── components/        # React components
│   │   │   ├── StockCard.tsx
│   │   │   ├── SummaryCard.tsx
│   │   │   ├── Header.tsx
│   │   │   └── LoadingSpinner.tsx
│   │   └── lib/
│   │       ├── api.ts         # API client
│   │       └── utils.ts       # Utility functions
│   ├── package.json
│   ├── tailwind.config.ts
│   └── Dockerfile
│
├── src/                        # Original ML code (unchanged)
├── docker-compose.yml          # Docker orchestration
├── start_web_app.sh           # Quick start script
└── WEB_APP_README.md          # This file
```

## 🔌 API Endpoints

### Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/predictions/{ticker}` | Get prediction for a single stock |
| POST | `/api/predictions/batch` | Get predictions for multiple stocks |
| GET | `/api/predictions/portfolio/recommendations` | Get ranked portfolio recommendations |
| GET | `/api/predictions/tickers/available` | List available tickers by tier |

### Backtest

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/backtest/run` | Run a new backtest |
| GET | `/api/backtest/summary` | Get backtest results summary |

### Data

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/data/stock/{ticker}` | Get stock price data |
| GET | `/api/data/politician-trades/{ticker}` | Get politician trading activity |
| GET | `/api/data/validation-results` | Get historical validation results |
| GET | `/api/data/market-status` | Check if market is open |

## 🎨 Features

### Dashboard
- **Real-time predictions** for tracked stocks
- **Signal strength indicators** (Strong Buy → Strong Sell)
- **Confidence meters** showing model certainty
- **Data source indicators** (stock days, politician trades, news)
- **Tier badges** indicating historical accuracy

### Design
- **Dark trading terminal aesthetic** with custom color palette
- **Responsive layout** for desktop and mobile
- **Smooth animations** and transitions
- **Glass morphism effects** for depth

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Keys
QUIVER_API_KEY=your_quiver_api_key
NEWS_API_KEY=your_newsapi_key

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Stock Tiers

Stocks are categorized by historical accuracy:

| Tier | Stocks | Accuracy | Recommended |
|------|--------|----------|-------------|
| 1 | WFC, PFE, BABA | 60-70% | ✅ Yes |
| 2 | NFLX, GOOGL, FDX | 50-58% | ✅ Yes |
| 3 | NVDA, TSLA | 38-43% | ⚠️ Caution |

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Check Python path
echo $PYTHONPATH
```

### Frontend can't connect to backend
1. Ensure backend is running on port 8000
2. Check CORS settings in `backend/app/main.py`
3. Verify `NEXT_PUBLIC_API_URL` is set correctly

### Model training is slow
The first request for each ticker will train a new model (~30-60 seconds).
Subsequent requests use cached models for instant predictions.

## 📊 Technology Stack

### Backend
- **FastAPI** - High-performance async web framework
- **Pydantic** - Data validation with type hints
- **XGBoost** - Gradient boosting ML model
- **yfinance** - Yahoo Finance data fetching

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS
- **Lucide React** - Beautiful icons

### Deployment
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. 
It should not be used for actual trading without professional financial advice. 
Past performance does not guarantee future results.

---

Built with ❤️ using FastAPI, Next.js, and XGBoost

