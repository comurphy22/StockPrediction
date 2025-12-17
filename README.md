# Stock Prediction with Politician Trading Signals

ML system predicting stock movements using congressional trading data, news sentiment, and technical indicators.

## Run the Web App

### Step 1: Clone & Setup

```bash
git clone https://github.com/comurphy22/StockPrediction.git
cd StockPrediction
```

### Step 2: Create API Key File

Create a `.env` file in the project root:
```
QUIVER_API_KEY=your_key_here
```
Get your free key at [quiverquant.com](https://www.quiverquant.com/)

### Step 3: Start Backend

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"

cd backend
PYTHONPATH=../src uvicorn app.main:app --port 8000
```

### Step 4: Start Frontend (new terminal)

```bash
cd frontend
npm install
npm run dev
```

### Step 5: Open App

Go to **http://localhost:3000**

---

## Results

| Sector | Stock | Accuracy |
|--------|-------|----------|
| Financials | WFC | **70%** |
| Healthcare | PFE | 60% |
| Tech | GOOGL | 50% |

Politician signals work best for **financial/healthcare sectors**.

## Authors

Conner Murphy & William Coleman
