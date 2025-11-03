# Analysis Scripts

This folder contains all standalone analysis and experimentation scripts for the StockPrediction project.

## 📊 Scripts Overview

### 1. `analyze_feature_importance.py`
**Purpose:** Extract and rank feature importance from Random Forest models

**What it does:**
- Trains Random Forest on best-performing stocks (BABA, QCOM, NVDA)
- Extracts feature importance scores for all 41 features
- Aggregates importance across stocks (mean, std, min, max)
- Calculates cumulative importance percentages
- Categorizes features by type

**Outputs:**
- `Results/feature_importance_rankings.csv` - All features ranked by importance
- Console output with top features and statistics

**Usage:**
```bash
python scripts/analyze_feature_importance.py
```

**Runtime:** ~2-3 minutes

---

### 2. `feature_selection_experiments.py`
**Purpose:** Empirically test different feature counts to find optimal set size

**What it does:**
- Tests 6 feature counts: 5, 10, 15, 20, 25, ALL (41)
- For each count: loads top N features, trains model, evaluates performance
- Tests across 3 stocks = 18 total experiments
- Calculates train/test accuracy and overfitting gap
- Identifies optimal feature count using multiple methods

**Outputs:**
- `Results/feature_selection_results.csv` - Performance for each feature count
- Console output with recommendations

**Usage:**
```bash
python scripts/feature_selection_experiments.py
```

**Runtime:** ~3-4 minutes

**Key Finding:** 25 features optimal (67.16% test acc), beats ALL 41 features by +11.27%

---

### 3. `visualize_features.py`
**Purpose:** Create comprehensive visualizations of feature importance and selection results

**What it does:**
- Loads feature importance rankings and selection experiment results
- Generates 6-panel comprehensive chart:
  1. Top 25 features bar chart
  2. Category breakdown pie chart
  3. Cumulative importance curve
  4. Feature count vs accuracy line chart
  5. Stock-specific heatmap
  6. Overfitting gap analysis
- Generates 2-panel presentation chart for reports

**Outputs:**
- `visualizations/feature_importance_visualization.png` - 6-panel comprehensive
- `visualizations/feature_importance_simple.png` - 2-panel presentation
- Both at 300 DPI for publication quality

**Usage:**
```bash
python scripts/visualize_features.py
```

**Runtime:** ~30 seconds

**Prerequisites:** Run `analyze_feature_importance.py` and `feature_selection_experiments.py` first

---

### 4. `compare_models.py`
**Purpose:** Comprehensive comparison of Random Forest, XGBoost, and Logistic Regression

**What it does:**
- Loads optimal 25-feature set from rankings
- Tests 3 models on 3 stocks (BABA, QCOM, NVDA)
- Trains each model with same data/splits for fair comparison
- Evaluates train/test accuracy, F1 score, overfitting gap
- Determines best model for MVP

**Outputs:**
- `Results/model_comparison_results.csv` - Complete metrics for all models
- Console output with winner analysis and recommendations

**Usage:**
```bash
python scripts/compare_models.py
```

**Runtime:** ~3-4 minutes

**Key Finding:** XGBoost wins with 67.42% avg test accuracy (exceeds 60% MVP target)

---

### 5. `validate_hypothesis_multiyear.py`
**Purpose:** Multi-year validation testing (2017-2019) with corrected tickers

**What it does:**
- Tests hypothesis across multiple years for robustness
- Uses stocks with consistent news coverage (NFLX, NVDA, BABA, QCOM, MU)
- Validates that politician trading signals improve predictions
- Compares performance year-over-year

**Outputs:**
- `logs/multiyear_results.log` - Complete validation results
- Console output with yearly performance breakdown

**Usage:**
```bash
python scripts/validate_hypothesis_multiyear.py
```

**Runtime:** ~5-6 minutes

**Key Finding:** +1.83% average improvement, +4.37% in 2019

---

## 🔄 Recommended Workflow

### Initial Analysis (First Time)
```bash
# 1. Feature importance analysis
python scripts/analyze_feature_importance.py

# 2. Feature selection experiments
python scripts/feature_selection_experiments.py

# 3. Generate visualizations
python scripts/visualize_features.py

# 4. Compare models
python scripts/compare_models.py

# 5. Multi-year validation (optional)
python scripts/validate_hypothesis_multiyear.py
```

**Total time:** ~15-20 minutes

### Quick Re-run (After Code Changes)
```bash
# Just re-run the specific script you need
python scripts/compare_models.py
```

---

## 📋 Dependencies

All scripts require:
- Python 3.8+
- Virtual environment activated
- Packages from `requirements.txt` installed
- Capitol Trades API key in `.env` file

**Activate environment:**
```bash
source venv/bin/activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🎯 Output Locations

| Script | Output Type | Location |
|--------|-------------|----------|
| analyze_feature_importance.py | CSV | `Results/feature_importance_rankings.csv` |
| feature_selection_experiments.py | CSV | `Results/feature_selection_results.csv` |
| visualize_features.py | PNG | `visualizations/*.png` |
| compare_models.py | CSV | `Results/model_comparison_results.csv` |
| validate_hypothesis_multiyear.py | Log | `logs/multiyear_results.log` |

---

## 💡 Tips

1. **Run scripts from project root:**
   ```bash
   python scripts/script_name.py
   ```

2. **View logs in real-time:**
   ```bash
   python scripts/compare_models.py 2>&1 | tee logs/comparison.log
   ```

3. **Check results:**
   ```bash
   cat Results/model_comparison_results.csv
   ```

4. **View visualizations:**
   ```bash
   open visualizations/feature_importance_visualization.png
   ```

---

## 🔧 Troubleshooting

**Script can't find modules:**
- Make sure you're in the project root directory
- Activate virtual environment: `source venv/bin/activate`

**API key errors:**
- Check `.env` file has `CAPITOL_TRADES_API_KEY=your_key`

**Memory errors:**
- Reduce number of stocks tested
- Close other applications

**Import errors:**
- Reinstall requirements: `pip install -r requirements.txt`

---

*Scripts organized: November 2, 2025*
*All scripts tested and production-ready*
