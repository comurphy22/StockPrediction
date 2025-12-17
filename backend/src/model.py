"""
Machine learning model training and evaluation functions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def train_model(X_train: pd.DataFrame, 
                y_train: pd.Series,
                model_type: str = 'random_forest',
                **kwargs) -> object:
    """
    Train a classification model for stock direction prediction.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target variable
    model_type : str
        Type of model to train:
        - 'random_forest': Random Forest Classifier
        - 'logistic': Logistic Regression
    **kwargs : dict
        Additional parameters to pass to the model
    
    Returns:
    --------
    object
        Trained model
    """
    print(f"\n=== Training {model_type.upper()} Model ===")
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Training target distribution:\n{y_train.value_counts()}")
    
    if model_type == 'random_forest':
        # Default parameters for Random Forest
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        # Update with user-provided parameters
        default_params.update(kwargs)
        
        model = RandomForestClassifier(**default_params)
        
    elif model_type == 'logistic':
        # Default parameters for Logistic Regression
        default_params = {
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs'
        }
        # Update with user-provided parameters
        default_params.update(kwargs)
        
        model = LogisticRegression(**default_params)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'random_forest' or 'logistic'")
    
    # Train the model
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    # Display feature importances for Random Forest
    if model_type == 'random_forest' and hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
    
    return model


def evaluate_model(model: object,
                  X_test: pd.DataFrame,
                  y_test: pd.Series,
                  verbose: bool = True) -> dict:
    """
    Evaluate a trained classification model.
    
    Parameters:
    -----------
    model : object
        Trained scikit-learn model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target variable
    verbose : bool
        Whether to print detailed evaluation metrics
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    if verbose:
        print("\n=== Model Evaluation ===")
        print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
        print(f"Test target distribution:\n{y_test.value_counts()}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    
    # Add ROC AUC if probabilities are available
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            metrics['roc_auc'] = roc_auc
        except Exception:
            metrics['roc_auc'] = None
    
    if verbose:
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)']))
        
        print("\n=== Confusion Matrix ===")
        cm = confusion_matrix(y_test, y_pred)
        print(f"                Predicted")
        print(f"                Down  Up")
        print(f"Actual  Down    {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"        Up      {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        print("\n=== Summary Metrics ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        if metrics.get('roc_auc'):
            print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        
        # Calculate baseline (majority class)
        baseline_accuracy = y_test.value_counts().max() / len(y_test)
        print(f"\nBaseline (majority class): {baseline_accuracy:.4f}")
        print(f"Improvement over baseline: {(accuracy - baseline_accuracy):.4f} ({(accuracy/baseline_accuracy - 1)*100:.2f}%)")
    
    return metrics


def backtest_strategy(model: object,
                     X_test: pd.DataFrame,
                     y_test: pd.Series,
                     prices: pd.Series = None,
                     initial_capital: float = 10000,
                     verbose: bool = True) -> dict:
    """
    Simple backtest of a trading strategy based on model predictions.
    
    Strategy:
    - Buy when model predicts price will go up (1)
    - Sell/Hold cash when model predicts price will go down (0)
    
    Parameters:
    -----------
    model : object
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Actual outcomes
    prices : pd.Series, optional
        Actual price changes (if None, uses binary outcomes)
    initial_capital : float
        Starting capital for backtest
    verbose : bool
        Whether to print results
    
    Returns:
    --------
    dict
        Backtest results including returns, win rate, etc.
    """
    if verbose:
        print("\n=== Simple Backtest ===")
    
    # Get predictions
    predictions = model.predict(X_test)
    
    # If prices not provided, use binary returns (1% gain if correct, -1% if wrong)
    if prices is None:
        returns = np.where(predictions == y_test, 0.01, -0.01)
    else:
        # Use actual price changes, but only when we predict up
        # Otherwise stay in cash (0% return)
        returns = np.where(predictions == 1, prices, 0)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    final_value = initial_capital * cumulative_returns[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    # Calculate win rate
    wins = sum((predictions == 1) & (y_test == 1))
    total_trades = sum(predictions == 1)
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    # Calculate buy-and-hold benchmark
    if prices is not None:
        buy_hold_return = (1 + prices).prod() - 1
    else:
        buy_hold_return = None
    
    results = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'num_trades': total_trades,
        'win_rate': win_rate,
        'buy_hold_return': buy_hold_return
    }
    
    if verbose:
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Value:     ${final_value:,.2f}")
        print(f"Total Return:    {total_return:.2%}")
        print(f"Number of Trades: {total_trades}")
        print(f"Win Rate:        {win_rate:.2%}")
        if buy_hold_return is not None:
            print(f"Buy & Hold Return: {buy_hold_return:.2%}")
            print(f"Strategy vs B&H: {(total_return - buy_hold_return):.2%}")
    
    return results


if __name__ == "__main__":
    # Example usage
    from data_loader import fetch_stock_data
    from feature_engineering import create_features, handle_missing_values
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    
    print("=== Model Training Example ===\n")
    
    # Fetch data
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Create features
    X, y, dates = create_features(stock_data)
    
    # Handle missing values
    X_clean = handle_missing_values(X, strategy='drop')
    y_clean = y.loc[X_clean.index]
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=3)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_clean), 1):
        print(f"\n{'='*50}")
        print(f"Fold {fold}")
        print(f"{'='*50}")
        
        X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Train Random Forest
        rf_model = train_model(X_train_scaled, y_train, model_type='random_forest')
        rf_metrics = evaluate_model(rf_model, X_test_scaled, y_test)
        
        # Train Logistic Regression
        lr_model = train_model(X_train_scaled, y_train, model_type='logistic')
        lr_metrics = evaluate_model(lr_model, X_test_scaled, y_test)
        
        print(f"\nRandom Forest F1: {rf_metrics['f1_score']:.4f}")
        print(f"Logistic Regression F1: {lr_metrics['f1_score']:.4f}")
        
        # Only show backtest for last fold
        if fold == 3:
            backtest_strategy(rf_model, X_test_scaled, y_test)
