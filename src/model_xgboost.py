"""
XGBoost Model Implementation

Implements XGBoost classifier for stock price movement prediction.
Provides same interface as existing Random Forest model for easy comparison.

Features:
- XGBoost Classifier with optimized hyperparameters
- Hyperparameter tuning via Grid Search
- Feature importance extraction
- Evaluation metrics (accuracy, precision, recall, F1)
"""

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import pandas as pd


def train_xgboost_model(X_train, y_train, 
                       n_estimators=100,
                       max_depth=6,
                       learning_rate=0.1,
                       subsample=0.8,
                       colsample_bytree=0.8,
                       gamma=0,
                       min_child_weight=1,
                       alpha=1.0,  # L1 regularization (optimal: 1.0, reduces overfitting)
                       reg_lambda=1,  # L2 regularization
                       random_state=42,
                       verbose=True,
                       use_gpu=False):
    """
    Train XGBoost classifier for binary classification.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training labels (0 or 1)
    n_estimators : int
        Number of boosting rounds (trees)
    max_depth : int
        Maximum tree depth
    learning_rate : float
        Step size shrinkage (eta)
    subsample : float
        Subsample ratio of training instances
    colsample_bytree : float
        Subsample ratio of features
    gamma : float
        Minimum loss reduction for split
    min_child_weight : int
        Minimum sum of instance weight in child
    random_state : int
        Random seed
    verbose : bool
        Print training progress
    use_gpu : bool
        Use GPU acceleration (tree_method='gpu_hist')
    
    Returns:
    --------
    model : xgb.XGBClassifier
        Trained XGBoost model
    """
    
    if verbose:
        print(f"\n=== Training XGBoost Model ===")
        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Training target distribution:")
        if hasattr(y_train, 'value_counts'):
            print(y_train.value_counts())
        else:
            unique, counts = np.unique(y_train, return_counts=True)
            for val, count in zip(unique, counts):
                print(f"  {val}: {count}")
    
    # Set tree method based on GPU availability
    tree_method = 'gpu_hist' if use_gpu else 'hist'
    
    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        min_child_weight=min_child_weight,
        reg_alpha=alpha,  # L1 regularization (Lasso) - penalizes absolute values
        reg_lambda=reg_lambda,  # L2 regularization (Ridge) - penalizes squared values
        tree_method=tree_method,
        random_state=random_state,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    if verbose:
        print(f"\nHyperparameters:")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  subsample: {subsample}")
        print(f"  colsample_bytree: {colsample_bytree}")
        print(f"  gamma: {gamma}")
        print(f"  min_child_weight: {min_child_weight}")
        print(f"  reg_alpha (L1): {alpha}")
        print(f"  reg_lambda (L2): {reg_lambda}")
        print(f"  tree_method: {tree_method}")
        print(f"\nTraining model...")
    
    # Train model
    model.fit(X_train, y_train, verbose=False)
    
    if verbose:
        print("Model training complete!")
        
        # Show feature importance (top 10)
        if hasattr(X_train, 'columns'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))
    
    return model


def tune_xgboost_hyperparameters(X_train, y_train, 
                                 param_grid=None,
                                 cv=3,
                                 scoring='accuracy',
                                 verbose=True,
                                 n_jobs=-1):
    """
    Perform grid search to find optimal XGBoost hyperparameters.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training labels
    param_grid : dict
        Parameter grid for search. If None, uses default grid.
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric ('accuracy', 'f1', 'precision', 'recall')
    verbose : bool
        Print search progress
    n_jobs : int
        Number of parallel jobs (-1 = all cores)
    
    Returns:
    --------
    best_model : xgb.XGBClassifier
        Model with best parameters
    best_params : dict
        Best hyperparameters found
    cv_results : dict
        Full cross-validation results
    """
    
    if verbose:
        print(f"\n=== XGBoost Hyperparameter Tuning ===")
        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Cross-validation folds: {cv}")
        print(f"Scoring metric: {scoring}")
    
    # Default parameter grid if not provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    
    if verbose:
        print(f"\nParameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"\nTotal combinations: {total_combinations}")
        print(f"Total fits: {total_combinations * cv}")
        print(f"\nStarting grid search...")
    
    # Base model
    base_model = xgb.XGBClassifier(
        tree_method='hist',
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1 if verbose else 0
    )
    
    grid_search.fit(X_train, y_train)
    
    if verbose:
        print(f"\n[OK] Grid search complete!")
        print(f"\nBest parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nBest {scoring} score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


def evaluate_xgboost_model(model, X_test, y_test, verbose=True):
    """
    Evaluate XGBoost model on test data.
    
    Parameters:
    -----------
    model : xgb.XGBClassifier
        Trained XGBoost model
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test labels
    verbose : bool
        Print evaluation metrics
    
    Returns:
    --------
    metrics : dict
        Dictionary containing accuracy, precision, recall, F1 score
    """
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    if verbose:
        print(f"\n=== XGBoost Model Evaluation ===")
        print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
    
    return metrics


def get_xgboost_feature_importance(model, feature_names=None, top_n=None):
    """
    Extract and rank feature importance from trained XGBoost model.
    
    Parameters:
    -----------
    model : xgb.XGBClassifier
        Trained model
    feature_names : list
        List of feature names
    top_n : int
        Return only top N features (None = all)
    
    Returns:
    --------
    importance_df : pd.DataFrame
        DataFrame with features and importance scores
    """
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    if top_n is not None:
        importance_df = importance_df.head(top_n)
    
    return importance_df


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("XGBoost Model Module - Test")
    print("="*70)
    
    # Generate synthetic data for testing
    print("\n[1/5] Generating synthetic test data...")
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=25,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(25)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    
    # Split data
    split_idx = int(len(X_df) * 0.8)
    X_train = X_df[:split_idx]
    X_test = X_df[split_idx:]
    y_train = y_series[:split_idx]
    y_test = y_series[split_idx:]
    
    print(f"      [OK] Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Test 1: Basic training
    print("\n[2/5] Testing basic XGBoost training...")
    model = train_xgboost_model(
        X_train, y_train,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        verbose=True
    )
    print("      [OK] Training successful")
    
    # Test 2: Evaluation
    print("\n[3/5] Testing model evaluation...")
    metrics = evaluate_xgboost_model(model, X_test, y_test, verbose=True)
    print("      [OK] Evaluation successful")
    
    # Test 3: Feature importance
    print("\n[4/5] Testing feature importance extraction...")
    importance = get_xgboost_feature_importance(model, feature_names, top_n=10)
    print("\nTop 10 Features:")
    print(importance.to_string(index=False))
    print("      [OK] Feature importance extraction successful")
    
    # Test 4: Hyperparameter tuning (quick test)
    print("\n[5/5] Testing hyperparameter tuning (small grid)...")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.3]
    }
    
    best_model, best_params, cv_results = tune_xgboost_hyperparameters(
        X_train, y_train,
        param_grid=param_grid,
        cv=3,
        verbose=True
    )
    print("      [OK] Hyperparameter tuning successful")
    
    print("\n" + "="*70)
    print("[OK] ALL TESTS PASSED!")
    print("="*70)
    print("\nXGBoost module is ready for use.")
    print("Import with: from model_xgboost import train_xgboost_model")
