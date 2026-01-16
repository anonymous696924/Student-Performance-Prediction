"""
Model training and evaluation module for student performance prediction.
Trains multiple algorithms and identifies at-risk students.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from pathlib import Path


class StudentPerformanceModel:
    """
    Trains and evaluates models for student performance prediction.
    Supports multiple algorithms with feature importance tracking.
    """
    
    def __init__(self, model_type='linear', random_state=42):
        """
        Initialize model.
        
        Args:
            model_type: 'linear', 'tree', 'random_forest', or 'xgboost'
            random_state: For reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._create_model()
        self.metrics = {}
        self.feature_importance = None
        self.feature_names = None
        
    def _create_model(self):
        """Create model based on type."""
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'tree':
            return DecisionTreeRegressor(random_state=self.random_state, max_depth=10)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=self.random_state, 
                                        n_jobs=-1, max_depth=15)
        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                return xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1)
            except ImportError:
                print("XGBoost not installed, falling back to RandomForest")
                return RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, cv_folds=5):
        """
        Train model with cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training target
            cv_folds: Number of folds for cross-validation
        """
        self.feature_names = X_train.columns.tolist()
        
        # Train model
        self.model.fit(X_train, y_train)
        print(f"✓ {self.model_type.upper()} model trained on {len(X_train)} samples")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, 
                                    scoring='r2')
        self.metrics['cv_r2_mean'] = cv_scores.mean()
        self.metrics['cv_r2_std'] = cv_scores.std()
        print(f"  Cross-validation R² = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Extract feature importance (tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'coef_'):  # Linear models
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(self.model.coef_)
            }).sort_values('importance', ascending=False)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
        """
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        self.metrics['test_mae'] = mae
        self.metrics['test_rmse'] = rmse
        self.metrics['test_r2'] = r2
        self.metrics['test_mse'] = mse
        
        print(f"\n  Test Set Performance:")
        print(f"  MAE  = {mae:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  R²   = {r2:.4f}")
        
        if r2 > 0.75:
            print(f"  ✓ Performance threshold (R² > 0.75) PASSED")
        else:
            print(f"  ⚠ Performance below target (R² > 0.75)")
        
        return y_pred
    
    def identify_at_risk_students(self, y_test, y_pred, threshold=70):
        """
        Identify students predicted to score below intervention threshold.
        
        Args:
            y_test: Actual test grades
            y_pred: Predicted grades
            threshold: Grade threshold for intervention (default: 70)
            
        Returns:
            DataFrame with at-risk student analysis
        """
        at_risk = pd.DataFrame({
            'actual_grade': y_test.values,
            'predicted_grade': y_pred,
            'difference': y_test.values - y_pred,
            'at_risk': y_pred < threshold
        })
        
        at_risk_count = at_risk['at_risk'].sum()
        total_count = len(at_risk)
        
        print(f"\n  At-Risk Analysis (threshold < {threshold}):")
        print(f"  Students flagged for intervention: {at_risk_count}/{total_count} "
              f"({100*at_risk_count/total_count:.1f}%)")
        
        return at_risk
    
    def save_model(self, filepath):
        """Save trained model and metadata."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in self.metrics.items()}
        }
        metadata_path = str(filepath).replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath):
        """Load trained model."""
        self.model = joblib.load(filepath)
        metadata_path = str(filepath).replace('.pkl', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.feature_names = metadata['feature_names']
        self.metrics = metadata['metrics']
        print(f"✓ Model loaded from {filepath}")


def train_and_evaluate(X, y, model_types=['linear', 'tree', 'random_forest']):
    """
    Train and evaluate multiple models.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_types: List of model types to train
        
    Returns:
        Dictionary of trained models and results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set: {len(X_train)} | Test set: {len(X_test)}\n")
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'='*50}")
        
        model = StudentPerformanceModel(model_type=model_type)
        model.train(X_train, y_train)
        y_pred = model.evaluate(X_test, y_test)
        at_risk = model.identify_at_risk_students(y_test, y_pred)
        
        results[model_type] = {
            'model': model,
            'y_pred': y_pred,
            'at_risk': at_risk
        }
        
        if model.feature_importance is not None:
            print(f"\n  Top 5 Important Features:")
            for idx, row in model.feature_importance.head(5).iterrows():
                print(f"    {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    return results, X_train, X_test, y_train, y_test
