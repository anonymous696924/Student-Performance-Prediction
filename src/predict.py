"""
Prediction module for student performance.
Loads trained model and generates predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from src.preprocess import StudentPerformancePreprocessor


class StudentPerformancePredictor:
    """
    Loads trained model and generates predictions with confidence analysis.
    """
    
    def __init__(self, model_path):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved .pkl model file
        """
        self.model_path = model_path
        self.model = None
        self.metadata = None
        self.preprocessor = StudentPerformancePreprocessor()
        self._load_model()
    
    def _load_model(self):
        """Load model and metadata."""
        self.model = joblib.load(self.model_path)
        
        metadata_path = str(self.model_path).replace('.pkl', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        print(f"✓ Loaded {self.metadata['model_type'].upper()} model")
        print(f"  Features: {len(self.metadata['feature_names'])}")
        print(f"  Test R²: {self.metadata['metrics'].get('test_r2', 'N/A')}")
    
    def predict(self, data_path, output_path=None, at_risk_threshold=70):
        """
        Generate predictions on new data.
        
        Args:
            data_path: Path to CSV with student data
            output_path: Optional path to save predictions
            at_risk_threshold: Grade threshold for flagging at-risk students
            
        Returns:
            DataFrame with predictions and analysis
        """
        # Load and preprocess data
        df_raw = pd.read_csv(data_path)
        print(f"\nLoading {len(df_raw)} student records...")
        
        # Keep original data for output
        df_original = df_raw.copy()
        
        # Preprocess
        self.preprocessor.fit(df_raw)
        X, _ = self.preprocessor.transform(df_raw)
        
        # Align features with trained model
        expected_features = self.metadata['feature_names']
        X = X[expected_features]
        
        # Generate predictions
        predictions = self.model.predict(X)
        
        # Build results DataFrame
        results = pd.DataFrame({
            'StudentID': df_original['StudentID'].values if 'StudentID' in df_original else range(len(predictions)),
            'Name': df_original['Name'].values if 'Name' in df_original else ['Unknown'] * len(predictions),
            'Predicted_Grade': predictions,
            'At_Risk': predictions < at_risk_threshold,
            'Risk_Level': pd.cut(predictions, 
                                 bins=[0, 50, 70, 85, 100],
                                 labels=['Critical', 'Intervention', 'Monitor', 'Good'])
        })
        
        # Add actual grades if available
        if 'FinalGrade' in df_original.columns:
            results['Actual_Grade'] = df_original['FinalGrade'].values
            results['Prediction_Error'] = results['Actual_Grade'] - results['Predicted_Grade']
        
        # Summary statistics
        print(f"\n{'='*60}")
        print(f"PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total students: {len(results)}")
        print(f"At-risk students (<{at_risk_threshold}): {results['At_Risk'].sum()} "
              f"({100*results['At_Risk'].sum()/len(results):.1f}%)")
        print(f"\nGrade Distribution:")
        print(results['Risk_Level'].value_counts().sort_index())
        
        print(f"\nPredicted Grade Stats:")
        print(f"  Mean: {results['Predicted_Grade'].mean():.2f}")
        print(f"  Std:  {results['Predicted_Grade'].std():.2f}")
        print(f"  Min:  {results['Predicted_Grade'].min():.2f}")
        print(f"  Max:  {results['Predicted_Grade'].max():.2f}")
        
        # Save results
        if output_path:
            results.to_csv(output_path, index=False)
            print(f"\n✓ Predictions saved to {output_path}")
        
        return results
    
    def predict_student(self, student_features):
        """
        Predict grade for a single student.
        
        Args:
            student_features: Dict with feature values
            
        Returns:
            Predicted grade and at-risk status
        """
        df = pd.DataFrame([student_features])
        self.preprocessor.fit(df)
        X, _ = self.preprocessor.transform(df)
        
        expected_features = self.metadata['feature_names']
        X = X[expected_features]
        
        prediction = self.model.predict(X)[0]
        
        return {
            'predicted_grade': float(prediction),
            'at_risk': prediction < 70,
            'risk_level': self._get_risk_level(prediction)
        }
    
    @staticmethod
    def _get_risk_level(grade):
        """Categorize risk level based on predicted grade."""
        if grade < 50:
            return 'Critical'
        elif grade < 70:
            return 'Intervention'
        elif grade < 85:
            return 'Monitor'
        else:
            return 'Good'


def generate_predictions(model_path, data_path, output_path=None):
    """
    Convenience function to generate predictions.
    
    Args:
        model_path: Path to trained model
        data_path: Path to student data CSV
        output_path: Optional output path for predictions
        
    Returns:
        Predictions DataFrame
    """
    predictor = StudentPerformancePredictor(model_path)
    results = predictor.predict(data_path, output_path)
    return results
