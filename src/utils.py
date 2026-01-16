"""
Utility functions for student performance prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_sample_prediction_input():
    """
    Create sample input for testing predictions.
    
    Returns:
        dict: Sample student features
    """
    sample = {
        'StudentID': 999,
        'Name': 'Test Student',
        'Gender': 'Male',
        'AttendanceRate': 85.0,
        'StudyHoursPerWeek': 15.0,
        'PreviousGrade': 78.0,
        'ExtracurricularActivities': 2,
        'ParentalSupport': 'Medium',
        'Online Classes Taken': False
    }
    return sample


def validate_student_data(df):
    """
    Validate student data for modeling.
    
    Args:
        df: DataFrame with student data
        
    Returns:
        tuple: (is_valid, errors_list)
    """
    errors = []
    
    # Check required columns
    required_cols = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 
                    'ExtracurricularActivities', 'ParentalSupport']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
    
    # Check data types and ranges
    if 'AttendanceRate' in df.columns:
        if not (df['AttendanceRate'].between(0, 100).all()):
            errors.append("AttendanceRate values outside [0, 100] range")
    
    if 'StudyHoursPerWeek' in df.columns:
        if (df['StudyHoursPerWeek'] < 0).any():
            errors.append("StudyHoursPerWeek contains negative values")
    
    if 'PreviousGrade' in df.columns:
        if not (df['PreviousGrade'].between(0, 100).all()).any():
            errors.append("PreviousGrade values outside [0, 100] range")
    
    return len(errors) == 0, errors


def get_intervention_recommendations(at_risk_df, threshold=70):
    """
    Generate intervention recommendations for at-risk students.
    
    Args:
        at_risk_df: DataFrame with predictions and actual grades
        threshold: Grade threshold (default: 70)
        
    Returns:
        DataFrame: Recommendations for each student
    """
    recommendations = []
    
    for idx, row in at_risk_df.iterrows():
        student_id = row.get('StudentID', idx)
        predicted = row['predicted_grade']
        
        if predicted < 50:
            rec = {
                'StudentID': student_id,
                'Risk_Level': 'Critical',
                'Priority': 'Urgent',
                'Recommendation': 'Immediate intervention - dedicated tutoring and study support',
                'Predicted_Grade': predicted
            }
        elif predicted < 70:
            rec = {
                'StudentID': student_id,
                'Risk_Level': 'Intervention Needed',
                'Priority': 'High',
                'Recommendation': 'Additional tutoring, study group participation, office hours',
                'Predicted_Grade': predicted
            }
        elif predicted < 85:
            rec = {
                'StudentID': student_id,
                'Risk_Level': 'Monitor',
                'Priority': 'Medium',
                'Recommendation': 'Regular check-ins, reinforce study habits',
                'Predicted_Grade': predicted
            }
        else:
            rec = {
                'StudentID': student_id,
                'Risk_Level': 'On Track',
                'Priority': 'Low',
                'Recommendation': 'Maintain current performance, encourage leadership roles',
                'Predicted_Grade': predicted
            }
        
        recommendations.append(rec)
    
    return pd.DataFrame(recommendations)


def export_model_report(model, X_test, y_test, y_pred, output_dir='./'):
    """
    Generate comprehensive model report.
    
    Args:
        model: Trained StudentPerformanceModel
        X_test: Test features
        y_test: Test target
        y_pred: Predictions
        output_dir: Directory to save report
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = f"""
# Student Performance Prediction Model Report

## Model Information
- Type: {model.model_type.upper()}
- Training Samples: {len(X_test)} (test set)
- Features: {len(X_test.columns)}

## Performance Metrics
- MAE:  {mean_absolute_error(y_test, y_pred):.4f}
- RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}
- R²:   {r2_score(y_test, y_pred):.4f}

## Cross-Validation Results
- CV R² Mean: {model.metrics.get('cv_r2_mean', 'N/A')}
- CV R² Std:  {model.metrics.get('cv_r2_std', 'N/A')}

## Feature Importance
"""
    
    if model.feature_importance is not None:
        report += "\n"
        for idx, row in model.feature_importance.head(10).iterrows():
            report += f"- {row['feature']}: {row['importance']:.4f}\n"
    
    report += "\n## Grade Distribution Analysis\n"
    report += f"- Students below 50: {(y_pred < 50).sum()}\n"
    report += f"- Students 50-70: {((y_pred >= 50) & (y_pred < 70)).sum()}\n"
    report += f"- Students 70-85: {((y_pred >= 70) & (y_pred < 85)).sum()}\n"
    report += f"- Students 85+: {(y_pred >= 85).sum()}\n"
    
    report_path = output_dir / f"model_report_{model.model_type}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    return str(report_path)
