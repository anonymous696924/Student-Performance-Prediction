"""
Example usage patterns for Student Performance Prediction system.
These examples show the most common workflows.
"""

# ============================================================================
# EXAMPLE 1: Full Training Pipeline
# ============================================================================

def example_train_and_evaluate():
    """Train all models and evaluate them."""
    from src.preprocess import preprocess_student_data
    from src.train import train_and_evaluate
    from src.utils import get_intervention_recommendations
    
    # Load and preprocess
    X, y, preprocessor = preprocess_student_data('data/student_performance_updated_1000.csv')
    
    # Train multiple models
    results, X_train, X_test, y_train, y_test = train_and_evaluate(
        X, y,
        model_types=['linear', 'tree', 'random_forest']
    )
    
    # Get intervention recommendations
    at_risk = results['random_forest']['at_risk']
    recommendations = get_intervention_recommendations(at_risk)
    
    # Save best model
    best_model = results['random_forest']['model']
    model_path = best_model.save_model('models/student_performance_model.pkl')
    
    return model_path, recommendations


# ============================================================================
# EXAMPLE 2: Batch Prediction on New Data
# ============================================================================

def example_batch_prediction():
    """Generate predictions for multiple students."""
    from src.predict import generate_predictions
    
    # Generate predictions
    predictions = generate_predictions(
        model_path='models/student_performance_model.pkl',
        data_path='data/new_students.csv',
        output_path='predictions_output.csv'
    )
    
    # Analyze results
    print(f"Total students: {len(predictions)}")
    print(f"At-risk students: {predictions['At_Risk'].sum()}")
    print(f"\nRisk distribution:\n{predictions['Risk_Level'].value_counts()}")
    
    return predictions


# ============================================================================
# EXAMPLE 3: Single Student Prediction
# ============================================================================

def example_single_prediction():
    """Predict grade for one student."""
    from src.predict import StudentPerformancePredictor
    
    # Initialize predictor with trained model
    predictor = StudentPerformancePredictor('models/student_performance_model.pkl')
    
    # Define student features
    student = {
        'Gender': 'Female',
        'AttendanceRate': 92.0,
        'StudyHoursPerWeek': 20.0,
        'PreviousGrade': 85.0,
        'ExtracurricularActivities': 2,
        'ParentalSupport': 'High',
        'Online Classes Taken': False
    }
    
    # Get prediction
    result = predictor.predict_student(student)
    
    print(f"Predicted Grade: {result['predicted_grade']:.1f}")
    print(f"At Risk: {result['at_risk']}")
    print(f"Risk Level: {result['risk_level']}")
    
    return result


# ============================================================================
# EXAMPLE 4: Feature Importance Analysis
# ============================================================================

def example_feature_importance():
    """Analyze which features matter most."""
    from src.preprocess import preprocess_student_data
    from src.train import StudentPerformanceModel
    from sklearn.model_selection import train_test_split
    
    # Prepare data
    X, y, _ = preprocess_student_data('data/student_performance_updated_1000.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    model = StudentPerformanceModel(model_type='random_forest')
    model.train(X_train, y_train)
    
    # Display feature importance
    print("Feature Importance (Random Forest):")
    print(model.feature_importance.head(10).to_string(index=False))
    
    return model.feature_importance


# ============================================================================
# EXAMPLE 5: Data Exploration & Visualization
# ============================================================================

def example_exploration():
    """Load and explore the data."""
    import pandas as pd
    
    # Load raw data
    df = pd.read_csv('data/student_performance_updated_1000.csv')
    
    print(f"Dataset Shape: {df.shape}")
    print(f"\nFirst 5 records:")
    print(df.head())
    
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    
    print(f"\nFinal Grade Distribution:")
    print(f"  Mean: {df['FinalGrade'].mean():.2f}")
    print(f"  Std:  {df['FinalGrade'].std():.2f}")
    print(f"  Min:  {df['FinalGrade'].min():.2f}")
    print(f"  Max:  {df['FinalGrade'].max():.2f}")
    
    # At-risk students in dataset
    at_risk_count = (df['FinalGrade'] < 70).sum()
    print(f"\nAt-risk students (<70): {at_risk_count}/{len(df)} ({100*at_risk_count/len(df):.1f}%)")
    
    return df


# ============================================================================
# EXAMPLE 6: Custom Preprocessing
# ============================================================================

def example_custom_preprocessing():
    """Apply custom preprocessing logic."""
    from src.preprocess import StudentPerformancePreprocessor
    import pandas as pd
    
    # Load data
    df = pd.read_csv('data/student_performance_updated_1000.csv')
    
    # Create preprocessor
    preprocessor = StudentPerformancePreprocessor()
    
    # Fit on full dataset
    X_full, y_full = preprocessor.fit_transform(df)
    
    # Get feature names after preprocessing
    print(f"Features after preprocessing: {X_full.columns.tolist()}")
    print(f"Shape: {X_full.shape}")
    
    # Save preprocessor for later use
    import pickle
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    return preprocessor


# ============================================================================
# EXAMPLE 7: Model Comparison
# ============================================================================

def example_model_comparison():
    """Compare performance of different models."""
    from src.preprocess import preprocess_student_data
    from src.train import StudentPerformanceModel
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    # Prepare data
    X, y, _ = preprocess_student_data('data/student_performance_updated_1000.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate each model
    results = {}
    for model_type in ['linear', 'tree', 'random_forest']:
        model = StudentPerformanceModel(model_type=model_type)
        model.train(X_train, y_train)
        y_pred = model.evaluate(X_test, y_test)
        
        results[model_type] = {
            'r2': model.metrics['test_r2'],
            'mae': model.metrics['test_mae'],
            'rmse': model.metrics['test_rmse']
        }
    
    # Compare results
    comparison_df = pd.DataFrame(results).T
    print("\nModel Comparison:")
    print(comparison_df.to_string())
    
    return comparison_df


# ============================================================================
# EXAMPLE 8: Identify Critical Cases
# ============================================================================

def example_critical_cases():
    """Find students needing urgent intervention."""
    from src.predict import generate_predictions
    
    # Generate predictions
    predictions = generate_predictions(
        model_path='models/student_performance_model.pkl',
        data_path='data/student_performance_updated_1000.csv'
    )
    
    # Filter critical cases
    critical = predictions[predictions['Predicted_Grade'] < 50]
    intervention = predictions[
        (predictions['Predicted_Grade'] >= 50) & 
        (predictions['Predicted_Grade'] < 70)
    ]
    
    print(f"Critical Cases (<50): {len(critical)} students")
    print(critical[['Name', 'Predicted_Grade', 'At_Risk']].head(10))
    
    print(f"\nIntervention Needed (50-70): {len(intervention)} students")
    print(intervention[['Name', 'Predicted_Grade', 'At_Risk']].head(10))
    
    return critical, intervention


# ============================================================================
# EXAMPLE 9: Cross-Validation Analysis
# ============================================================================

def example_cross_validation():
    """Analyze cross-validation performance."""
    from src.preprocess import preprocess_student_data
    from src.train import StudentPerformanceModel
    
    # Prepare data
    X, y, _ = preprocess_student_data('data/student_performance_updated_1000.csv')
    
    # Train with cross-validation
    model = StudentPerformanceModel(model_type='random_forest')
    model.train(X, y, cv_folds=5)
    
    print(f"Cross-Validation Results:")
    print(f"  Mean R²: {model.metrics['cv_r2_mean']:.4f}")
    print(f"  Std R²:  {model.metrics['cv_r2_std']:.4f}")
    print(f"  95% CI:  [{model.metrics['cv_r2_mean'] - 1.96*model.metrics['cv_r2_std']:.4f}, "
          f"{model.metrics['cv_r2_mean'] + 1.96*model.metrics['cv_r2_std']:.4f}]")
    
    return model.metrics


# ============================================================================
# EXAMPLE 10: Generate Intervention Recommendations
# ============================================================================

def example_recommendations():
    """Generate actionable recommendations for counselors."""
    from src.predict import generate_predictions
    from src.utils import get_intervention_recommendations
    
    # Generate predictions
    predictions = generate_predictions(
        model_path='models/student_performance_model.pkl',
        data_path='data/student_performance_updated_1000.csv'
    )
    
    # Get recommendations
    recommendations = get_intervention_recommendations(predictions)
    
    # Group by risk level
    for risk_level in ['Critical', 'Intervention Needed', 'Monitor', 'On Track']:
        group = recommendations[recommendations['Risk_Level'] == risk_level]
        print(f"\n{risk_level.upper()} ({len(group)} students):")
        print(f"  Recommendation: {group['Recommendation'].iloc[0]}")
        print(f"  Sample students:")
        for _, row in group.head(3).iterrows():
            print(f"    - {row['StudentID']}: {row['Predicted_Grade']:.1f}")
    
    return recommendations


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == '__main__':
    import sys
    
    examples = {
        '1': ('Full training pipeline', example_train_and_evaluate),
        '2': ('Batch prediction', example_batch_prediction),
        '3': ('Single student prediction', example_single_prediction),
        '4': ('Feature importance', example_feature_importance),
        '5': ('Data exploration', example_exploration),
        '6': ('Custom preprocessing', example_custom_preprocessing),
        '7': ('Model comparison', example_model_comparison),
        '8': ('Critical cases', example_critical_cases),
        '9': ('Cross-validation', example_cross_validation),
        '10': ('Recommendations', example_recommendations),
    }
    
    print("Available Examples:")
    for key, (desc, _) in examples.items():
        print(f"  {key}. {desc}")
    
    print("\nTo run an example:")
    print("  python examples.py 1  (or any number 1-10)")
    print("Or import and call directly:")
    print("  from examples import example_train_and_evaluate")
    print("  model_path, recs = example_train_and_evaluate()")
