#!/usr/bin/env python
"""
Main training script for student performance prediction model.
Usage: python train_model.py
"""

import sys
from pathlib import Path
from src.preprocess import preprocess_student_data
from src.train import train_and_evaluate
from src.utils import get_intervention_recommendations

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / 'student_performance_updated_1000.csv'
MODEL_DIR = PROJECT_ROOT / 'models'


def main():
    """Train and evaluate models."""
    
    print("\n" + "="*70)
    print("STUDENT PERFORMANCE PREDICTION - MODEL TRAINING")
    print("="*70)
    
    # Load and preprocess data
    print("\n[1/3] Loading and preprocessing data...")
    X, y, preprocessor = preprocess_student_data(str(DATA_PATH))
    
    # Train models
    print("\n[2/3] Training models...")
    results, X_train, X_test, y_train, y_test = train_and_evaluate(
        X, y,
        model_types=['linear', 'tree', 'random_forest']
    )
    
    # Save best model
    print("\n[3/3] Saving best model...")
    best_model = results['random_forest']['model']
    model_path = best_model.save_model(str(MODEL_DIR / 'student_performance_model.pkl'))
    
    # Generate intervention recommendations
    print("\n" + "="*70)
    print("INTERVENTION RECOMMENDATIONS")
    print("="*70)
    at_risk = results['random_forest']['at_risk']
    recommendations = get_intervention_recommendations(at_risk)
    
    # Display critical cases
    critical = recommendations[recommendations['Risk_Level'] == 'Critical']
    if len(critical) > 0:
        print(f"\n⚠️  CRITICAL CASES: {len(critical)} students need immediate intervention")
        print(critical[['StudentID', 'Predicted_Grade', 'Recommendation']].to_string())
    
    intervention = recommendations[recommendations['Risk_Level'] == 'Intervention Needed']
    if len(intervention) > 0:
        print(f"\n⚠️  INTERVENTION NEEDED: {len(intervention)} students need support")
    
    print("\n" + "="*70)
    print("✓ Training complete! Model saved to: " + model_path)
    print("="*70)
    
    return model_path


if __name__ == '__main__':
    main()
