"""
Student Performance Prediction Package
Predicts student grades and identifies at-risk students for intervention.
"""

__version__ = "1.0.0"
__author__ = "AI-Generated"

from src.preprocess import StudentPerformancePreprocessor, preprocess_student_data
from src.train import StudentPerformanceModel, train_and_evaluate
from src.predict import StudentPerformancePredictor, generate_predictions

__all__ = [
    'StudentPerformancePreprocessor',
    'preprocess_student_data',
    'StudentPerformanceModel',
    'train_and_evaluate',
    'StudentPerformancePredictor',
    'generate_predictions',
]
