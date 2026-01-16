#!/usr/bin/env python
"""
Prediction script for generating grades on new student data.
Usage: python predict.py --input <csv_file> --output <output_file>
"""

import sys
import argparse
from pathlib import Path
from src.predict import StudentPerformancePredictor

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """Generate predictions on new student data."""
    
    parser = argparse.ArgumentParser(
        description='Generate student performance predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --input data/new_students.csv
  python predict.py --input data/new_students.csv --output predictions.csv
        """
    )
    
    parser.add_argument('--input', '-i', 
                       default='data/student_performance_updated_1000.csv',
                       help='Path to student data CSV')
    parser.add_argument('--output', '-o',
                       help='Path to save predictions CSV')
    parser.add_argument('--model', '-m',
                       default='models/student_performance_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--threshold', '-t', type=int, default=70,
                       help='Grade threshold for at-risk identification (default: 70)')
    
    args = parser.parse_args()
    
    # Validate files
    input_path = Path(args.input)
    model_path = Path(args.model)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("STUDENT PERFORMANCE PREDICTION")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Model: {model_path}")
    print(f"At-risk threshold: {args.threshold}")
    
    # Generate predictions
    predictor = StudentPerformancePredictor(model_path)
    predictions = predictor.predict(str(input_path), args.output, args.threshold)
    
    # Show summary
    print("\n" + "="*70)
    if args.output:
        print(f"✓ Predictions saved to: {args.output}")
    print("="*70)
    
    return predictions


if __name__ == '__main__':
    main()
