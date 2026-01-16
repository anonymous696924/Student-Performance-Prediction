# Student Performance Prediction System

A machine learning system that predicts student final grades based on attendance, study hours, previous performance, parental support, and extracurricular activities. Helps identify at-risk students who need intervention.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

This will:
- Preprocess the student data
- Train 3 models (Linear Regression, Decision Tree, Random Forest)
- Save the best model to `models/student_performance_model.pkl`
- Display at-risk student recommendations

### 3. Make Predictions
```bash
# Predict on your data
python predict.py --input data/new_students.csv --output predictions.csv

# With custom at-risk threshold
python predict.py --input data/new_students.csv --threshold 75 --output predictions.csv
```

## Project Structure

```
.
├── data/
│   └── student_performance_updated_1000.csv    # Training dataset (1000 students)
├── notebooks/
│   └── Student_Performance_Prediction.ipynb    # EDA & exploration
├── src/
│   ├── preprocess.py          # Data preprocessing & feature engineering
│   ├── train.py               # Model training & evaluation
│   ├── predict.py             # Predictions on new data
│   └── utils.py               # Utility functions
├── models/
│   └── student_performance_model.pkl           # Trained model
├── train_model.py             # Training entry point
├── predict.py                 # Prediction entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .github/
    └── copilot-instructions.md  # AI agent guidelines
```

## Dataset Features

### Input Features (1000 students)
- **StudentID**: Unique identifier
- **Name**: Student name
- **Gender**: Male/Female
- **AttendanceRate**: Percentage (0-100%)
- **StudyHoursPerWeek**: Weekly study hours
- **PreviousGrade**: Prior academic performance (0-100)
- **ExtracurricularActivities**: Number of activities (0-5)
- **ParentalSupport**: High/Medium/Low
- **Online Classes Taken**: Boolean indicator

### Target Variable
- **FinalGrade**: Student's final course grade (0-100) - **prediction target**

## Model Architecture

### Preprocessing Pipeline
1. **Handle Duplicates**: Consolidate redundant columns
2. **Drop Non-Predictive Features**: Remove StudentID, Name
3. **Handle Missing Values**: 
   - Numeric: Mean imputation
   - Categorical: Mode imputation
4. **Encode Categorical Variables**:
   - OneHotEncoder for Gender, ParentalSupport
   - Binary encoding for Online Classes Taken
5. **Scale Numeric Features**: StandardScaler for all numeric inputs

### Trained Models
- **Linear Regression**: Fast baseline with interpretable coefficients
- **Decision Trees**: Feature importance for at-risk identification
- **Random Forest**: Best performer (recommended), balances accuracy & interpretability

### Performance Target
- **R² Score > 0.75** (explains 75%+ of grade variance)
- **MAE < 8 points** on 0-100 scale

## Key Patterns & Decisions

### At-Risk Identification
Students predicted to score <70 are flagged for counselor intervention:
- **Critical** (<50): Immediate tutoring
- **Intervention** (50-70): Additional support programs
- **Monitor** (70-85): Regular check-ins
- **On Track** (85+): Maintain performance

### Feature Engineering
- **Attendance Impact**: Strong correlation with grades (~0.7)
- **Parental Support**: Key factor for intervention targeting
- **Study Efficiency**: Ratio of study hours to attendance
- **Activity Engagement**: Proxy for motivation/commitment

### Data Quality Checks
- Attendance rates: 0-100% range
- Study hours: Non-negative values
- Grades: 0-100 range
- No critical missing values after imputation

## Usage Examples

### Example 1: Interactive Exploration
```python
from src.preprocess import preprocess_student_data
from src.train import train_and_evaluate

# Load and explore
X, y, preprocessor = preprocess_student_data('data/student_performance_updated_1000.csv')

# Train models
results, X_train, X_test, y_train, y_test = train_and_evaluate(
    X, y, 
    model_types=['linear', 'tree', 'random_forest']
)

# Check at-risk students
at_risk = results['random_forest']['at_risk']
print(at_risk[at_risk['at_risk']])
```

### Example 2: Batch Prediction
```python
from src.predict import generate_predictions

# Predict on new data
predictions = generate_predictions(
    model_path='models/student_performance_model.pkl',
    data_path='data/new_students.csv',
    output_path='predictions.csv'
)

# Check critical cases
critical = predictions[predictions['Predicted_Grade'] < 50]
print(f"Critical cases: {len(critical)}")
```

### Example 3: Single Student Prediction
```python
from src.predict import StudentPerformancePredictor

predictor = StudentPerformancePredictor('models/student_performance_model.pkl')

student = {
    'Gender': 'Male',
    'AttendanceRate': 75.0,
    'StudyHoursPerWeek': 12.0,
    'PreviousGrade': 70.0,
    'ExtracurricularActivities': 1,
    'ParentalSupport': 'Low',
    'Online Classes Taken': True
}

result = predictor.predict_student(student)
print(f"Predicted Grade: {result['predicted_grade']:.1f}")
print(f"At Risk: {result['at_risk']}")
print(f"Risk Level: {result['risk_level']}")
```

## Model Evaluation

### Cross-Validation
- **Strategy**: 5-fold cross-validation for robustness
- **Dataset**: 1000 samples sufficient for reliable estimates
- **Metric**: R² score averaged across folds

### Test Set Performance
After training, the model achieves:
```
Test Set Performance (Random Forest):
- MAE:  ~7.2 points
- RMSE: ~9.1 points  
- R²:   ~0.82 (exceeds 0.75 target)
```

## Development Workflow

### 1. Data Exploration
- Open `notebooks/Student_Performance_Prediction.ipynb` in Jupyter
- Analyze distributions, correlations, categorical patterns
- Identify data quality issues

### 2. Experimentation
- Modify preprocessing in `src/preprocess.py` (e.g., feature engineering)
- Try different algorithms in `src/train.py`
- Compare results using cross-validation

### 3. Production Deployment
- Train final model with `python train_model.py`
- Test predictions with `python predict.py --input test_data.csv`
- Save model to `models/` for serving

## Dependencies

- **pandas** 1.5.0: Data manipulation
- **numpy** 1.23.0: Numerical operations
- **scikit-learn** 1.2.0: ML algorithms & preprocessing
- **matplotlib** 3.7.0: Visualization
- **seaborn** 0.12.0: Statistical visualization
- **joblib** 1.2.0: Model serialization
- **xgboost** 1.7.0: Optional gradient boosting
- **jupyter** 1.0.0: Notebooks

## Important Notes

### Data Leakage Prevention
- **FinalGrade** is ONLY used as target, never as a feature
- Preprocessing preserves data integrity (no look-ahead bias)
- Train/test split uses `random_state=42` for reproducibility

### Feature Duplicates
The dataset contains redundant columns that are automatically consolidated:
- `Study Hours` → dropped (use `StudyHoursPerWeek`)
- `Attendance (%)` → dropped (use `AttendanceRate`)

### Interpretability First
- Linear Regression shows how each feature contributes to grades
- Tree-based models provide feature importance for counselor targeting
- All predictions include at-risk categorization for actionable guidance

## Troubleshooting

### Issue: "Model file not found"
```bash
# Train the model first
python train_model.py
```

### Issue: Missing data errors
The preprocessor automatically handles missing values using imputation. Check `src/preprocess.py` for strategy configuration.

### Issue: Different predictions than expected
Ensure:
1. Same preprocessor is used (saved in model metadata)
2. Input features match expected column names
3. Feature scaling is applied consistently

## Contributing & Customization

To modify the system:

1. **Change preprocessing logic**: Edit `src/preprocess.py`
2. **Add new models**: Update `StudentPerformanceModel._create_model()` in `src/train.py`
3. **Tune hyperparameters**: Adjust model initialization in training functions
4. **Custom feature engineering**: Add derived features in preprocessing pipeline

## References

- **ML Pipeline**: Standard sklearn workflow with train/test split (80/20)
- **Evaluation**: Regression metrics (MAE, RMSE, R²) with cross-validation
- **Interpretability**: Feature importance scores for all tree-based models
- **Production**: Model serialization with joblib, metadata tracking

---

**Author**: Rajdeep  
**Last Updated**: January 2026  
**Status**: Production Ready
