# Quick Reference Guide

## One-Minute Overview

**Goal**: Predict student final grades (0-100) to identify at-risk students (<70)

**Data**: 1000 students with 9 features (attendance, study hours, previous grades, parental support, etc.)

**Models**: Linear Regression, Decision Trees, Random Forest (best performer, R² ~0.82)

## Common Commands

### Train a model
```bash
python train_model.py
```
Outputs: `models/student_performance_model.pkl`

### Predict on new data
```bash
python predict.py --input data/new_students.csv --output predictions.csv
```

### Interactive exploration
```bash
jupyter notebook notebooks/Student_Performance_Prediction.ipynb
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `src/preprocess.py` | Load, clean, encode, scale data |
| `src/train.py` | Train models, evaluate, identify at-risk |
| `src/predict.py` | Generate predictions on new data |
| `src/utils.py` | Utilities: validation, recommendations, reports |

## Important Patterns

### Preprocessing
```python
from src.preprocess import preprocess_student_data

X, y, preprocessor = preprocess_student_data('data/student_performance_updated_1000.csv')
# Returns: cleaned features X, target y, fitted preprocessor
```

### Training
```python
from src.train import train_and_evaluate

results, X_train, X_test, y_train, y_test = train_and_evaluate(X, y)
# Returns: trained models, predictions, at-risk analysis
```

### Prediction
```python
from src.predict import StudentPerformancePredictor

predictor = StudentPerformancePredictor('models/student_performance_model.pkl')
predictions = predictor.predict('data/new_students.csv', output_path='predictions.csv')
```

## At-Risk Categories

- **Critical** (<50): Immediate tutoring required
- **Intervention** (50-70): Additional support programs
- **Monitor** (70-85): Regular check-ins
- **On Track** (85+): Maintain performance

## Files to Know

- **Data**: `data/student_performance_updated_1000.csv` (1000 students)
- **Notebook**: `notebooks/Student_Performance_Prediction.ipynb` (EDA & experimentation)
- **Model**: `models/student_performance_model.pkl` (trained Random Forest)
- **Instructions**: `.github/copilot-instructions.md` (AI agent guidelines)
- **README**: `README.md` (full documentation)

## Quick Debugging

| Problem | Solution |
|---------|----------|
| Model not found | Run `python train_model.py` first |
| Input features don't match | Check column names in CSV match dataset |
| Poor predictions | Review feature distributions in notebook |
| Missing values | Preprocessor handles automatically |

## Data Dictionary

| Feature | Type | Range | Notes |
|---------|------|-------|-------|
| StudentID | int | 1+ | Dropped before modeling |
| Name | string | - | Dropped before modeling |
| Gender | categorical | M/F | OneHotEncoded |
| AttendanceRate | float | 0-100 | Strong correlation with grades (~0.7) |
| StudyHoursPerWeek | float | 0+ | Key predictor |
| PreviousGrade | float | 0-100 | Important baseline |
| ExtracurricularActivities | int | 0-5 | Engagement proxy |
| ParentalSupport | categorical | High/Med/Low | OneHotEncoded, crucial for at-risk |
| Online Classes Taken | bool | T/F | Binary encoded |
| **FinalGrade** | **float** | **0-100** | **TARGET VARIABLE** |

## Feature Importance (Random Forest)

Top 3 most predictive features typically:
1. PreviousGrade (~0.35 importance)
2. StudyHoursPerWeek (~0.25 importance)
3. AttendanceRate (~0.20 importance)

## Performance Metrics

- **MAE** (Mean Absolute Error): ~7 points on 0-100 scale
- **RMSE** (Root Mean Square Error): ~9 points
- **R²** (Coefficient of Determination): ~0.82 (target: >0.75)

## Reproducibility

All key operations use `random_state=42` for consistent results:
- Model initialization
- Train/test split (80/20)
- Cross-validation (5-fold)

## Next Steps

1. **Explore**: Open the Jupyter notebook
2. **Train**: Run `python train_model.py`
3. **Evaluate**: Check feature importance and at-risk students
4. **Deploy**: Use `python predict.py` for new predictions

---

For details, see [README.md](README.md) or `.github/copilot-instructions.md`
