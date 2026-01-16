# AI Coding Agent Instructions - Student Performance Prediction

## Project Overview
A machine learning system that predicts student final grades based on multiple input features (attendance, study hours, previous grades, extracurricular activities, parental support) to identify at-risk students needing intervention.

## Data Architecture

### Dataset Structure
The project uses `student_performance_updated_1000.csv` with 1000 student records:

**Input Features:**
- `StudentID` (int): Unique identifier
- `Name` (string): Student name
- `Gender` (categorical): Male/Female
- `AttendanceRate` (float): Percentage (0-100)
- `StudyHoursPerWeek` (float): Numeric hours
- `PreviousGrade` (float): Prior academic performance
- `ExtracurricularActivities` (int): Count of activities
- `ParentalSupport` (categorical): High/Medium/Low
- `Study Hours` (float): Alternative hours metric
- `Attendance (%)` (float): Duplicate attendance measure
- `Online Classes Taken` (boolean): Course delivery method

**Target Variable:**
- `FinalGrade` (float): Student's final course grade (prediction target)

### Data Handling Patterns
- **Missing Values**: Perform statistical imputation (mean/median) for numeric features; mode for categorical
- **Categorical Encoding**: Use OneHotEncoder for `Gender`, `ParentalSupport`; LabelEncoder for `Online Classes Taken`
- **Feature Scaling**: Apply StandardScaler to numeric features (especially `StudyHoursPerWeek`, `AttendanceRate`)
- **Feature Duplicates**: Resolve redundant columns (`StudyHoursPerWeek` vs `Study Hours`, `AttendanceRate` vs `Attendance (%)`) - investigate source and consolidate before model training

## ML Workflow & Development

### Standard Pipeline
```
Data Loading → Exploration → Preprocessing → Train/Test Split (80/20) 
→ Model Training → Evaluation → Hyperparameter Tuning → Prediction
```

### Recommended Algorithms
- **Primary**: Linear Regression (interpretability for guidance counselors)
- **Baseline**: Decision Trees (feature importance for at-risk identification)
- **Advanced**: Random Forest, Gradient Boosting (improved accuracy for score prediction)

### Evaluation Metrics
- **Regression Metrics**: MAE, RMSE, R² Score (prioritize R² for variance explanation)
- **Business Metric**: MSE for identifying students predicted <70 (intervention threshold)

### Model Training Conventions
```python
# Standard workflow structure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[input_features]
y = df['FinalGrade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Always use random_state=42 for reproducibility
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

## Project-Specific Patterns

### Feature Engineering Priorities
1. **Attendance-Score Relationship**: Strong correlation expected; ensure no scaling causes information loss
2. **Parental Support Impact**: Categorical feature crucial for at-risk identification; validate encoding captures class imbalance
3. **Study Efficiency Ratio**: Consider creating `study_hours / attendance_rate` as derived feature
4. **Activity-Engagement**: Extracurricular activities → engagement proxy; model feature importance

### Model Interpretability Requirements
- Always generate feature importance scores (`.feature_importances_` for tree-based; coefficients for linear)
- Track which features most strongly predict low FinalGrade (<70) for intervention targeting
- Document prediction confidence intervals for guidance decisions

### Validation Approach
- **Cross-Validation**: Use 5-fold CV to ensure robustness with 1000-sample dataset
- **Stratification**: Not applicable (regression target); ensure test set represents full grade range
- **Performance Threshold**: Target R² > 0.75 before deployment

## Development Workflows

### Experimentation Loop
1. **Notebook-First**: Use Jupyter notebooks (`.ipynb`) for exploration and model iteration
2. **Documentation**: Each notebook cell should include markdown explaining "why" (e.g., "Removing outliers >3 SD to avoid prediction skew")
3. **Reproducibility**: Pin model versions, random seeds, and preprocessing step order

### Scripts & Automation
- **`preprocess.py`**: Data cleaning, encoding, scaling (reusable across train/predict)
- **`train.py`**: Model training with hyperparameter search, save trained model as `.pkl`
- **`predict.py`**: Load model, apply preprocessing, generate predictions with confidence metrics
- **Testing**: Unit tests for preprocessing (data shape, dtype), model shape compatibility

### File Organization
```
.
├── data/
│   └── student_performance_updated_1000.csv
├── notebooks/
│   └── Student_Performance_Prediction.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py (helper functions)
├── models/
│   └── trained_model.pkl
├── requirements.txt
└── .github/
    └── copilot-instructions.md
```

## Dependencies & Environment
- **Core**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn (for EDA)
- **Notebooks**: jupyter
- **Optional**: xgboost (for advanced models)

Create `requirements.txt` with pinned versions for reproducibility:
```
pandas==1.5.0
numpy==1.23.0
scikit-learn==1.2.0
jupyter==1.0.0
```

## Key Considerations for AI Agents

- **Imbalanced Features**: Verify all input features contribute meaningfully; drop StudentID/Name before modeling
- **Real-World Application**: Model predictions guide counselor decisions; prioritize interpretability over marginal accuracy gains
- **Intervention Threshold**: Code should surface students predicted <70 separately for counselor prioritization
- **Data Leakage Risk**: Never use FinalGrade-derived features in preprocessing; this is the target variable only
