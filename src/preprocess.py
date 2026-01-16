"""
Data preprocessing module for student performance prediction.
Handles missing values, categorical encoding, feature scaling, and duplicate resolution.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer


class StudentPerformancePreprocessor:
    """
    Preprocesses student performance data following standard ML pipeline.
    Handles missing values, categorical encoding, and feature scaling.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer_numeric = SimpleImputer(strategy='mean')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
        
    def load_data(self, filepath):
        """Load CSV dataset."""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def resolve_duplicate_features(self, df):
        """
        Resolve redundant feature columns.
        - StudyHoursPerWeek: Use primary over 'Study Hours'
        - AttendanceRate: Use primary over 'Attendance (%)'
        """
        df = df.copy()
        
        # Keep StudyHoursPerWeek, drop Study Hours if present
        if 'Study Hours' in df.columns:
            print("Dropping redundant 'Study Hours' column (using StudyHoursPerWeek)")
            df = df.drop('Study Hours', axis=1)
        
        # Keep AttendanceRate, drop Attendance (%) if present
        if 'Attendance (%)' in df.columns:
            print("Dropping redundant 'Attendance (%)' column (using AttendanceRate)")
            df = df.drop('Attendance (%)', axis=1)
        
        return df
    
    def drop_non_predictive_features(self, df):
        """Remove StudentID and Name (non-predictive identifiers)."""
        df = df.copy()
        to_drop = [col for col in ['StudentID', 'Name'] if col in df.columns]
        if to_drop:
            print(f"Dropping non-predictive features: {to_drop}")
            df = df.drop(to_drop, axis=1)
        return df
    
    def fit(self, df):
        """
        Fit preprocessor on training data.
        Learns encoders and scaler parameters.
        """
        df = df.copy()
        df = self.resolve_duplicate_features(df)
        df = self.drop_non_predictive_features(df)
        
        # Identify feature types
        self.numeric_features = df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        self.numeric_features = [f for f in self.numeric_features if f != 'FinalGrade']
        
        self.categorical_features = df.select_dtypes(
            include=['object', 'bool']).columns.tolist()
        
        print(f"Numeric features: {self.numeric_features}")
        print(f"Categorical features: {self.categorical_features}")
        
        # Fit numeric imputer and scaler
        if self.numeric_features:
            self.imputer_numeric.fit(df[self.numeric_features])
            numeric_imputed = self.imputer_numeric.transform(df[self.numeric_features])
            self.scaler.fit(numeric_imputed)
        
        # Fit categorical imputer and encoders
        if self.categorical_features:
            self.imputer_categorical.fit(df[self.categorical_features])
        
        return self
    
    def transform(self, df):
        """
        Apply preprocessing to data.
        Returns feature matrix X and target y (if present).
        """
        df = df.copy()
        df = self.resolve_duplicate_features(df)
        df = self.drop_non_predictive_features(df)
        
        # Handle missing values
        if self.numeric_features:
            df[self.numeric_features] = self.imputer_numeric.transform(
                df[self.numeric_features])
        
        if self.categorical_features:
            df[self.categorical_features] = self.imputer_categorical.transform(
                df[self.categorical_features])
        
        # Scale numeric features
        if self.numeric_features:
            df[self.numeric_features] = self.scaler.transform(
                df[self.numeric_features])
        
        # Encode categorical features (Online Classes Taken: binary)
        if 'Online Classes Taken' in self.categorical_features:
            df['Online Classes Taken'] = df['Online Classes Taken'].astype(int)
        
        # One-hot encode other categorical features
        encoding_features = [f for f in self.categorical_features 
                           if f != 'Online Classes Taken']
        if encoding_features:
            encoded = pd.get_dummies(df[encoding_features], drop_first=True)
            df = df.drop(encoding_features, axis=1)
            df = pd.concat([df, encoded], axis=1)
        
        # Separate features and target
        if 'FinalGrade' in df.columns:
            X = df.drop('FinalGrade', axis=1)
            y = df['FinalGrade']
            return X, y
        else:
            return df, None
    
    def fit_transform(self, df):
        """Fit preprocessor and transform data."""
        self.fit(df)
        X, y = self.transform(df)
        # Remove rows with NaN in target
        if y is not None:
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]
        return X, y


def preprocess_student_data(filepath, train=True):
    """
    Convenience function to load and preprocess student performance data.
    
    Args:
        filepath: Path to student_performance_updated_1000.csv
        train: If True, fit preprocessor; if False, requires fitted preprocessor
        
    Returns:
        X, y: Feature matrix and target vector
        preprocessor: Fitted preprocessor object
    """
    preprocessor = StudentPerformancePreprocessor()
    df = preprocessor.load_data(filepath)
    X, y = preprocessor.fit_transform(df)
    return X, y, preprocessor
