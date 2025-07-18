# -*- coding: utf-8 -*-

# preprocessing.py
"""Functions for creating preprocessing pipelines."""
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_preprocessing_pipeline(categorical_cols, numerical_cols):
    """
    Create a preprocessing pipeline for numerical and categorical features.

    This function builds a ColumnTransformer that:
    - Standardizes numerical features using StandardScaler.
    - Applies one-hot encoding to categorical features using OneHotEncoder.

    Parameters:
    -----------
    categorical_cols : list of str
        List of names of categorical columns.
    numerical_cols : list of str
        List of names of numerical columns.

    Returns:
    --------
    sklearn.compose.ColumnTransformer
        A preprocessor that applies appropriate transformations to numeric and categorical features.
    """
    # Categorical features preprocessing
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Numerical features preprocessing
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor