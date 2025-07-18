# -*- coding: utf-8 -*-

# data_processing.py
"""Functions for loading and processing the telco churn dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE

def load_data(filepath):
    """Load the telco churn dataset."""
    df = pd.read_csv(filepath)
    
    # Drop any index column if present
    if df.columns[0] == 'Unnamed: 0':
        df = df.drop(df.columns[0], axis=1)
    
    # Fill missing values
    df.fillna(0, inplace=True)
    
    # Check for data imbalance
    print(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True)}")
    
    return df

def feature_engineering(df):
    """
    Generate new features that may improve model performance.

    - Adds a ratio of TotalCharges to MonthlyCharges (with smoothing).
    - Adds squared MonthlyCharges.
    - Buckets tenure into categorical tenure groups.
    - Drops rows with missing tenure values.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with additional engineered features.
    """
    # Creating interaction terms or domain-specific features
    if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
        # Create ratio feature
        df['Charges_Ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)  # Adding 1 to avoid division by zero
        
        # Create polynomial features
        df['MonthlyCharges_Squared'] = df['MonthlyCharges'] ** 2
    
    # If we have a tenure column
    if 'tenure' in df.columns:
        # Drop rows where 'tenure' is NaN
        df = df.dropna(subset=['tenure'])
        
        # Bucket tenure into categories
        df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                                  labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'], right=False)
    
    return df

def prepare_data_splits(df, target_col='Churn'):
    """
    Split the dataset into training, validation, and holdout test sets. Also identify categorical and numerical columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The preprocessed DataFrame with features and target.
    target_col : str, optional (default='Churn')
        The name of the target column.

    Returns:
    --------
    tuple
        A tuple containing:
        - X_train : pd.DataFrame
        - X_val : pd.DataFrame
        - X_holdout : pd.DataFrame
        - y_train : pd.Series
        - y_val : pd.Series
        - y_holdout : pd.Series
        - categorical_cols : list of str
        - numerical_cols : list of str
    """
    # Separate features and target
    X = df.drop(columns=target_col)
    y = df[target_col]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")
    
    # First split: separate out the holdout test set
    X_temp, X_holdout, y_temp, y_holdout = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    # Second split: create validation set from the remaining data
    # Adjusted validation size to account for first split
    adjusted_val_size = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_size, stratify=y_temp, random_state=RANDOM_STATE
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")
    print(f"Holdout test set size: {X_holdout.shape[0]} samples")
    
    return X_train, X_val, X_holdout, y_train, y_val, y_holdout, categorical_cols, numerical_cols