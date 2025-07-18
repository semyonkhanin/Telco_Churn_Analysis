# -*- coding: utf-8 -*-
"""
Data Cleaning Module for Telco Customer Churn Dataset

This module handles the initial data cleaning and preprocessing of the raw telco customer churn dataset.
It performs the following operations:
- Handles missing values
- Maps categorical variables
- Creates derived features
- Performs outlier detection
- Generates data quality visualizations
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw telco dataset from CSV file."""
    return pd.read_csv(filepath)

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean numeric columns in the dataset.
    
    - Removes ID columns if they exist
    - Converts TotalCharges to numeric
    - Handles missing values
    """
    df_clean = df.copy()
    
    # Remove ID columns if they exist
    id_columns = ['id', 'ID', 'CustomerId', 'customerID', 'customer_id']
    for col in id_columns:
        if col in df_clean.columns:
            df_clean = df_clean.drop(col, axis=1)
            print(f"Dropped ID column: {col}")
    
    # Convert TotalCharges to numeric, replacing empty spaces with NaN
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Fill NaN values with 0 (assuming these are new customers)
    df_clean['TotalCharges'].fillna(0, inplace=True)
    
    return df_clean

def standardize_service_indicators(df: pd.DataFrame, service_cols: List[str]) -> pd.DataFrame:
    """
    Standardize service indicator values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    service_cols : List[str]
        List of service-related column names
    """
    df_clean = df.copy()
    for col in service_cols:
        df_clean[col] = df_clean[col].replace({'No phone service': 'No', 'No internet service': 'No'})
    return df_clean

def encode_categorical_variables(df: pd.DataFrame, binary_map: Dict) -> pd.DataFrame:
    """
    Encode categorical variables using provided mapping.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    binary_map : Dict
        Dictionary mapping categorical values to binary (0/1)
    """
    df_clean = df.copy()
    
    # Apply binary mapping
    for col in df_clean.select_dtypes(include=['object']):
        df_clean[col] = df_clean[col].map(binary_map).fillna(df_clean[col])
    
    return df_clean

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new derived features from existing ones."""
    df_clean = df.copy()
    
    # Calculate average monthly charges
    df_clean['AvgMonthlyCharges'] = df_clean['TotalCharges'] / df_clean['tenure'].replace(0, 1)
    
    # Create contract duration category
    df_clean['ContractDuration'] = df_clean['Contract'].map({
        'Month-to-month': 1,
        'One year': 12,
        'Two year': 24
    })
    
    return df_clean

def detect_and_visualize_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    """Detect outliers using z-score method and create visualizations."""
    z_scores = df[numeric_cols].apply(zscore)
    threshold = 3
    outliers = (np.abs(z_scores) > threshold)
    df_outliers = df[outliers.any(axis=1)]
    
    print(f"Z-score method: {df_outliers.shape[0]} outliers detected")
    
    # Create outlier visualizations
    for col in numeric_cols:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[col], orient='h')
        plt.title(f'Boxplot for Outlier Detection in {col}')
        plt.show()

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Plot correlation matrix heatmap."""
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def run_full_cleaning_pipeline(
    df: pd.DataFrame,
    service_cols: List[str],
    binary_map: Dict
) -> pd.DataFrame:
    """
    Run the complete cleaning pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw input dataframe
    service_cols : List[str]
        List of service-related column names
    binary_map : Dict
        Dictionary mapping categorical values to binary (0/1)
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    df = clean_numeric_columns(df)
    df = standardize_service_indicators(df, service_cols)
    df = encode_categorical_variables(df, binary_map)
    df = create_derived_features(df)
    return df

def main():
    """Main execution function for data cleaning pipeline."""
    # Configuration
    INPUT_FILE = "telco_customer_churn_original.csv"
    OUTPUT_FILE = "clean_telco_dataset.csv"
    
    binary_map = {
        "Yes": 1,
        "No": 0,
        "Female": 0,
        "Male": 1
    }
    
    service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies"
    ]
    
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Execute cleaning pipeline
    print("Loading raw data...")
    df = load_raw_data(INPUT_FILE)
    
    print("Cleaning numeric columns...")
    df = clean_numeric_columns(df)
    
    print("Standardizing service indicators...")
    df = standardize_service_indicators(df, service_cols)
    
    print("Encoding categorical variables...")
    df = encode_categorical_variables(df, binary_map)
    
    print("Creating derived features...")
    df = create_derived_features(df)
    
    print("Detecting outliers and generating visualizations...")
    detect_and_visualize_outliers(df, numeric_cols)
    
    print("Generating correlation matrix...")
    plot_correlation_matrix(df)
    
    print(f"Saving cleaned dataset to {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Data cleaning completed successfully!")

if __name__ == "__main__":
    main()