# -*- coding: utf-8 -*-
"""Main script for Telco Customer Churn prediction."""
# Import modules
from data_processing import load_data, feature_engineering, prepare_data_splits
from preprocessing import create_preprocessing_pipeline
from feature_selection import select_features_chi2, apply_feature_selection
from model_building import build_and_evaluate_models, build_stacking_ensemble, evaluate_on_holdout_test
from visualization import analyze_feature_importance, plot_roc_curves, plot_confusion_matrices
from cleaning import run_full_cleaning_pipeline

def main(raw_data=True, use_feature_selection=True, num_features=None):
    """
    Main execution function for the churn prediction pipeline.

    This function performs the entire machine learning workflow, including:
    - Loading and preprocessing the Telco churn dataset.
    - Performing feature engineering.
    - Optional feature selection using chi-square test.
    - Splitting the data into training, validation, and holdout test sets.
    - Creating a preprocessing pipeline for categorical and numerical features.
    - Training and evaluating multiple classification models (e.g., Logistic Regression, Random Forest, etc.).
    - Analyzing feature importance for selected models.
    - Building a stacking ensemble using the top-performing models.
    - Visualizing model performance with ROC curves and confusion matrices.
    - Evaluating all models on the unseen holdout test set.

    Parameters:
    -----------
    raw_data : bool, optional (default=True)
        Whether to process raw data or use already cleaned data
    use_feature_selection : bool, optional (default=True)
        Whether to use chi-square feature selection
    num_features : int or None, optional (default=None)
        Number of top features to select. If None, an optimal number is determined

    Returns:
    dict: A dictionary containing the following:
        - 'validation_results': pd.DataFrame with model performance metrics on the validation set.
        - 'test_results': pd.DataFrame with model performance metrics on the holdout test set.
        - 'best_models': dict of {model_name: trained_pipeline} for top models.
        - 'stacking_model': Trained stacking ensemble pipeline.
        - 'all_model_results': dict with detailed metrics for all models.
        - 'ensemble_results': dict with metrics for the ensemble on validation data.
        - 'feature_importance_logistic': pd.DataFrame with feature importances for Logistic Regression.
        - 'feature_importance_rf': pd.DataFrame with feature importances for Random Forest.
        - 'feature_scores': pd.DataFrame with chi-square scores for features (if feature selection used).
        - 'data': dict with train/validation/test splits:
            - 'X_train', 'X_val', 'X_holdout': pd.DataFrames of features.
            - 'y_train', 'y_val', 'y_holdout': pd.Series of labels.
    """
    if raw_data:
        print("\n===== Loading and Cleaning Raw Data =====")
        # Load raw data
        df = load_data("telco_customer_churn_original.csv")
        
        # Define cleaning parameters
        service_cols = [
            "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies"
        ]
        
        binary_map = {
            "Yes": 1,
            "No": 0,
            "Female": 0,
            "Male": 1
        }
        
        # Run cleaning pipeline
        print("Cleaning data...")
        df = run_full_cleaning_pipeline(df, service_cols, binary_map)
        
        # Save cleaned data
        df.to_csv("clean_telco_dataset.csv", index=False)
        print("Cleaned data saved to clean_telco_dataset.csv")
    else:
        print("\n===== Loading Pre-cleaned Data =====")
        df = load_data("clean_telco_dataset.csv")

    # Perform feature engineering
    df = feature_engineering(df)
    
    # Split data into train, validation, and holdout test sets
    X_train, X_val, X_holdout, y_train, y_val, y_holdout, categorical_cols, numerical_cols = prepare_data_splits(df)
    
    feature_scores = None
    
    # Apply feature selection if enabled
    if use_feature_selection:
        print("\n===== Applying Chi-Square Feature Selection =====")
        
        # If num_features is not provided, we'll use 'all' to get scores for all features first
        initial_k = 'all' if num_features is None else num_features
        
        # Select features based on chi-square test
        selected_features, feature_scores = select_features_chi2(
            X_train, y_train, categorical_cols, numerical_cols, k=initial_k
        )
        
        # If num_features is None, we'll determine an optimal number based on p-values
        if num_features is None:
            # Count features with p-value < 0.05
            significant_features = feature_scores[feature_scores['p_value'] < 0.05]
            num_significant = len(significant_features)
            
            # Use at least 10 features or the number of significant features, whichever is larger
            optimal_k = max(10, num_significant)
            print(f"\nAutomatically selecting top {optimal_k} features based on significance")
            
            # Re-select features with optimal k
            selected_features, feature_scores = select_features_chi2(
                X_train, y_train, categorical_cols, numerical_cols, k=optimal_k, plot=True
            )
        
        # Apply the feature selection to the datasets
        X_train, categorical_cols, numerical_cols = apply_feature_selection(
            X_train, categorical_cols, numerical_cols, selected_features
        )
        
        X_val, _, _ = apply_feature_selection(
            X_val, categorical_cols, numerical_cols, selected_features
        )
        
        X_holdout, _, _ = apply_feature_selection(
            X_holdout, categorical_cols, numerical_cols, selected_features
        )
        
        print("\nAfter feature selection:")
        print(f"X_train shape: {X_train.shape}")
        print(f"Categorical columns: {len(categorical_cols)}")
        print(f"Numerical columns: {len(numerical_cols)}")
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(categorical_cols, numerical_cols)
    
    # Build and evaluate models using training and validation sets
    results_df, best_models, all_results = build_and_evaluate_models(
        X_train, X_val, y_train, y_val, preprocessor
    )
    
    # Print model validation results
    print("\nModel Validation Performance Results:")
    print(results_df)
    
    # Analyze feature importance for top models
    feature_importance_logistic = analyze_feature_importance(
        best_models['Logistic Regression'], X_train, preprocessor, categorical_cols, numerical_cols
    )
    
    feature_importance_rf = analyze_feature_importance(
        best_models['Random Forest'], X_train, preprocessor, categorical_cols, numerical_cols
    )
    
    # Build stacking ensemble using the best models
    stacking_model, ensemble_results = build_stacking_ensemble(
        X_train, X_val, y_train, y_val, best_models
    )
    
    # Final evaluation on holdout test set
    test_results_df = evaluate_on_holdout_test(best_models, stacking_model, X_holdout, y_holdout)
    
    # Print final test results
    print("\nFinal Test Performance Results:")
    print(test_results_df)
    
    # Plot ROC curves on test data
    all_models = {**best_models, "Stacking Ensemble": stacking_model}
    plot_roc_curves(all_models, X_holdout, y_holdout)
    
    # Plot confusion matrices on test data
    plot_confusion_matrices(all_models, X_holdout, y_holdout)
    
    # Return all results for further analysis
    return {
        'validation_results': results_df,
        'test_results': test_results_df,
        'best_models': best_models,
        'stacking_model': stacking_model,
        'all_model_results': all_results,
        'ensemble_results': ensemble_results,
        'feature_importance_logistic': feature_importance_logistic,
        'feature_importance_rf': feature_importance_rf,
        'feature_scores': feature_scores,
        'data': {
            'X_train': X_train,
            'X_val': X_val,
            'X_holdout': X_holdout,
            'y_train': y_train,
            'y_val': y_val,
            'y_holdout': y_holdout
        }
    }

if __name__ == "__main__":
    # Run with raw data cleaning and feature selection
    results = main(raw_data=True, use_feature_selection=True)
    
    # To use pre-cleaned data:
    #results = main(raw_data=False, use_feature_selection=True)
    
    # To run with specific number of features:
    #results = main(raw_data=True, use_feature_selection=True, num_features=15)