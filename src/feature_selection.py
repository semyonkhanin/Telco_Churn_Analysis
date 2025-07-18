# feature_selection.py
"""Functions for feature selection using statistical tests."""
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def select_features_chi2(X, y, categorical_cols, numerical_cols, k='all', plot=True):
    """
    Select features based on chi-square independence test for classification problems.
    
    Parameters:
    -----------
    X : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The target variable (binary).
    categorical_cols : list
        List of categorical column names.
    numerical_cols : list
        List of numerical column names.
    k : int or 'all', optional (default='all')
        Number of top features to select. If 'all', returns scores for all features.
    plot : bool, optional (default=True)
        Whether to plot feature importance scores.
        
    Returns:
    --------
    tuple
        - selected_features : list
            List of selected feature names.
        - feature_scores : pd.DataFrame
            DataFrame with feature names and their chi-square scores.
    """
    # Create a copy of the data
    X_temp = X.copy()
    
    # Chi-square requires non-negative features, so we'll scale numerical features to [0, 1]
    if len(numerical_cols) > 0:
        scaler = MinMaxScaler()
        X_temp[numerical_cols] = scaler.fit_transform(X_temp[numerical_cols])
    
    # Handle categorical features - convert to dummy variables if not already
    X_encoded = pd.get_dummies(X_temp, columns=categorical_cols, drop_first=False)
    
    # Apply chi-square feature selection
    if k == 'all':
        k = X_encoded.shape[1]  # Use all features
    
    # Initialize SelectKBest with chi2
    selector = SelectKBest(chi2, k=k)
    
    # Fit the selector
    selector.fit(X_encoded, y)
    
    # Get scores
    scores = selector.scores_
    
    # Create DataFrame of all feature scores
    feature_scores = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Score': scores,
        'p_value': selector.pvalues_
    }).sort_values('Score', ascending=False)
    
    # Get indices of selected features
    selected_indices = selector.get_support(indices=True)
    
    # Get selected feature names
    selected_features = X_encoded.columns[selected_indices].tolist()
    
    # Get non-selected feature names (the ones being dropped)
    non_selected_features = [feat for feat in X_encoded.columns if feat not in selected_features]
    
    # Map back to original feature names (before one-hot encoding)
    original_cols = categorical_cols + numerical_cols
    
    # Find which original columns are completely dropped
    # A column is considered dropped if none of its one-hot encoded features are selected
    dropped_original_cols = []
    
    for col in original_cols:
        # For numerical columns, check direct presence
        if col in numerical_cols:
            if col not in selected_features:
                dropped_original_cols.append(col)
        # For categorical columns, check if any of its encoded features are selected
        else:
            # Get all encoded features for this categorical column
            encoded_features = [f for f in X_encoded.columns if f.startswith(f"{col}_")]
            # If none of the encoded features were selected, the column is dropped
            if not any(feat in selected_features for feat in encoded_features):
                dropped_original_cols.append(col)
    
    # Plot feature scores if requested
    if plot:
        plt.figure(figsize=(12, 8))
        top_features = feature_scores.head(20)
        sns.barplot(x='Score', y='Feature', data=top_features)
        plt.title('Top 20 Features by Chi-Square Test')
        plt.tight_layout()
        plt.show()
        
        # Plot p-values
        plt.figure(figsize=(12, 8))
        significant_features = feature_scores[feature_scores['p_value'] < 0.05].head(20)
        sns.barplot(x='p_value', y='Feature', data=significant_features)
        plt.title('Top 20 Significant Features (p < 0.05)')
        plt.axvline(x=0.05, color='r', linestyle='--')
        plt.tight_layout()
        plt.show()
    
    # Print feature selection summary
    total_features = len(X_encoded.columns)
    print("\n===== FEATURE SELECTION SUMMARY =====")
    print(f"Selected {len(selected_features)} features out of {total_features} encoded features")
    print(f"Original columns: {len(original_cols)}")
    print(f"Dropped columns: {len(dropped_original_cols)} ({len(dropped_original_cols)/len(original_cols)*100:.1f}%)")
    
    # Print top selected features
    print("\nTOP 10 SELECTED FEATURES:")
    for i, feature in enumerate(selected_features[:10]):
        score = feature_scores[feature_scores['Feature'] == feature]['Score'].values[0]
        p_val = feature_scores[feature_scores['Feature'] == feature]['p_value'].values[0]
        print(f"{i+1}. {feature} (Score: {score:.4f}, p-value: {p_val:.4e})")
    
    # Print dropped original columns
    if dropped_original_cols:
        print("\nDROPPED ORIGINAL COLUMNS:")
        for i, col in enumerate(sorted(dropped_original_cols)):
            # Find the highest score among encoded features of this column
            if col in numerical_cols:
                # For numerical columns, get the direct score
                score = feature_scores[feature_scores['Feature'] == col]['Score'].values[0]
                p_val = feature_scores[feature_scores['Feature'] == col]['p_value'].values[0]
                print(f"{i+1}. {col} (Score: {score:.4f}, p-value: {p_val:.4e})")
            else:
                # For categorical columns, find all encoded features
                encoded_feats = [f for f in X_encoded.columns if f.startswith(f"{col}_")]
                if encoded_feats:
                    # Get scores for these features
                    scores_df = feature_scores[feature_scores['Feature'].isin(encoded_feats)]
                    best_score = scores_df['Score'].max()
                    worst_p_val = scores_df['p_value'].min()
                    print(f"{i+1}. {col} (Best Score: {best_score:.4f}, Best p-value: {worst_p_val:.4e})")
                else:
                    print(f"{i+1}. {col} (No encoded features found)")
    
    return selected_features, feature_scores

def create_feature_mask(original_features, selected_features):
    """
    Create a boolean mask for the original features based on selected features.
    
    Parameters:
    -----------
    original_features : list
        List of original feature names.
    selected_features : list
        List of selected feature names.
        
    Returns:
    --------
    list
        A boolean mask where True indicates feature is selected.
    """
    return [feature in selected_features for feature in original_features]

def apply_feature_selection(X, categorical_cols, numerical_cols, selected_features):
    """
    Apply feature selection to drop non-selected features from DataFrame.
    
    Parameters:
    -----------
    X : pd.DataFrame
        The feature matrix.
    categorical_cols : list
        List of categorical column names.
    numerical_cols : list
        List of numerical column names.
    selected_features : list
        List of selected feature names.
        
    Returns:
    --------
    tuple
        - X_selected : pd.DataFrame
            DataFrame with only selected features.
        - categorical_cols_selected : list
            Updated list of categorical columns.
        - numerical_cols_selected : list
            Updated list of numerical columns.
    """
    # Create a copy of the data
    X_selected = X.copy()
    
    # Get all column names
    all_cols = categorical_cols + numerical_cols
    
    # Create a mask for columns to keep
    keep_mask = [col in selected_features for col in all_cols]
    
    # Filter columns to keep only selected features
    selected_cols = [col for i, col in enumerate(all_cols) if keep_mask[i]]
    X_selected = X_selected[selected_cols]
    
    # Update categorical and numerical column lists
    categorical_cols_selected = [col for col in categorical_cols if col in selected_cols]
    numerical_cols_selected = [col for col in numerical_cols if col in selected_cols]
    
    # Calculate how many columns were dropped from each type
    dropped_categorical = len(categorical_cols) - len(categorical_cols_selected)
    dropped_numerical = len(numerical_cols) - len(numerical_cols_selected)
    
    print("\n===== APPLIED FEATURE SELECTION =====")
    print(f"Selected {len(selected_cols)} original columns out of {len(all_cols)}")
    print(f"- Categorical: {len(categorical_cols_selected)}/{len(categorical_cols)} kept ({dropped_categorical} dropped)")
    print(f"- Numerical: {len(numerical_cols_selected)}/{len(numerical_cols)} kept ({dropped_numerical} dropped)")
    
    # Print list of dropped columns by type
    if dropped_categorical > 0:
        dropped_cat_cols = [col for col in categorical_cols if col not in categorical_cols_selected]
        print("\nDropped categorical columns:")
        for i, col in enumerate(sorted(dropped_cat_cols)):
            print(f"{i+1}. {col}")
    
    if dropped_numerical > 0:
        dropped_num_cols = [col for col in numerical_cols if col not in numerical_cols_selected]
        print("\nDropped numerical columns:")
        for i, col in enumerate(sorted(dropped_num_cols)):
            print(f"{i+1}. {col}")
    
    return X_selected, categorical_cols_selected, numerical_cols_selected