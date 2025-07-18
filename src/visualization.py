# -*- coding: utf-8 -*-
# visualization.py
"""Functions for visualizing model results and feature importance."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from config import PLOTS_DIR
import os

def analyze_feature_importance(best_model, X_train, preprocessor, categorical_cols, numerical_cols):
    """
    Analyze and visualize feature importance or coefficients for a trained model pipeline.

    Parameters:
    - best_model: Pipeline
        The trained pipeline containing a preprocessor and a model.
    - X_train: pd.DataFrame
        The training feature dataset used for fitting the preprocessor.
    - preprocessor: ColumnTransformer
        The preprocessing transformer used within the pipeline.
    - categorical_cols: list
        List of categorical column names used in the preprocessor.
    - numerical_cols: list
        List of numerical column names used in the preprocessor.

    Returns:
    - pd.DataFrame or None:
        A DataFrame with feature importance or coefficients, or None if unsupported model.
    """
    # Get the model name
    model_name = type(best_model.named_steps['model']).__name__
    
    # First, fit preprocessor on the entire training set to ensure correct feature extraction
    X_preprocessed = preprocessor.fit_transform(X_train)
    
    # Get feature names after preprocessing
    feature_names = []
    
    # Add numerical feature names
    if len(numerical_cols) > 0:
        feature_names.extend(numerical_cols)
    
    # Add one-hot encoded feature names
    if len(categorical_cols) > 0:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        encoded_features = ohe.get_feature_names_out(categorical_cols)
        feature_names.extend(encoded_features)
    
    # For Random Forest and tree-based models
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        return _plot_tree_importances(best_model, feature_names, model_name, X_preprocessed)
    
    # For Logistic Regression
    elif hasattr(best_model.named_steps['model'], 'coef_'):
        return _plot_logistic_coefficients(best_model, feature_names, model_name, X_preprocessed)
    
    else:
        print(f"Feature importance analysis not implemented for {model_name}")
        return None

def _plot_tree_importances(model, feature_names, model_name, X_preprocessed):
    """
    Plot feature importances for tree-based models (e.g., Random Forest, XGBoost).

    Parameters:
    - model: Pipeline
        Trained pipeline with a tree-based model.
    - feature_names: list
        List of feature names after preprocessing.
    - model_name: str
        Name of the model (for title labeling).
    - X_preprocessed: np.ndarray
        Preprocessed feature matrix for debugging and shape matching.

    Returns:
    - pd.DataFrame:
        DataFrame containing features and their importance scores.
    """
    # Get feature importances
    importances = model.named_steps['model'].feature_importances_
    
    # Handle potential mismatch between feature names and importances
    if len(importances) != len(feature_names):
        feature_names = _adjust_feature_names(feature_names, importances, X_preprocessed)
    
    # Create DataFrame of feature importances
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plot top 20 feature importances
    plt.figure(figsize=(12, 8))
    top_features = feature_imp_df.head(20)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top 20 Feature Importances ({model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name}_feature_importance.png'))
    plt.close()
    
    return feature_imp_df

def _plot_logistic_coefficients(model, feature_names, model_name, X_preprocessed):
    """
   Plot coefficients for linear models (e.g., Logistic Regression, SGDClassifier).

   Parameters:
   - model: Pipeline
       Trained pipeline with a linear model.
   - feature_names: list
       List of feature names after preprocessing.
   - model_name: str
       Name of the model (for title labeling).
   - X_preprocessed: np.ndarray
       Preprocessed feature matrix for debugging and shape matching.

   Returns:
   - pd.DataFrame:
       DataFrame containing feature coefficients and odds ratios.
   """
    # Get coefficients
    coefficients = model.named_steps['model'].coef_[0]
    
    # Handle potential mismatch between feature names and coefficients
    if len(coefficients) != len(feature_names):
        feature_names = _adjust_feature_names(feature_names, coefficients, X_preprocessed)
    
    # Calculate odds ratios
    odds_ratios = np.exp(coefficients)
    
    # Create DataFrame of coefficients with Odds Ratios
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients),
        'Odds_Ratio': odds_ratios
    }).sort_values(by='Abs_Coefficient', ascending=False)
    
    # Plot top 20 coefficients
    plt.figure(figsize=(12, 8))
    top_features = coef_df.head(20)
    colors = ['red' if c < 0 else 'green' for c in top_features['Coefficient']]
    sns.barplot(x='Coefficient', y='Feature', data=top_features, palette=colors)
    plt.title(f'Top 20 Feature Coefficients ({model_name})')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name}_coefficients.png'))
    plt.close()
    
    return coef_df

def _adjust_feature_names(feature_names, coefficients, X_preprocessed):
    """
    Adjust feature names to match the number of model coefficients or importances.

    Parameters:
    - feature_names: list
        List of original feature names after preprocessing.
    - coefficients: np.ndarray
        Coefficients or feature importances from the model.
    - X_preprocessed: np.ndarray
        Preprocessed feature matrix for shape verification.

    Returns:
    - list:
        Adjusted list of feature names matching the number of coefficients/importances.
    """
    print(f"WARNING: Number of features ({len(feature_names)}) doesn't match " 
          f"number of coefficients/importances ({len(coefficients)})")
    
    # Debug information
    num_features = X_preprocessed.shape[1]
    print(f"Number of features after preprocessing: {num_features}")
    
    # Adjust feature_names
    if len(feature_names) > len(coefficients):
        feature_names = feature_names[:len(coefficients)]
        print("Truncated feature names to match coefficients/importances")
    else:
        # Create generic names for extras
        additional_names = [f"Feature_{i}" for i in range(len(feature_names), len(coefficients))]
        feature_names.extend(additional_names)
        print("Extended feature names to match coefficients/importances")
    
    return feature_names

def plot_roc_curves(best_models, X_test, y_test):
    """
    Plot ROC curves for multiple trained classifiers.

    Parameters:
    - best_models: dict
        Dictionary of {model_name: trained_pipeline}.
    - X_test: pd.DataFrame or np.ndarray
        Test features.
    - y_test: pd.Series or np.ndarray
        True binary labels.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 8))
    
    for name, model in best_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curves.png'))
    plt.close()

def plot_confusion_matrices(best_models, X_test, y_test):
    """
    Plot confusion matrices for multiple trained classifiers.
    """
    n_models = len(best_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (name, model) in enumerate(best_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Create custom annotation texts for all cells
        annot = np.array([[str(val) for val in row] for row in cm])
        
        # Set up the heatmap with annotations forced
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=axes[i], 
                   cbar=True, annot_kws={"size": 12})
        
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        
        # Add explicit labels
        axes[i].set_xticklabels(['Negative', 'Positive'])
        axes[i].set_yticklabels(['Negative', 'Positive'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrices.png'))
    plt.close()