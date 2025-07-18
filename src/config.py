# -*- coding: utf-8 -*-

"""Configuration parameters for the churn prediction model."""
import warnings
warnings.filterwarnings('ignore')
import os

# Random state for reproducibility
RANDOM_STATE = 42

# Data splitting parameters
TEST_SIZE = 0.1        # Holdout test set size (10% of total data)
VALIDATION_SIZE = 0.15  # Validation set size (15% of total data)

# Feature selection parameters
USE_FEATURE_SELECTION = True
FEATURE_SELECTION_METHOD = 'chi2'  # Options: 'chi2', 'f_classif', 'mutual_info'
# Number of features to select, if None it will be determined automatically
NUM_FEATURES = None  

# Cross-validation parameters
CV_FOLDS = 5

# Evaluation metric
SCORING = 'recall'  # Options: 'precision', 'recall', 'f1', 'accuracy', 'roc_auc'

# Model hyperparameter grids
MODEL_PARAMS = {
    "Logistic Regression": {
        'model__C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'model__class_weight': [None, 'balanced', {0: 1, 1: 3}, {0: 1, 1: 5}],
        'model__solver': ['liblinear', 'saga'],
        'model__penalty': ['l1', 'l2']
    },
    "Random Forest": {
        'model__n_estimators': [100, 200, 500],
        'model__max_depth': [None, 15, 25, 35],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__class_weight': [None, 'balanced', 'balanced_subsample']
    },
    "XGBoost": {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7],
        'model__subsample': [0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.8, 0.9, 1.0],
        'model__scale_pos_weight': [1, 3, 5, 7]
    },
    "SGD": {
        "model__alpha": [0.0001, 0.001, 0.01, 0.1],
        "model__penalty": ["l2", "l1", "elasticnet"],
        "model__loss": ["log_loss"],
        "model__max_iter": [1000, 2000],
        "model__learning_rate": ["optimal", "adaptive"],
        "model__class_weight": [None, 'balanced', {0: 1, 1: 3}]
    }
}

# Top N models to use for stacking ensemble
TOP_N_MODELS = 3

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, 'visualizations', 'plots')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)